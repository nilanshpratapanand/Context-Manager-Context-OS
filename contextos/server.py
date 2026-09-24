"""ContextOS chat - a local chat app with a context store behind every conversation.

For every message the server:

  1. scores its difficulty and picks the smart or fast lane (router.py)
  2. selects only the relevant context from that conversation's store
  3. streams the reply from the first working model in the lane
  4. if a model fails - even mid-reply - builds a direction-aware handoff packet
     and continues on the next one
  5. commits the durable state the reply declared back to the store

The transcript is for display. The store is the memory.

    python -m contextos.server                # uses .env keys
    python -m contextos.server --offline      # no keys, simulated replies

Then open http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import threading
import time
import traceback
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator, Optional

from . import ContextOS, __version__
from . import router
from .budget import render
from .handoff import classify, should_migrate
from .chats import ChatStore, title_from
from .live import (FAST_ORDER, MODEL_ENV_OVERRIDE, PROVIDERS, SMART_ORDER,
                   ProviderError, load_env, stream_events, strip_reasoning)
from .units import KINDS, count_tokens

HERE = pathlib.Path(__file__).parent

# The model is asked to start its reply with a block like:
#   <context>
#   decision | /project/decisions/db | PostgreSQL 16, chosen for JSONB
#   constraint | /project/constraints/no-deps | no new runtime dependencies
#   </context>
SYSTEM = """You are a capable engineering assistant working on an ongoing task.

You are given only the RELEVANT context for this turn, not the whole conversation.
Trust it. If something you need is missing, say so plainly rather than inventing it.

A different model may take over this task at any moment and will see ONLY what you
save. So you MUST START your reply with a <context> block that saves every input
the user gave this turn (numbers, names, requirements, rules, choices) and every
result you will give. Then write your answer below it. One line per item:

kind | /address/with-hyphens | value in one line

kind is one of: decision, constraint, blocker, fact, goal.
Example - user asked "a shirt costs 800, 10% off, then 18% tax; final price?":

<context>
fact | /task/inputs/shirt-price | shirt base price is 800
constraint | /task/rules/discount | 10% discount applied first
constraint | /task/rules/tax | 18% tax applied on the discounted price
fact | /task/results/final-price | discounted 720, final price 849.60
</context>

Addresses are lowercase; the first segment must be user, project, task, agent, tool
or artifact. Reuse an existing address when updating the same item. Skip the block
only for pure small talk (greetings, thanks)."""

CONTEXT_RE = re.compile(r"<context>(.*?)</context>", re.DOTALL | re.IGNORECASE)

_RUN = re.compile(r"(\S)\1{29,}")


def is_degenerate(text: str) -> bool:
    """Free endpoints occasionally return a collapsed sample like '!!!!!!...'.
    Accepting it would put junk in front of the user instead of falling back."""
    body = CONTEXT_RE.sub("", text or "").strip()
    if not body:
        return False
    if _RUN.search(body) and len(_RUN.sub("", body).strip()) < len(body) * 0.5:
        return True
    return sum(c.isalnum() for c in body) < len(body) * 0.2


_PARTIAL_TAGS = ("<context>", "<think>", "<thinking>", "<reasoning>")


def visible_text(raw: str) -> str:
    """What the user should see of a reply that is still streaming: no <context>
    block, no reasoning, and no half-arrived tag that might become either."""
    t = strip_reasoning(CONTEXT_RE.sub("", raw or ""))
    cut = t.lower().find("<context")
    if cut >= 0:
        t = t[:cut]
    low = t.lower()
    for tag in _PARTIAL_TAGS:
        for k in range(len(tag) - 1, 0, -1):
            if low.endswith(tag[:k]):
                t = t[:-k]
                break
    return t.strip()


class Engine:
    """All the automatic behaviour lives here; the HTTP layer is a thin shell.

    Each conversation has its own ContextOS store (data/ctx/<id>.db) - its memory.
    The transcript in data/chats.db is only for display and editing.
    """

    OFFLINE_TIERS = {"offline-a": 0.85, "offline-b": 0.55, "offline-c": 0.5}

    def __init__(self, data_dir: str, env: dict[str, str], offline: bool,
                 budget: int = 1500) -> None:
        self.data = pathlib.Path(data_dir)
        (self.data / "ctx").mkdir(parents=True, exist_ok=True)
        self.chats = ChatStore(str(self.data / "chats.db"))
        self._ctxs: dict[str, ContextOS] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()
        self.env = env
        self.offline = offline
        self.budget = budget
        self.events: list[dict[str, Any]] = []
        self.forced_failures: set[str] = set()   # demo switch in the Providers panel
        self.cooldown = router.Cooldown()
        self.mode = env.get("LLM_ROUTING", "auto").strip().lower()
        self.threshold = float(env.get("LLM_ROUTE_THRESHOLD", router.DEFAULT_THRESHOLD))
        self.smart, self.fast = self._lanes()
        self.current = self.order[0] if self.order else None
        self._default: Optional[str] = None

    # ------------------------------------------------------------- providers
    def _lanes(self) -> tuple[list[str], list[str]]:
        if self.offline:
            return ["offline-a"], ["offline-b", "offline-c"]

        def lane(var: str, default: list[str]) -> list[str]:
            raw = self.env.get(var, "")
            names = [n.strip() for n in raw.split(",") if n.strip()] or default
            return [n for n in names if n in PROVIDERS and PROVIDERS[n].available(self.env)]
        return lane("LLM_SMART_ORDER", SMART_ORDER), lane("LLM_FAST_ORDER", FAST_ORDER)

    @property
    def order(self) -> list[str]:
        return self.smart + [n for n in self.fast if n not in self.smart]

    def _model(self, name: str) -> str:
        if self.offline:
            return "simulated"
        return self.env.get(MODEL_ENV_OVERRIDE.get(name, ""), PROVIDERS[name].model)

    def provider_info(self) -> list[dict[str, Any]]:
        return [{"name": n, "lane": "smart" if n in self.smart else "fast",
                 "tier": self._tier(n), "model": self._model(n),
                 "current": n == self.current, "failed": n in self.forced_failures,
                 "cooldown": self.cooldown.remaining(n)} for n in self.order]

    def _tier(self, name: str) -> float:
        return self.OFFLINE_TIERS.get(name, 0.5) if self.offline else PROVIDERS[name].tier

    def _stream(self, name: str, system: str, user: str) -> Iterator[Any]:
        """Yields answer text as str, or ("think", text) for streamed reasoning."""
        if name in self.forced_failures:
            raise ProviderError("429 rate limit exceeded (simulated for demo)")
        if self.offline:
            for word in re.split(r"(\s+)", self._offline_reply(name, user)):
                time.sleep(0.004)
                yield word
            return
        for kind, chunk in stream_events(PROVIDERS[name], system, user, self.env):
            yield chunk if kind == "text" else ("think", chunk)

    def _offline_reply(self, name: str, user: str) -> str:
        """Canned but context-aware, so the loop is demonstrable with no keys."""
        asked = user.rsplit("## Now", 1)[-1].strip() or user.strip()
        seen = re.findall(r"(/(?:user|project|task|agent|tool|artifact)/[a-z0-9/_-]+)",
                          user)[:6]
        slug = re.sub(r"[^a-z0-9]+", "-", asked.lower()).strip("-")[:24] or "note"
        return (f"<context>\nfact | /project/notes/{slug} | user asked: {asked[:120]}\n"
                f"</context>\n**[{name}]** simulated reply - no model was called.\n\n"
                f"You asked: *{asked[:300]}*\n\n"
                f"Context I was given covers: {', '.join(seen) if seen else 'nothing yet'}"
                "\n\nStart the server without `--offline` to use real providers.")

    # ---------------------------------------------------------- conversations
    def ctx_for(self, cid: str) -> ContextOS:
        with self._guard:
            if cid not in self._ctxs:
                self._ctxs[cid] = ContextOS(str(self.data / "ctx" / f"{cid}.db"))
                self._locks[cid] = threading.Lock()
            return self._ctxs[cid]

    def delete_conversation(self, cid: str) -> bool:
        with self._guard:
            ctx = self._ctxs.pop(cid, None)
            self._locks.pop(cid, None)
        if ctx:
            ctx.close()
        for suffix in ("", "-wal", "-shm"):
            (self.data / "ctx" / f"{cid}.db{suffix}").unlink(missing_ok=True)
        return self.chats.delete(cid)

    def memory(self, cid: str) -> dict[str, Any]:
        ctx = self.ctx_for(cid)
        units = [{"address": u.address, "kind": u.kind, "value": u.value,
                  "source": u.source, "tokens": u.tokens, "pinned": u.pinned,
                  "version": u.version} for u in ctx.list("", live_only=True)]
        units.sort(key=lambda d: (d["kind"], d["address"]))
        return {"units": units, "stats": ctx.stats(), "conflicts": ctx.conflicts()}

    def forget(self, cid: str, address: str) -> bool:
        return self.ctx_for(cid).delete(address)

    def export(self, cid: str) -> str:
        conv = self.chats.get(cid) or {"title": "Chat", "messages": []}
        out = [f"# {conv['title']}", ""]
        for m in conv["messages"]:
            who = "You" if m["role"] == "user" else m["meta"].get("provider", "Assistant")
            out += [f"**{who}:**", "", m["content"], ""]
        return "\n".join(out)

    # --------------------------------------------------------------- extract
    def _commit(self, text: str, source: str,
                ctx: Optional[ContextOS] = None) -> list[dict[str, Any]]:
        """Write the reply's <context> lines to the store (D2, write-time commit).
        Each item records whether it created the address, so regenerating or
        editing can undo exactly what this reply added."""
        ctx = ctx or self.ctx
        # Every block, not just the first: some models "think aloud" and mention
        # <context> before writing the real one. Malformed lines are dropped below.
        lines = [ln for blk in CONTEXT_RE.findall(text or "") for ln in blk.splitlines()]
        written = []
        for line in lines:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 3:
                continue
            kind, address, value = parts[0].lower(), parts[1], "|".join(parts[2:]).strip()
            if kind not in KINDS or not value:
                continue
            importance = {"goal": 1.0, "constraint": 0.95, "blocker": 0.85,
                          "decision": 0.9, "fact": 0.7}.get(kind, 0.5)
            try:
                existed = ctx.get(address) is not None
                u = ctx.put(address, value, kind=kind, source=source,
                            importance=importance, pinned=(kind == "goal"))
                written.append({"address": u.address, "kind": u.kind, "value": u.value,
                                "created": not existed})
            except Exception:
                continue          # a malformed address must never break the turn
        return written

    @staticmethod
    def _route_info(d: router.Decision) -> dict[str, Any]:
        return {"lane": d.lane, "difficulty": d.score, "reasons": d.reasons,
                "forced": d.forced}

    def _undo(self, cid: str, removed: list[dict[str, Any]]) -> None:
        ctx = self.ctx_for(cid)
        for m in removed:
            for w in m["meta"].get("written", []):
                if w.get("created"):
                    ctx.delete(w["address"])

    # ------------------------------------------------------------------ turn
    def chat_stream(self, cid: Optional[str], prompt: str = "", *,
                    lane: Optional[str] = None, route: Optional[str] = None,
                    regenerate: Optional[str] = None,
                    edit: Optional[str] = None) -> Iterator[dict[str, Any]]:
        """One turn as a stream of events for the UI.

        regenerate=<assistant message id> replaces that reply;
        edit=<user message id> replaces that message (and everything after it)
        with `prompt`. Either way the store writes of the dropped replies are
        undone first, so memory matches the visible conversation.
        """
        if not self.order:
            yield {"type": "error", "error": "No providers available. Add a key to "
                                             ".env, or start the server with --offline."}
            return
        if not cid or not self.chats.get(cid):
            cid = self.chats.create()["id"]
        ctx = self.ctx_for(cid)
        with self._locks[cid]:
            yield from self._turn(cid, ctx, prompt, lane, route, regenerate, edit)

    def _turn(self, cid: str, ctx: ContextOS, prompt: str, lane: Optional[str],
              route: Optional[str], regenerate: Optional[str],
              edit: Optional[str]) -> Iterator[dict[str, Any]]:
        if regenerate:
            self._undo(cid, self.chats.truncate_from(cid, regenerate))
            users = [m for m in self.chats.messages(cid) if m["role"] == "user"]
            if not users:
                yield {"type": "error", "error": "Nothing to regenerate."}
                return
            user_msg_rec = users[-1]
            prompt = user_msg_rec["content"]
        else:
            prompt = (prompt or "").strip()
            if not prompt:
                yield {"type": "error", "error": "Empty message."}
                return
            if edit:
                removed = self.chats.truncate_from(cid, edit)
                self._undo(cid, removed)
                # Editing the opening message changes what the chat is about: an
                # auto title and the goal both came from it, so both follow the edit.
                if removed and not self.chats.messages(cid):
                    old = router.parse_override(removed[0]["content"])[1]
                    if self.chats.get(cid)["title"] == title_from(old):
                        self.chats.rename(cid, "New chat")
                    ctx.delete("/task/goal")
            user_msg_rec = self.chats.add(cid, "user", prompt)

        forced, text_in = router.parse_override(prompt)
        conv = self.chats.get(cid)
        if conv["title"] == "New chat":
            self.chats.rename(cid, title_from(text_in))
            conv = self.chats.get(cid)
        yield {"type": "start", "conversation": {k: conv[k] for k in
                                                ("id", "title", "created", "updated")},
               "user_message": user_msg_rec}

        wanted = forced or (lane if lane in (router.SMART, router.FAST) else self.mode)
        decision = router.decide(text_in, wanted, self.threshold)
        yield {"type": "route", **self._route_info(decision)}

        if not ctx.get("/task/goal"):
            ctx.put("/task/goal", text_in.strip()[:400], kind="goal",
                    importance=1.0, pinned=True, source="user")

        history = [m for m in self.chats.messages(cid) if m["seq"] < user_msg_rec["seq"]]
        recent = "\n".join(f"{m['role']}: {m['content'][:400]}" for m in history[-2:])
        selection = ctx.select(text_in, budget_tokens=self.budget)
        context_text = render(selection.units) or "(nothing on file yet)"
        user_msg = (f"## Context on file\n{context_text}\n\n"
                    + (f"## Last exchange\n{recent}\n\n" if recent else "")
                    + f"## Now\n{text_in}")

        chain = router.build_chain(decision.lane, self.smart, self.fast,
                                   self.cooldown.active)
        if route in self.order:
            chain = [route] + [n for n in chain if n != route]
        attempts: list[dict[str, Any]] = []
        switched: list[dict[str, Any]] = []
        raw, shown, used, name = "", "", None, chain[0]
        t0 = time.time()

        try:
            for i, name in enumerate(chain):
                yield {"type": "model", "provider": name, "model": self._model(name),
                       "lane": "smart" if name in self.smart else "fast"}
                raw, shown, thinking, t_think = "", "", 0, None
                try:
                    for chunk in self._stream(name, SYSTEM, user_msg):
                        if isinstance(chunk, tuple):
                            thinking += len(chunk[1])
                            t_think = t_think or time.time()
                            yield {"type": "thinking", "text": chunk[1]}
                            continue
                        raw += chunk
                        vis = visible_text(raw)
                        if len(vis) >= 30 and is_degenerate(vis):
                            raise ProviderError("garbled reply (repeated characters)")
                        # Update `shown` before yielding: a Stop lands at the yield,
                        # and the partial reply saved must include this piece.
                        prev, shown = shown, vis
                        if vis.startswith(prev):
                            if len(vis) > len(prev):
                                yield {"type": "delta", "text": vis[len(prev):]}
                        else:
                            yield {"type": "replace", "text": vis}
                    if not shown.strip():
                        raise ProviderError("empty reply (the model may have spent its "
                                            "whole output budget on reasoning)")
                    used = name
                    break
                except ProviderError as exc:
                    benched = self.cooldown.hit(name, str(exc))
                    attempts.append({"provider": name, "error": str(exc)[:200],
                                     "cooldown": benched})
                    if shown:
                        yield {"type": "reset"}
                        shown = ""
                    nxt = chain[i + 1] if i + 1 < len(chain) else None
                    if nxt is None:
                        break
                    # The point of the project: rebuild the context for the model we
                    # are moving TO, in the direction we are moving.
                    direction = classify(self._tier(name), self._tier(nxt))
                    packet = ctx.handoff(direction=direction, budget_tokens=self.budget,
                                         from_model=name, to_model=nxt,
                                         difficulty=decision.score)
                    ok, why = should_migrate(direction, decision.score)
                    # The packet carries the store; the last exchange is what the
                    # failed model was also given, so the new one must not lose it.
                    user_msg = (f"{packet.render()}\n\n"
                                + (f"## Last exchange\n{recent}\n\n" if recent else "")
                                + f"## Now\n{text_in}")
                    sw = {"from": name, "to": nxt, "direction": direction,
                          "reason": str(exc)[:160], "packet_tokens": packet.tokens_selected,
                          "full_replay_tokens": packet.tokens_stored,
                          "omitted": len(packet.omitted), "gate": None if ok else why}
                    switched.append(sw)
                    self.events.append({"ts": time.time(), "kind": "switch", **sw})
                    yield {"type": "switch", **sw}
        except GeneratorExit:
            # The user pressed Stop. Keep what they saw, but commit nothing: a
            # half-written <context> block is not trustworthy state.
            if shown.strip():
                self.chats.add(cid, "assistant", shown,
                               {"provider": name, "model": self._model(name),
                                "stopped": True, **self._route_info(decision)})
            raise

        if used is None:
            self.events.append({"ts": time.time(), "kind": "all_failed",
                                "detail": attempts})
            yield {"type": "error", "error": "Every provider failed.",
                   "attempts": attempts, "switched": switched}
            return

        self.current = used
        written = self._commit(raw, used, ctx)
        if not written and decision.score > 0.05:
            # The model saved nothing. Keep the user's own words so a later
            # handoff still carries this turn's inputs.
            addr = f"/task/inputs/turn-{user_msg_rec['seq']}"
            existed = ctx.get(addr) is not None
            u = ctx.put(addr, text_in.strip()[:500], kind="fact", source="user",
                        importance=0.75)
            written = [{"address": u.address, "kind": u.kind, "value": u.value,
                        "created": not existed}]
        stats = ctx.stats()
        meta = {"provider": used, "model": self._model(used),
                "lane_used": "smart" if used in self.smart else "fast",
                **self._route_info(decision), "attempts": attempts,
                "switched": switched, "written": written,
                "context_sent": selection.addresses(),
                "thought_ms": int((time.time() - t_think) * 1000) if t_think and thinking else 0,
                "tokens": {"stored": stats["live_tokens"],
                           "sent": count_tokens(context_text),
                           "omitted_units": len(selection.omitted)},
                "ms": int((time.time() - t0) * 1000)}
        msg = self.chats.add(cid, "assistant", visible_text(raw), meta)
        yield {"type": "done", "message": msg}

    # ------------------------------------------------ non-streaming wrapper
    @property
    def ctx(self) -> ContextOS:
        """The default conversation's store, for callers without a chat id."""
        if self._default is None:
            self._default = self.chats.create()["id"]
        return self.ctx_for(self._default)

    def chat(self, prompt: str, cid: Optional[str] = None, **kw) -> dict[str, Any]:
        self.ctx                                       # ensure the default exists
        out: dict[str, Any] = {"switched": [], "attempts": []}
        for ev in self.chat_stream(cid or self._default, prompt, **kw):
            if ev["type"] == "route":
                out["route"] = {k: ev[k] for k in ("lane", "difficulty", "reasons",
                                                   "forced")}
            elif ev["type"] == "switch":
                out["switched"].append(ev)
            elif ev["type"] == "error":
                out.update(error=ev["error"], attempts=ev.get("attempts", []))
            elif ev["type"] == "done":
                m = ev["message"]
                out.update(reply=m["content"], provider=m["meta"]["provider"],
                           written=m["meta"]["written"], attempts=m["meta"]["attempts"],
                           tokens=m["meta"]["tokens"], message=m)
        return out

    # ----------------------------------------------------------------- state
    def state(self) -> dict[str, Any]:
        return {"version": __version__, "offline": self.offline,
                "providers": self.provider_info(), "events": self.events[-30:],
                "budget": self.budget,
                "routing": {"mode": self.mode, "threshold": self.threshold}}

    def toggle_failure(self, name: str) -> bool:
        if name in self.forced_failures:
            self.forced_failures.discard(name)
            self.cooldown.clear([name])      # un-failing a provider restores it at once
            return False
        self.forced_failures.add(name)
        return True

    def preview_handoff(self, cid: str, direction: str) -> dict[str, Any]:
        p = self.ctx_for(cid).handoff(direction=direction, budget_tokens=self.budget,
                                      difficulty=0.7)
        return {"markdown": p.render(), "tokens": p.tokens_selected,
                "full_replay": p.tokens_stored, "reduction": p.reduction,
                "omitted": p.omitted[:40], "notes": p.notes}

    def close(self) -> None:
        for c in self._ctxs.values():
            c.close()
        self.chats.close()


_CONV = re.compile(r"^/api/conversations/([0-9a-f]{12})(?:/([a-z]+))?$")


class Handler(BaseHTTPRequestHandler):
    engine: Engine = None            # set in serve()
    port: int = 8000
    server_version = "ContextOS"

    def log_message(self, fmt, *args):    # keep the console clean
        pass

    # ------------------------------------------------------------- plumbing
    def _send(self, code: int, body: bytes, ctype: str,
              extra: Optional[dict[str, str]] = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: Any, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json; charset=utf-8")

    def _body(self) -> dict[str, Any]:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            data = json.loads(self.rfile.read(n).decode("utf-8", "replace"))
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}

    def _trusted(self) -> bool:
        """The server holds your API keys, so only this page may drive it.
        Host blocks DNS rebinding; Origin blocks other websites posting to
        localhost; requiring JSON forces a CORS preflight we never answer."""
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
        if host not in ("127.0.0.1", "localhost"):
            return False
        origin = self.headers.get("Origin")
        if origin and origin not in (f"http://127.0.0.1:{self.port}",
                                     f"http://localhost:{self.port}"):
            return False
        if self.command == "POST":
            return (self.headers.get("Content-Type") or "").startswith("application/json")
        return True

    # ------------------------------------------------------------------ GET
    def do_GET(self) -> None:
        if not self._trusted():
            self._send(403, b"forbidden", "text/plain")
            return
        path, _, query = self.path.partition("?")
        eng = self.engine
        try:
            if path in ("/", "/index.html"):
                page = HERE / "dashboard.html"
                self._send(200, page.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/state":
                self._json(eng.state())
            elif path == "/api/conversations":
                q = urllib.parse.parse_qs(query).get("q", [""])[0]
                self._json({"conversations": eng.chats.list(q)})
            elif m := _CONV.match(path):
                cid, action = m.groups()
                if not eng.chats.get(cid):
                    self._json({"error": "no such conversation"}, 404)
                elif action is None:
                    self._json(eng.chats.get(cid))
                elif action == "memory":
                    self._json(eng.memory(cid))
                elif action == "export":
                    title = re.sub(r"[^\w -]", "", eng.chats.get(cid)["title"])[:40]
                    self._send(200, eng.export(cid).encode(), "text/markdown; charset=utf-8",
                               {"Content-Disposition":
                                f'attachment; filename="{title or "chat"}.md"'})
                else:
                    self._json({"error": "not found"}, 404)
            else:
                self._send(404, b"not found", "text/plain")
        except Exception as exc:
            traceback.print_exc()
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    # ----------------------------------------------------------------- POST
    def do_POST(self) -> None:
        if not self._trusted():
            self._json({"error": "forbidden"}, 403)
            return
        eng = self.engine
        try:
            body = self._body()
            if self.path == "/api/chat":
                self._stream_chat(body)
            elif self.path == "/api/conversations":
                self._json(eng.chats.create())
            elif m := _CONV.match(self.path):
                cid, action = m.groups()
                if not eng.chats.get(cid):
                    self._json({"error": "no such conversation"}, 404)
                elif action == "rename":
                    eng.chats.rename(cid, str(body.get("title", "")))
                    self._json(eng.chats.get(cid))
                elif action == "delete":
                    self._json({"deleted": eng.delete_conversation(cid)})
                elif action == "forget":
                    self._json({"forgotten": eng.forget(cid, str(body.get("address", "")))})
                elif action == "handoff":
                    self._json(eng.preview_handoff(cid, body.get("direction", "escalate")))
                else:
                    self._json({"error": "not found"}, 404)
            elif self.path == "/api/fail":
                self._json({"failing": eng.toggle_failure(str(body.get("provider", "")))})
            elif self.path == "/api/check":
                from .live import check as live_check
                self._json({"offline": eng.offline,
                            "rows": [] if eng.offline else live_check(eng.env)})
            else:
                self._json({"error": "not found"}, 404)
        except Exception as exc:
            traceback.print_exc()
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def _stream_chat(self, body: dict[str, Any]) -> None:
        """One JSON event per line. The connection closes when the turn ends
        (HTTP/1.0), so no chunked encoding is needed. If the browser goes away -
        the user pressed Stop - the write fails and the turn is closed, which
        saves the partial reply."""
        gen = self.engine.chat_stream(
            body.get("conversation_id") or None, str(body.get("prompt") or ""),
            lane=body.get("lane"), route=body.get("route"),
            regenerate=body.get("regenerate"), edit=body.get("edit"))
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        try:
            for ev in gen:
                self.wfile.write((json.dumps(ev) + "\n").encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        finally:
            gen.close()


def serve(data: str, port: int, offline: bool, budget: int, env_path: str,
          open_browser: bool = True) -> None:
    env = load_env(env_path)
    if not offline and not any(p.available(env) for p in PROVIDERS.values()):
        print("No provider keys found in .env - starting in offline mode instead.")
        offline = True
    # Simulated chats never mix with real ones.
    data_dir = str(pathlib.Path(data) / "offline") if offline else data
    engine = Engine(data_dir, env, offline, budget)
    Handler.engine, Handler.port = engine, port

    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    httpd.daemon_threads = True
    url = f"http://127.0.0.1:{port}"
    print("=" * 62)
    print(f"  ContextOS chat  {url}")
    if engine.offline:
        print("  Mode: OFFLINE (simulated replies)")
    else:
        print(f"  Smart lane: {', '.join(engine.smart) or '(none)'}")
        print(f"  Fast lane:  {', '.join(engine.fast) or '(none)'}")
        print(f"  Routing: {engine.mode}, threshold {engine.threshold}")
    print(f"  Chats saved in: {pathlib.Path(data_dir).resolve()}")
    print("=" * 62)
    print("  Press Ctrl+C to stop.")
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        httpd.server_close()
        engine.close()


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="ContextOS chat")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--data", default="chat_data",
                    help="folder for conversations and their context stores")
    ap.add_argument("--env", default=".env")
    ap.add_argument("--budget", type=int, default=1500,
                    help="tokens of context sent per turn")
    ap.add_argument("--offline", action="store_true",
                    help="no API calls - simulated replies, full UI")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)
    serve(a.data, a.port, a.offline, a.budget, a.env, not a.no_browser)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
