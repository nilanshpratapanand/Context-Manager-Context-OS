"""ContextOS dashboard - a local web app that does the whole loop automatically.

You type a prompt. Behind it, for every turn, the server:

  1. selects only the relevant context from the store (not the whole history)
  2. calls the current provider with that context
  3. if the provider fails or is rate-limited, classifies the direction, builds a
     handoff packet, switches provider, and retries - the task continues
  4. extracts durable state from the reply and commits it back to the store

The conversation history is deliberately NOT the memory. The store is. History is
capped at a few turns; everything that matters is written to an address.

    python -m contextos.server                # uses .env keys
    python -m contextos.server --offline      # no keys, canned replies, full UI

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
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional

from . import ContextOS, __version__
from . import router
from .budget import render
from .handoff import classify, should_migrate
from .live import (FAST_ORDER, MODEL_ENV_OVERRIDE, PROVIDERS, SMART_ORDER,
                   ProviderError, complete, load_env)
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

# Errors that mean "this provider is done for now, move to another one".
SWITCH_SIGNALS = ("429", "rate limit", "rate_limit", "quota", "insufficient",
                  "capacity", "overloaded", "503", "502", "500", "timed out",
                  "network", "unauthorized", "401", "403")


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


def wants_switch(msg: str) -> bool:
    m = (msg or "").lower()
    return any(s in m for s in SWITCH_SIGNALS)


class Engine:
    """All the automatic behaviour lives here; the HTTP layer is a thin shell."""

    def __init__(self, db: str, env: dict[str, str], offline: bool,
                 budget: int = 1500) -> None:
        self.ctx = ContextOS(db)
        self.env = env
        self.offline = offline
        self.budget = budget
        self.lock = threading.Lock()
        self.history: list[dict[str, str]] = []
        self.events: list[dict[str, Any]] = []
        self.forced_failures: set[str] = set()   # demo button
        self.cooldown = router.Cooldown()
        self.mode = env.get("LLM_ROUTING", "auto").strip().lower()
        self.threshold = float(env.get("LLM_ROUTE_THRESHOLD", router.DEFAULT_THRESHOLD))
        self.smart, self.fast = self._lanes()
        self.current = self.order[0] if self.order else None
        self.turns = 0

    # ------------------------------------------------------------- providers
    OFFLINE_TIERS = {"offline-a": 0.85, "offline-b": 0.55, "offline-c": 0.5}

    def _lanes(self) -> tuple[list[str], list[str]]:
        if self.offline:
            return ["offline-a"], ["offline-b", "offline-c"]

        def lane(var: str, default: list[str]) -> list[str]:
            raw = self.env.get(var, "")
            names = [n.strip() for n in raw.split(",") if n.strip()] or default
            return [n for n in names if n in PROVIDERS and PROVIDERS[n].available(self.env)]
        return lane("LLM_SMART_ORDER", SMART_ORDER), lane("LLM_FAST_ORDER", FAST_ORDER)

    def go_offline(self) -> None:
        self.offline = True
        self.smart, self.fast = self._lanes()
        self.current = self.order[0]

    @property
    def order(self) -> list[str]:
        return self.smart + [n for n in self.fast if n not in self.smart]

    def provider_info(self) -> list[dict[str, Any]]:
        out = []
        for name in self.order:
            lane = "smart" if name in self.smart else "fast"
            info = {"name": name, "lane": lane, "tier": self._tier(name),
                    "current": name == self.current,
                    "failed": name in self.forced_failures,
                    "cooldown": self.cooldown.remaining(name)}
            if self.offline:
                info["model"] = "simulated"
            else:
                info["model"] = self.env.get(MODEL_ENV_OVERRIDE.get(name, ""),
                                             PROVIDERS[name].model)
            out.append(info)
        return out

    def _tier(self, name: str) -> float:
        return self.OFFLINE_TIERS.get(name, 0.5) if self.offline else PROVIDERS[name].tier

    def _call(self, name: str, system: str, user: str) -> tuple[str, int]:
        if name in self.forced_failures:
            raise ProviderError("429 rate limit exceeded (simulated for demo)")
        if self.offline:
            return self._offline_reply(name, user), count_tokens(system + user)
        # Reasoning models (Qwen and friends) spend output tokens on a <think>
        # block that complete() strips. Too small a cap and the whole budget goes
        # to thinking, leaving an empty answer - so this is deliberately generous.
        text, used = complete(PROVIDERS[name], system, user, self.env, max_tokens=1600)
        if not text.strip():
            raise ProviderError("empty reply (the model may have spent its whole "
                                "output budget on reasoning)")
        if is_degenerate(text):
            raise ProviderError("garbled reply (repeated characters) - treating as a failure")
        return text, used

    def _offline_reply(self, name: str, user: str) -> str:
        """Canned but context-aware, so the loop is demonstrable with no keys."""
        goal = self.ctx.get("/task/goal")
        seen = [u.address for u in self.ctx.list("", live_only=True)][:6]
        # `user` is the assembled prompt (context + question). Only the question
        # itself should name the address, or the store fills with junk addresses.
        asked = user.rsplit("## Now", 1)[-1].strip() or user.strip()
        body = (f"[{name}] Working on it.\n\n"
                f"Goal on file: {goal.value if goal else 'not set yet'}\n"
                f"Context I was given covers: {', '.join(seen) if seen else 'nothing yet'}\n\n"
                f"You asked: {asked[:300]}\n\n"
                "This is the offline simulator - no model was called. Start the server "
                "without --offline to use your real providers.")
        slug = re.sub(r"[^a-z0-9]+", "-", asked.lower()).strip("-")[:24] or "note"
        return body + (f"\n\n<context>\nfact | /project/notes/{slug} | "
                       f"user asked about: {user.strip()[:120]}\n</context>")

    # --------------------------------------------------------------- extract
    def _commit(self, text: str, source: str) -> list[dict[str, str]]:
        """Pull the <context> block out of the reply and write it to the store.
        This is D2 - write-time commit - done for the model rather than by it."""
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
                u = self.ctx.put(address, value, kind=kind, source=source,
                                 importance=importance,
                                 pinned=(kind == "goal"))
                written.append({"address": u.address, "kind": u.kind, "value": u.value})
            except Exception:
                continue          # a malformed address must never break the turn
        return written

    @staticmethod
    def strip_block(text: str) -> str:
        return CONTEXT_RE.sub("", text or "").strip()

    # ------------------------------------------------------------------ turn
    def chat(self, prompt: str) -> dict[str, Any]:
        with self.lock:
            return self._chat(prompt)

    def _chat(self, prompt: str) -> dict[str, Any]:
        if not self.order:
            return {"error": "No providers available. Add a key to .env, or start the "
                             "server with --offline."}
        forced, prompt = router.parse_override(prompt)
        decision = router.decide(prompt, forced or self.mode, self.threshold)
        self.turns += 1
        if self.turns == 1 and not self.ctx.get("/task/goal"):
            self.ctx.put("/task/goal", prompt.strip()[:400], kind="goal",
                         importance=1.0, pinned=True, source="user")

        selection = self.ctx.select(prompt, budget_tokens=self.budget)
        context_text = render(selection.units) or "(nothing on file yet)"
        recent = "\n".join(f"{h['role']}: {h['text'][:400]}" for h in self.history[-2:])

        user_msg = (f"## Context on file\n{context_text}\n\n"
                    + (f"## Last exchange\n{recent}\n\n" if recent else "")
                    + f"## Now\n{prompt}")

        attempts: list[dict[str, Any]] = []
        switched: list[dict[str, Any]] = []
        text, used_provider, prompt_tokens = "", None, 0

        chain = router.build_chain(decision.lane, self.smart, self.fast,
                                   self.cooldown.active)
        difficulty = decision.score

        for i, name in enumerate(chain):
            try:
                text, prompt_tokens = self._call(name, SYSTEM, user_msg)
                used_provider = name
                attempts.append({"provider": name, "ok": True})
                break
            except ProviderError as exc:
                benched = self.cooldown.hit(name, str(exc))
                attempts.append({"provider": name, "ok": False, "error": str(exc)[:200],
                                 "cooldown": benched})
                if not wants_switch(str(exc)) and i == len(chain) - 1:
                    break
                nxt = chain[i + 1] if i + 1 < len(chain) else None
                if nxt is None:
                    break
                # This is the whole point of the project: rebuild the context for the
                # model we are moving TO, in the direction we are moving.
                direction = classify(self._tier(name), self._tier(nxt))
                packet = self.ctx.handoff(direction=direction, budget_tokens=self.budget,
                                          from_model=name, to_model=nxt,
                                          difficulty=difficulty)
                ok, why = should_migrate(direction, difficulty)
                # The packet carries the store; the last exchange is what the
                # failed model was also given, so the new one must not lose it.
                user_msg = (f"{packet.render()}\n\n"
                            + (f"## Last exchange\n{recent}\n\n" if recent else "")
                            + f"## Now\n{prompt}")
                switched.append({
                    "from": name, "to": nxt, "direction": direction,
                    "reason": str(exc)[:160],
                    "packet_tokens": packet.tokens_selected,
                    "full_replay_tokens": packet.tokens_stored,
                    "omitted": len(packet.omitted),
                    "gate": None if ok else why,
                })

        if used_provider is None:
            self.events.append({"ts": time.time(), "kind": "all_failed",
                                "detail": attempts})
            return {"error": "Every provider failed.", "attempts": attempts,
                    "switched": switched, "route": self._route_info(decision)}

        self.current = used_provider
        written = self._commit(text, used_provider)
        if not written and decision.score > 0.05:
            # The model saved nothing. Keep the user's own words so a later
            # handoff still carries this turn's inputs.
            u = self.ctx.put(f"/task/inputs/turn-{self.turns}", prompt.strip()[:500],
                             kind="fact", source="user", importance=0.75)
            written = [{"address": u.address, "kind": u.kind, "value": u.value}]
        reply = self.strip_block(text)

        self.history.append({"role": "user", "text": prompt})
        self.history.append({"role": "assistant", "text": reply})
        self.history = self.history[-6:]        # history is not the memory, the store is

        for s in switched:
            self.events.append({"ts": time.time(), "kind": "switch", **s})

        stats = self.ctx.stats()
        return {
            "reply": reply,
            "provider": used_provider,
            "route": self._route_info(decision),
            "attempts": attempts,
            "switched": switched,
            "written": written,
            "context_sent": selection.addresses(),
            "tokens": {
                "stored": stats["live_tokens"],
                "sent": count_tokens(context_text),
                "prompt": prompt_tokens,
                "omitted_units": len(selection.omitted),
            },
        }

    @staticmethod
    def _route_info(d: router.Decision) -> dict[str, Any]:
        return {"lane": d.lane, "difficulty": d.score, "reasons": d.reasons,
                "forced": d.forced}

    # ----------------------------------------------------------------- state
    def state(self) -> dict[str, Any]:
        units = [{
            "address": u.address, "kind": u.kind, "value": u.value,
            "source": u.source, "importance": u.importance, "tokens": u.tokens,
            "pinned": u.pinned, "version": u.version,
        } for u in self.ctx.list("", live_only=True)]
        units.sort(key=lambda d: (d["kind"], d["address"]))
        return {
            "version": __version__,
            "offline": self.offline,
            "units": units,
            "stats": self.ctx.stats(),
            "providers": self.provider_info(),
            "conflicts": self.ctx.conflicts(),
            "events": self.events[-30:],
            "budget": self.budget,
            "turns": self.turns,
            "routing": {"mode": self.mode, "threshold": self.threshold},
        }

    def reset(self) -> None:
        with self.lock:
            for u in self.ctx.list("", live_only=True):
                self.ctx.delete(u.address)
            self.history.clear()
            self.events.clear()
            self.forced_failures.clear()
            self.cooldown.clear()
            self.turns = 0
            self.current = self.order[0] if self.order else None

    def toggle_failure(self, name: str) -> bool:
        if name in self.forced_failures:
            self.forced_failures.discard(name)
            self.cooldown.clear([name])      # un-failing a provider restores it at once
            return False
        self.forced_failures.add(name)
        return True

    def preview_handoff(self, direction: str) -> dict[str, Any]:
        p = self.ctx.handoff(direction=direction, budget_tokens=self.budget,
                             difficulty=0.7)
        return {"markdown": p.render(), "tokens": p.tokens_selected,
                "full_replay": p.tokens_stored, "reduction": p.reduction,
                "omitted": p.omitted[:40], "notes": p.notes}


class Handler(BaseHTTPRequestHandler):
    engine: Engine = None            # set in serve()
    server_version = "ContextOS"

    def log_message(self, fmt, *args):    # keep the console clean
        pass

    # ------------------------------------------------------------- plumbing
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: Any, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json; charset=utf-8")

    def _body(self) -> dict[str, Any]:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8", "replace"))
        except json.JSONDecodeError:
            return {}

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            page = HERE / "dashboard.html"
            if not page.exists():
                self._send(500, b"dashboard.html is missing", "text/plain")
                return
            self._send(200, page.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/state":
            self._json(self.engine.state())
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        try:
            if self.path == "/api/chat":
                prompt = (self._body().get("prompt") or "").strip()
                if not prompt:
                    self._json({"error": "empty prompt"}, 400)
                    return
                self._json(self.engine.chat(prompt))
            elif self.path == "/api/reset":
                self.engine.reset()
                self._json({"ok": True})
            elif self.path == "/api/fail":
                name = self._body().get("provider", "")
                self._json({"failing": self.engine.toggle_failure(name)})
            elif self.path == "/api/check":
                from .live import check as live_check
                if self.engine.offline:
                    self._json({"offline": True, "rows": []})
                else:
                    self._json({"offline": False,
                                "rows": live_check(self.engine.env)})
            elif self.path == "/api/handoff":
                d = self._body().get("direction", "escalate")
                self._json(self.engine.preview_handoff(d))
            else:
                self._json({"error": "not found"}, 404)
        except Exception as exc:
            traceback.print_exc()
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)


def serve(db: str, port: int, offline: bool, budget: int, env_path: str,
          open_browser: bool = True) -> None:
    env = load_env(env_path)
    engine = Engine(db, env, offline, budget)
    Handler.engine = engine

    if not engine.order:
        print("No provider keys found in .env - starting in offline mode instead.")
        engine.go_offline()

    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"
    print("=" * 62)
    print(f"  ContextOS dashboard  {url}")
    if engine.offline:
        print("  Mode: OFFLINE (simulated replies)")
    else:
        print(f"  Smart lane: {', '.join(engine.smart) or '(none)'}")
        print(f"  Fast lane:  {', '.join(engine.fast) or '(none)'}")
        print(f"  Routing: {engine.mode}, threshold {engine.threshold}")
    print(f"  Store: {db}   context budget: {budget} tokens")
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
        engine.ctx.close()


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="ContextOS dashboard")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--db", default="dashboard.db")
    ap.add_argument("--env", default=".env")
    ap.add_argument("--budget", type=int, default=1500,
                    help="tokens of context sent per turn")
    ap.add_argument("--offline", action="store_true",
                    help="no API calls - simulated replies, full UI")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)
    serve(a.db, a.port, a.offline, a.budget, a.env, not a.no_browser)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
