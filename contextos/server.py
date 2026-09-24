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
from .budget import render
from .handoff import classify, should_migrate
from .live import PROVIDERS, MODEL_ENV_OVERRIDE, Provider, ProviderError, complete, load_env
from .units import KINDS, count_tokens

HERE = pathlib.Path(__file__).parent

# The model is asked to end its reply with a block like:
#   <context>
#   decision | /project/decisions/db | PostgreSQL 16, chosen for JSONB
#   constraint | /project/constraints/no-deps | no new runtime dependencies
#   </context>
SYSTEM = """You are a capable engineering assistant working on an ongoing task.

You are given only the RELEVANT context for this turn, not the whole conversation.
Trust it. If something you need is missing, say so plainly rather than inventing it.

After your reply, if this turn established anything durable - a decision, a rule or
constraint, a blocker, a goal, or a fact worth keeping - append a block exactly like:

<context>
decision | /project/decisions/short-name | the decision, in one line
constraint | /project/constraints/short-name | the rule that must hold
blocker | /task/blockers/short-name | what is blocking progress
fact | /project/area/short-name | the established fact
goal | /task/goal | what we are trying to achieve
</context>

Rules for that block: lowercase addresses, words separated by hyphens, roots must be
one of user/project/task/agent/tool/artifact. Omit the block entirely if this turn
established nothing durable. Never put chit-chat in it."""

CONTEXT_RE = re.compile(r"<context>(.*?)</context>", re.DOTALL | re.IGNORECASE)

# Errors that mean "this provider is done for now, move to another one".
SWITCH_SIGNALS = ("429", "rate limit", "rate_limit", "quota", "insufficient",
                  "capacity", "overloaded", "503", "502", "500", "timed out",
                  "network", "unauthorized", "401", "403")


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
        self.order = self._provider_order()
        self.current = self.order[0] if self.order else None
        self.turns = 0

    # ------------------------------------------------------------- providers
    def _provider_order(self) -> list[str]:
        if self.offline:
            return ["offline-a", "offline-b", "offline-c"]
        explicit = self.env.get("LLM_PROVIDER_ORDER", "")
        names = [n.strip() for n in explicit.split(",") if n.strip()] if explicit else \
            [n for n in PROVIDERS if PROVIDERS[n].available(self.env)]
        return [n for n in names if n in PROVIDERS and PROVIDERS[n].available(self.env)] \
            if not self.offline else names

    def provider_info(self) -> list[dict[str, Any]]:
        out = []
        for name in self.order:
            if self.offline:
                out.append({"name": name, "model": "simulated", "tier": 0.5,
                            "current": name == self.current,
                            "failed": name in self.forced_failures})
                continue
            p = PROVIDERS[name]
            out.append({
                "name": name,
                "model": self.env.get(MODEL_ENV_OVERRIDE.get(name, ""), p.model),
                "tier": p.tier,
                "current": name == self.current,
                "failed": name in self.forced_failures,
            })
        return out

    def _tier(self, name: str) -> float:
        return 0.5 if self.offline else PROVIDERS[name].tier

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
        m = CONTEXT_RE.search(text or "")
        if not m:
            return []
        written = []
        for line in m.group(1).splitlines():
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

        start_at = self.order.index(self.current) if self.current in self.order else 0
        chain = self.order[start_at:] + self.order[:start_at]

        for i, name in enumerate(chain):
            try:
                text, prompt_tokens = self._call(name, SYSTEM, user_msg)
                used_provider = name
                attempts.append({"provider": name, "ok": True})
                break
            except ProviderError as exc:
                attempts.append({"provider": name, "ok": False, "error": str(exc)[:200]})
                if not wants_switch(str(exc)) and i == len(chain) - 1:
                    break
                nxt = chain[i + 1] if i + 1 < len(chain) else None
                if nxt is None:
                    break
                # This is the whole point of the project: rebuild the context for the
                # model we are moving TO, in the direction we are moving.
                direction = classify(self._tier(name), self._tier(nxt))
                packet = self.ctx.handoff(direction=direction, budget_tokens=self.budget,
                                          from_model=name, to_model=nxt, difficulty=0.7)
                ok, why = should_migrate(direction, 0.7)
                user_msg = (f"{packet.render()}\n\n## Now\n{prompt}")
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
                    "switched": switched}

        self.current = used_provider
        written = self._commit(text, used_provider)
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
        }

    def reset(self) -> None:
        with self.lock:
            for u in self.ctx.list("", live_only=True):
                self.ctx.delete(u.address)
            self.history.clear()
            self.events.clear()
            self.forced_failures.clear()
            self.turns = 0
            self.current = self.order[0] if self.order else None

    def toggle_failure(self, name: str) -> bool:
        if name in self.forced_failures:
            self.forced_failures.discard(name)
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
        engine.offline = True
        engine.order = ["offline-a", "offline-b", "offline-c"]
        engine.current = engine.order[0]

    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"
    mode = "OFFLINE (simulated replies)" if engine.offline else \
        "LIVE: " + ", ".join(engine.order)
    print("=" * 62)
    print(f"  ContextOS dashboard  {url}")
    print(f"  Mode: {mode}")
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
