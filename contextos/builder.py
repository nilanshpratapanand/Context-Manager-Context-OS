"""Build mode: goal -> research -> plan -> (your approval) -> build feature by
feature, each verified -> security review -> report.

Each phase uses the lane that suits it: reading and summarising the web on the
fast lane, planning, coding and security work on the smart lane. The engine, not
the model, decides whether a feature works: it runs the feature's test command
itself after the agent says it's done, and re-runs every earlier feature's tests
so a new feature can't silently break an old one.

Safety: file tools can't leave the workspace; any command other than the plan's
own test commands needs approval (those can be auto-approved); untrusted MCP
tools need approval; web content is treated as data.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

from . import ContextOS
from .agent import Agent, ModelPool, remember_tool
from .budget import render
from .live import ProviderError
from .skills import catalog, discover, skills_tool
from .tools import Tool, ToolError, Workspace, builtin_tools, clip, security_scan

RESEARCH_TOOLS = ("web_search", "fetch_url", "wikipedia", "arxiv_search", "arxiv_read")
BUILD_TOOLS = ("list_files", "read_file", "search_code", "write_file", "edit_file",
               "run_command", "security_scan")


def check_workspace(path: str, app_root: Optional[str] = None) -> Path:
    """Refuse folders where a mistaken write would do real damage."""
    p = Path(path).expanduser().resolve()
    home = Path.home().resolve()
    bad = {home, Path(p.anchor)}
    for env in ("SystemRoot", "ProgramFiles", "ProgramFiles(x86)", "windir"):
        if os.environ.get(env):
            bad.add(Path(os.environ[env]).resolve())
    for d in ("/", "/bin", "/etc", "/usr", "/var", "/System", "/Applications"):
        bad.add(Path(d).resolve())
    if p in bad or p in (home / "Desktop", home / "Documents", home / "Downloads"):
        raise ToolError(f"{p} is too broad - pick or create a dedicated project folder")
    if app_root and (p == Path(app_root).resolve() or Path(app_root).resolve() in p.parents
                     and "chat_data" not in p.parts):
        raise ToolError("the workspace can't be inside ContextOS's own folder")
    return p


def _json_obj(text: str) -> dict[str, Any]:
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj = dec.raw_decode(text, i)[0]
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON object in the reply")


def _norm(cmd: str) -> str:
    return re.sub(r"\s+", " ", cmd.strip())


# The plan is model-written, and research reads untrusted web pages, so a test
# command is only run unasked if it is plainly a test runner with no shell
# chaining, redirection or substitution. Anything else asks, plan or not.
_SAFE_TEST = re.compile(
    r"^(python3?|py)( -[0-9.]+)? -m (unittest|pytest)( [\w./:=,@-]+)*$"
    r"|^pytest( [\w./:=,@-]+)*$"
    r"|^(npm|pnpm|yarn) (run )?test( [\w./:=,@-]+)*$"
    r"|^node --test( [\w./:=,@-]+)*$"
    r"|^go test( [\w./:=,@-]+)*$"
    r"|^cargo test( [\w./:=,@-]+)*$")


def safe_test_command(cmd: str) -> bool:
    return bool(_SAFE_TEST.match(_norm(cmd)))


def fix_hint(output: str) -> str:
    """Name the cause for failure patterns seen in real builds with free models."""
    if re.search(r"NO TESTS RAN|Ran 0 tests|no tests ran|collected 0 items", output):
        return ("Likely cause: the test file contains no tests. Tests must be methods named "
                "test_* on a unittest.TestCase subclass, and must really check the feature - "
                "an empty or placeholder test does not count.")
    if re.search(r"FileNotFoundError|No such file|does not exist", output):
        return ("Likely cause: a test reads an input file that doesn't exist. Tests must "
                "create their own input files, e.g. write them into a "
                "tempfile.TemporaryDirectory() in setUp, and pass that path.")
    if re.search(r"ModuleNotFoundError|ImportError", output):
        return ("Likely cause: an import path. Tests run from the project folder, so import "
                "the package by its folder name, and make sure it has an __init__.py.")
    return ""


class BuildRun:
    def __init__(self, goal: str, workspace: str, env: dict[str, str], *,
                 pool: Optional[ModelPool] = None, connectors: Any = None,
                 auto_approve_tests: bool = True, research: bool = True,
                 review_plan: bool = True, max_features: int = 6,
                 app_root: Optional[str] = None) -> None:
        self.id = uuid.uuid4().hex[:10]
        self.goal = goal.strip()
        self.ws = Workspace(str(check_workspace(workspace, app_root)))
        self.pool = pool or ModelPool(env)
        self.connectors = connectors
        self.auto_tests, self.do_research, self.review = auto_approve_tests, research, review_plan
        self.max_features = max(1, min(max_features, 12))
        self.ctx = ContextOS(str(self.ws.root / ".contextos_memory.db"))
        self.events: list[dict[str, Any]] = []
        self.cond = threading.Condition()
        self.stop = threading.Event()
        self.pending: dict[str, dict[str, Any]] = {}
        self.plan: dict[str, Any] = {}
        self.results: list[dict[str, Any]] = []
        self.status = "created"
        self.started = time.time()
        self.thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------ events
    def emit(self, ev: dict[str, Any]) -> None:
        with self.cond:
            ev = {"seq": len(self.events), "ts": round(time.time(), 2), **ev}
            self.events.append(ev)
            self.cond.notify_all()

    def events_after(self, n: int, timeout: float = 25) -> list[dict[str, Any]]:
        with self.cond:
            if len(self.events) <= n and self.status not in ("done", "failed", "stopped"):
                self.cond.wait(timeout)
            return self.events[n:]

    def summary(self) -> dict[str, Any]:
        return {"id": self.id, "goal": self.goal, "workspace": str(self.ws.root),
                "status": self.status, "started": self.started,
                "features": [{"name": r["name"], "passed": r["passed"]} for r in self.results],
                "waiting": [p["kind"] for p in self.pending.values()]}

    # --------------------------------------------------------- approvals
    def _wait_for(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        aid = uuid.uuid4().hex[:8]
        slot = {"kind": kind, "event": threading.Event(), "answer": None}
        self.pending[aid] = slot
        self.emit({"type": kind, "id": aid, **payload})
        while not slot["event"].wait(0.5):
            if self.stop.is_set():
                self.pending.pop(aid, None)
                return {"allow": False, "stopped": True}
        self.pending.pop(aid, None)
        return slot["answer"] or {}

    def answer(self, aid: str, answer: dict[str, Any]) -> bool:
        slot = self.pending.get(aid)
        if not slot:
            return False
        slot["answer"] = answer
        slot["event"].set()
        self.emit({"type": "answered", "id": aid, "allow": bool(answer.get("allow"))})
        return True

    def _test_commands(self) -> set[str]:
        cmds = {_norm(f.get("test_command", "")) for f in self.plan.get("features", [])}
        cmds.add(_norm(self.plan.get("test_all", "")))
        return {c for c in cmds if c}

    def approve(self, tool: Tool, args: dict[str, Any]) -> bool:
        cmd = _norm(str(args.get("command", "")))
        if tool.name == "run_command" and self.auto_tests and \
                cmd in self._test_commands() and safe_test_command(cmd):
            self.emit({"type": "auto_approved", "command": args.get("command")})
            return True
        what = args.get("command") if tool.name == "run_command" else \
            f"{tool.name} {json.dumps(args)[:300]}"
        ans = self._wait_for("approval", {"tool": tool.name, "action": what,
                                          "risk": tool.risk})
        return bool(ans.get("allow"))

    # ------------------------------------------------------------- tools
    def _tools(self, names: tuple[str, ...], extra_mcp: bool) -> dict[str, Tool]:
        base = builtin_tools(self.ws)
        tools = {n: base[n] for n in names}
        sk = discover(str(self.ws.root / "skills"), "skills")
        tools["use_skill"] = skills_tool(sk)
        tools["remember"] = remember_tool(self.ctx)
        if extra_mcp and self.connectors:
            for t in self.connectors.tools():
                tools[t.name] = t
        return tools

    def _agent(self, names: tuple[str, ...], mcp: bool = True) -> Agent:
        sk = discover(str(self.ws.root / "skills"), "skills")
        return Agent(self.pool, self._tools(names, mcp), self.ctx, emit=self.emit,
                     approve=self.approve, stop=self.stop, skills_catalog=catalog(sk))

    def _run_cmd(self, command: str) -> tuple[bool, str]:
        """The engine's own verification run."""
        tool = builtin_tools(self.ws)["run_command"]
        if not self.approve(tool, {"command": command}):
            return False, "verification run was not approved"
        try:
            out = self.ws.run_command(command, "300")
        except ToolError as e:
            return False, str(e)
        return out.startswith("exit code 0"), out

    # ------------------------------------------------------------- phases
    def start(self) -> None:
        self.thread = threading.Thread(target=self._main, daemon=True)
        self.thread.start()

    def _phase(self, name: str, detail: str = "") -> None:
        self.emit({"type": "phase", "phase": name, "detail": detail})

    def _main(self) -> None:
        self.status = "running"
        try:
            if not self.pool.ready():
                raise ProviderError("no models configured - add an API key to .env")
            self.ctx.put("/task/goal", self.goal, kind="goal", pinned=True, importance=1.0)
            brief = self._research() if self.do_research else ""
            if self.stop.is_set():
                raise InterruptedError
            self._plan(brief)
            if self.review:
                self._phase("review", "Waiting for you to approve the plan")
                ans = self._wait_for("plan_review", {"plan": self.plan})
                if not ans.get("allow"):
                    raise InterruptedError
                if isinstance(ans.get("plan"), dict) and ans["plan"].get("features"):
                    self.plan = self._clean_plan(ans["plan"])
                    self.emit({"type": "plan", "plan": self.plan, "edited": True})
            self._build()
            if self.stop.is_set():
                raise InterruptedError
            sec = self._security()
            self._report(brief, sec)
            self.status = "done"
            self.emit({"type": "done", "passed": sum(r["passed"] for r in self.results),
                       "total": len(self.results),
                       "report": "BUILD_REPORT.md"})
        except InterruptedError:
            self.status = "stopped"
            self.emit({"type": "stopped"})
        except Exception as e:                   # surface, don't die silently
            self.status = "failed"
            self.emit({"type": "failed", "error": f"{type(e).__name__}: {e}"})
        finally:
            with self.cond:
                self.cond.notify_all()

    def _research(self) -> str:
        self._phase("research", "Searching the web, docs and papers")
        ag = self._agent(RESEARCH_TOOLS, mcp=False)
        r = ag.run(
            f"Research how to build this, before any code is written:\n\n{self.goal}\n\n"
            "Answer: the standard approaches, which fits best, libraries or formats to use "
            "(prefer the Python standard library), known pitfalls, and how to test it. "
            "Load the web-research skill first. Save each important finding with remember, "
            "including its source URL. Keep it focused: about 4-8 searches or reads. "
            "Finish with a research brief of 5-10 bullet points.",
            "You are a meticulous research analyst preparing a software build.",
            lane="fast", max_steps=14)
        brief = r.summary if r.status == "done" else \
            "Research was incomplete: " + r.summary
        self.ctx.put("/project/research/brief", brief[:3000], kind="fact", importance=0.9)
        self.emit({"type": "research", "status": r.status, "brief": brief})
        return brief

    PLAN_PROMPT = """Plan the build as JSON only, in this exact shape:
{"summary": "one paragraph: what will be built and the main design decisions",
 "stack": "language, libraries, storage",
 "run_command": "how a user runs the finished program",
 "test_all": "one command that runs every test",
 "features": [
   {"name": "short name",
    "description": "what it does, precisely",
    "files": ["paths it creates or changes"],
    "test_command": "command that runs ONLY this feature's tests",
    "acceptance": "what the tests must prove"}
 ]}
Rules: at most {max} features, ordered so each builds on the previous; the first sets up
the project skeleton. Every feature must be testable by its test_command alone, with no
network and no user input. Prefer the standard library. For Python use unittest, e.g.
"python -m unittest tests.test_storage -v" and test_all "python -m unittest discover -s tests -v"."""

    def _clean_plan(self, plan: dict[str, Any]) -> dict[str, Any]:
        feats = [f for f in plan.get("features", []) if isinstance(f, dict)
                 and f.get("name") and f.get("test_command")][:self.max_features]
        if not feats:
            raise ValueError("the plan has no features with a test_command")
        return {"summary": str(plan.get("summary", "")), "stack": str(plan.get("stack", "")),
                "run_command": str(plan.get("run_command", "")),
                "test_all": str(plan.get("test_all", "")),
                "features": [{"name": str(f["name"])[:80],
                              "description": str(f.get("description", ""))[:800],
                              "files": [str(x) for x in f.get("files", [])][:12],
                              "test_command": str(f["test_command"])[:200],
                              "acceptance": str(f.get("acceptance", ""))[:400]} for f in feats]}

    def _plan(self, brief: str) -> None:
        self._phase("plan", "Analysing the research and planning features")
        memory = render(self.ctx.select(self.goal, budget_tokens=1500).units)
        user = (f"## Goal\n{self.goal}\n\n## Research brief\n{brief or '(skipped)'}\n\n"
                f"## Saved findings\n{memory}\n\n"
                + self.PLAN_PROMPT.replace("{max}", str(self.max_features)))
        last_err = ""
        for attempt in range(3):
            text, provider = self.pool.ask("smart", "You are a senior software architect. "
                                           "Reply with JSON only.",
                                           user + (f"\n\nYour last reply was unusable: {last_err}. "
                                                   "Reply with the JSON object only." if last_err else ""),
                                           emit=self.emit)
            try:
                self.plan = self._clean_plan(_json_obj(text))
                break
            except ValueError as e:
                last_err = str(e)
        else:
            raise ValueError(f"no usable plan after 3 tries: {last_err}")
        for i, f in enumerate(self.plan["features"], 1):
            self.ctx.put(f"/project/plan/feature-{i}", f"{f['name']}: {f['description']}",
                         kind="decision", importance=0.8)
        self.emit({"type": "plan", "plan": self.plan, "provider": provider})

    def _build(self) -> None:
        feats = self.plan["features"]
        for i, f in enumerate(feats, 1):
            if self.stop.is_set():
                return
            self._phase("build", f"Feature {i}/{len(feats)}: {f['name']}")
            done = [x["name"] for x in self.results]
            task = (f"## Project goal\n{self.goal}\n\n## Plan\n{self.plan['summary']}\n"
                    f"Stack: {self.plan['stack']}\n\n## Your feature ({i} of {len(feats)}): "
                    f"{f['name']}\n{f['description']}\nFiles: {', '.join(f['files'])}\n"
                    f"Acceptance: {f['acceptance']}\n\nAlready built: "
                    f"{', '.join(done) or 'nothing yet'} (don't break them).\n\n"
                    f"Write the code AND its tests, then run: {f['test_command']}\n"
                    "Finish only when that command passes. Load a matching skill first "
                    "(for Python: python-project).")
            passed, output, rounds = False, "", 0
            for rounds in range(1, 4):
                r = self._agent(BUILD_TOOLS).run(task, "You are a careful software engineer "
                                                 "who builds one feature at a time and "
                                                 "proves it with tests.", lane="smart",
                                                 max_steps=22)
                if self.stop.is_set():
                    return
                ok, output = self._run_cmd(f["test_command"])
                self.emit({"type": "verify", "feature": f["name"], "command": f["test_command"],
                           "passed": ok, "output": clip(output, 2500), "round": rounds})
                if ok:
                    passed = True
                    break
                task = (f"The feature '{f['name']}' is not done: `{f['test_command']}` "
                        f"fails when the engine runs it:\n\n{clip(output, 3000)}\n"
                        f"{fix_hint(output)}\n"
                        f"Fix the cause (not the test, unless the test is wrong). Original "
                        f"brief:\n{f['description']}\nAcceptance: {f['acceptance']}")
            regress = ""
            if passed and self.plan.get("test_all") and i > 1:
                ok_all, regress = self._run_cmd(self.plan["test_all"])
                self.emit({"type": "verify", "feature": "all tests so far",
                           "command": self.plan["test_all"], "passed": ok_all,
                           "output": clip(regress, 2000), "round": 1})
                passed = passed and ok_all
            self.results.append({"name": f["name"], "passed": passed, "rounds": rounds,
                                 "output": clip(output, 1500)})
            self.ctx.put(f"/project/progress/feature-{i}",
                         f"{f['name']}: {'passed' if passed else 'FAILED'} its tests",
                         kind="fact" if passed else "blocker", importance=0.8)

    def _security(self) -> str:
        self._phase("security", "Scanning for vulnerabilities")
        before = security_scan(self.ws)
        self.emit({"type": "security", "stage": "scan", "report": before})
        if re.search(r"\[(high|medium)\]", before):
            r = self._agent(BUILD_TOOLS).run(
                f"Security review of this project. Load the security-review skill. The "
                f"automated scan reported:\n\n{before}\n\nConfirm each finding, fix real ones "
                f"with the smallest change, and keep every test passing "
                f"({self.plan.get('test_all') or 'run the tests'}). Finish with a list of what "
                "you fixed and anything you judged a false positive.",
                "You are an application security engineer.", lane="smart", max_steps=18)
            after = security_scan(self.ws)
            ok, out = self._run_cmd(self.plan["test_all"]) if self.plan.get("test_all") else (True, "")
            self.emit({"type": "security", "stage": "fixed", "summary": r.summary,
                       "report": after, "tests_pass": ok})
            return f"Before fixes:\n{before}\n\nReviewer:\n{r.summary}\n\nAfter fixes:\n{after}"
        return before

    def _report(self, brief: str, sec: str) -> None:
        self._phase("report", "Writing BUILD_REPORT.md")
        lines = [f"# Build report", "", f"**Goal:** {self.goal}", "",
                 f"**Result:** {sum(r['passed'] for r in self.results)} of "
                 f"{len(self.results)} features pass their tests.", "",
                 "## How to run", "", f"```\n{self.plan.get('run_command', '')}\n```", "",
                 "## How to test", "", f"```\n{self.plan.get('test_all', '')}\n```", "",
                 "## Plan", "", self.plan.get("summary", ""), "",
                 f"Stack: {self.plan.get('stack', '')}", "", "## Features", ""]
        for f, r in zip(self.plan["features"], self.results):
            lines.append(f"- {'PASS' if r['passed'] else 'FAIL'} **{f['name']}**: "
                         f"{f['description']} (`{f['test_command']}`, {r['rounds']} round(s))")
        lines += ["", "## Research", "", brief or "(skipped)", "",
                  "## Security", "", "```", sec.strip()[:6000], "```", "",
                  "_Written by ContextOS from the engine's own test runs._"]
        (self.ws.root / "BUILD_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
