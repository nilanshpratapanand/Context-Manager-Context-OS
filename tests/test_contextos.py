"""Test suite. Run: python -m pytest tests/ -q   (or python tests/test_contextos.py)"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextos import ContextOS  # noqa: E402
from contextos.bench import default_tasks, run as bench_run  # noqa: E402
from contextos.handoff import POLICY, classify, should_migrate  # noqa: E402
from contextos.units import AddressError, count_tokens, normalize_address  # noqa: E402


class _Raises:
    def __init__(self, exc):
        self.exc = exc

    def __enter__(self):
        return self

    def __exit__(self, t, v, tb):
        if t is None:
            raise AssertionError(f"expected {self.exc.__name__}")
        return issubclass(t, self.exc)


def raises(exc):
    return _Raises(exc)


def new_ctx():
    return ContextOS()


# ------------------------------------------------------------------ addresses
def test_address_normalisation():
    for raw, want in [
        ("/project/architecture/db", "/project/architecture/db"),
        ("project/architecture/db", "/project/architecture/db"),
        ("/Project/Architecture/DB/", "/project/architecture/db"),
    ]:
        assert normalize_address(raw) == want, raw


def test_bad_addresses_rejected():
    for bad in ["", "/", "/nonsense/x", "/project/" + "a/" * 20,
                "/project/has space", "/project/!!"]:
        with raises(AddressError):
            normalize_address(bad)


# ---------------------------------------------------------------------- store
def test_put_get_roundtrip():
    ctx = new_ctx()
    u = ctx.put("/project/decisions/db", "PostgreSQL 16", kind="decision", importance=0.9)
    got = ctx.get("/project/decisions/db")
    assert got is not None and got.value == "PostgreSQL 16"
    assert got.version == 1 and got.tokens == u.tokens and got.live


def test_identical_write_is_noop():
    ctx = new_ctx()
    ctx.put("/project/decisions/db", "PostgreSQL 16")
    ctx.put("/project/decisions/db", "PostgreSQL 16")
    assert ctx.get("/project/decisions/db").version == 1


def test_versioning_and_history():
    ctx = new_ctx()
    ctx.put("/project/decisions/db", "MySQL", source="a")
    ctx.put("/project/decisions/db", "PostgreSQL 16", source="a", supersede=True)
    u = ctx.get("/project/decisions/db")
    assert u.version == 2 and u.value == "PostgreSQL 16"
    hist = ctx.history("/project/decisions/db")
    assert len(hist) == 1 and hist[0]["value"] == "MySQL"


def test_conflict_detected_across_sources():
    ctx = new_ctx()
    ctx.put("/project/decisions/db", "PostgreSQL", source="architect")
    ctx.put("/project/decisions/db", "MongoDB", source="rogue")
    cs = ctx.conflicts()
    assert len(cs) == 1
    assert cs[0]["old_source"] == "architect" and cs[0]["new_source"] == "rogue"


def test_supersede_flag_suppresses_conflict():
    ctx = new_ctx()
    ctx.put("/project/decisions/db", "PostgreSQL", source="architect")
    ctx.put("/project/decisions/db", "MongoDB", source="rogue", supersede=True)
    assert ctx.conflicts() == []


def test_superseded_units_are_not_live():
    ctx = new_ctx()
    ctx.put("/task/blockers/redis", "no ACL", kind="blocker")
    assert ctx.supersede("/task/blockers/redis") is True
    assert ctx.get("/task/blockers/redis") is None
    assert ctx.get("/task/blockers/redis", include_superseded=True) is not None


def test_expire_respects_lifetime_and_pins():
    ctx = new_ctx()
    ctx.put("/user/preferences/x", "keep me", lifetime="permanent")
    ctx.put("/tool/search/a", "chatter", kind="tool_result", lifetime="ephemeral")
    ctx.put("/task/progress/x", "pinned", lifetime="ephemeral", pinned=True)
    assert ctx.expire() == 1
    assert ctx.get("/tool/search/a") is None
    assert ctx.get("/user/preferences/x") is not None
    assert ctx.get("/task/progress/x") is not None


def test_artifact_records_path_and_hash():
    ctx = new_ctx()
    u = ctx.put_artifact("/artifact/auth/handler", "src/auth.py", "def f(): pass")
    assert u.meta["path"] == "src/auth.py" and len(u.meta["sha256"]) == 64
    assert "src/auth.py" in u.render()


def test_persists_to_disk():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.db")
        c1 = ContextOS(path)
        c1.put("/task/goal", "ship it", kind="goal")
        c1.close()
        c2 = ContextOS(path)
        assert c2.get("/task/goal").value == "ship it"
        c2.close()


# ------------------------------------------------------------------ retrieval
def test_exact_address_beats_search():
    ctx = new_ctx()
    ctx.put("/project/architecture/database", "PostgreSQL 16")
    hits = ctx.search("/project/architecture/database")
    assert len(hits) == 1 and hits[0].why == "exact"


def test_hybrid_search_finds_the_right_unit():
    ctx = new_ctx()
    ctx.put("/project/architecture/database", "PostgreSQL 16 for JSONB", kind="decision")
    ctx.put("/project/architecture/queue", "RabbitMQ for job fan-out", kind="decision")
    for i in range(30):
        ctx.put(f"/tool/search/n{i}", "unrelated chatter about sockets and retries",
                kind="tool_result", importance=0.1)
    assert ctx.search("what database did we decide on")[0].unit.address == \
        "/project/architecture/database"


def test_search_scopes_by_prefix():
    ctx = new_ctx()
    ctx.put("/project/notes/db", "postgres notes")
    ctx.put("/user/preferences/db", "prefers postgres")
    hits = ctx.search("db postgres", prefix="/user")
    assert all(h.unit.address.startswith("/user") for h in hits)


# ------------------------------------------------------------------- budgets
def test_selection_respects_budget():
    ctx = new_ctx()
    for i in range(60):
        ctx.put(f"/tool/search/n{i}", "x " * 200, kind="tool_result", importance=0.2)
    sel = ctx.select("anything", budget_tokens=500)
    assert sel.tokens_selected <= 500 and sel.omitted


def test_structural_units_survive_a_tiny_budget():
    ctx = new_ctx()
    ctx.put("/task/goal", "the goal", kind="goal", importance=1.0)
    ctx.put("/project/constraints/c", "a hard constraint", kind="constraint")
    for i in range(40):
        ctx.put(f"/tool/search/n{i}", "chatter " * 60, kind="tool_result", importance=0.1)
    p = ctx.handoff(direction="escalate", budget_tokens=300)
    kinds = {u.kind for u in p.units}
    assert "goal" in kinds and "constraint" in kinds


# -------------------------------------------------------------------- handoff
def _seeded(c):
    c.put("/task/goal", "Add OAuth2 to billing", kind="goal", importance=1.0, pinned=True)
    c.put("/project/constraints/no-deps", "no new deps", kind="constraint", importance=0.95)
    c.put("/project/decisions/lib", "use authlib", kind="decision", importance=0.9)
    c.put("/task/blockers/redis", "redis has no ACL", kind="blocker", importance=0.8)
    c.put_artifact("/artifact/auth/token", "src/auth/token.py", "async def token(): ...")
    for i in range(50):
        c.put(f"/tool/search/n{i}", "chatter " * 40, kind="tool_result", importance=0.1)
    return c


def test_escalate_drops_tool_chatter_downshift_keeps_it():
    ctx = new_ctx()
    _seeded(ctx)
    esc = ctx.handoff(direction="escalate", budget_tokens=4000)
    down = ctx.handoff(direction="downshift", budget_tokens=4000)
    assert not any(u.kind == "tool_result" for u in esc.units)
    assert any(u.kind == "tool_result" for u in down.units)
    assert esc.tokens_selected < down.tokens_selected


def test_packet_never_exceeds_its_rendered_budget():
    ctx = new_ctx()
    _seeded(ctx)
    for direction in POLICY:
        for budget in (400, 900, 2000):
            p = ctx.handoff(direction=direction, budget_tokens=budget)
            scale = POLICY[direction]["budget_scale"]
            assert p.tokens_selected <= max(int(budget * scale), 1) or \
                any("Structural context" in n for n in p.notes)


def test_omitted_manifest_accounts_for_everything():
    ctx = new_ctx()
    _seeded(ctx)
    p = ctx.handoff(direction="escalate", budget_tokens=800)
    live = {u.address for u in ctx.list("", live_only=True)}
    covered = {u.address for u in p.units} | {o["address"] for o in p.omitted}
    assert live == covered, "every live unit must be either sent or declared missing"


def test_packet_carries_goal_constraints_and_blockers():
    ctx = new_ctx()
    _seeded(ctx)
    text = ctx.handoff(direction="escalate", budget_tokens=2000).render()
    assert "Add OAuth2 to billing" in text
    assert "no new deps" in text and "redis has no ACL" in text


def test_conflict_warning_reaches_the_packet():
    ctx = new_ctx()
    _seeded(ctx)
    ctx.put("/project/decisions/lib", "use oauthlib", source="other", importance=0.9)
    assert any("conflict" in n.lower()
               for n in ctx.handoff(direction="escalate", budget_tokens=2000).notes)


def test_direction_classification():
    assert classify(0.3, 0.9) == "escalate"
    assert classify(0.9, 0.3) == "downshift"
    assert classify(0.5, 0.55) == "lateral"


def test_easy_task_escalation_is_refused():
    assert should_migrate("escalate", 0.9)[0] is True
    assert should_migrate("escalate", 0.1)[0] is False
    assert should_migrate("downshift", 0.1)[0] is True


def test_reduction_is_measured_against_full_replay():
    ctx = new_ctx()
    _seeded(ctx)
    p = ctx.handoff(direction="escalate", budget_tokens=800)
    assert p.tokens_stored > p.tokens_selected and 0 < p.reduction < 1


# ------------------------------------------------------------------ benchmark
def test_benchmark_ground_truth_is_satisfiable():
    """Every required address must actually be written by an earlier step,
    otherwise the benchmark would be measuring an impossible target."""
    for task in default_tasks():
        written: set[str] = {"/task/goal"}
        for step in task.steps:
            for req in step.requires:
                assert req in written, f"{task.name}: {req} required before it is written"
            for w in step.writes:
                written.add(w["address"])


def test_contextos_beats_baselines_on_sufficiency():
    res = bench_run(direction="lateral", budgets=(800, 1500))["summary"]
    co, best_base = res["contextos"], max(
        (res[k] for k in ("full_replay", "recency", "summary")),
        key=lambda a: a["recall_strict"])
    assert co["recall_strict"] >= best_base["recall_strict"]
    assert co["mean_tokens"] < best_base["mean_tokens"]
    assert co["over_budget_rate"] == 0.0


def test_tokens_counted_consistently():
    assert count_tokens("") == 0
    assert count_tokens("hello world") > 0


def _run() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  ok   {name}")
        except Exception as exc:
            failed.append((name, exc))
            print(f"  FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    if failed:
        import traceback
        for name, exc in failed:
            print(f"\n--- {name} ---")
            traceback.print_exception(type(exc), exc, exc.__traceback__)
    return 1 if failed else 0




# ----------------------------------------------------------------- live harness
from contextos import live as _live  # noqa: E402


def test_load_env_handles_crlf_and_bom():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, ".env")
        with open(p, "wb") as fh:
            fh.write(b"\xef\xbb\xbf# comment\r\nGROQ_API_KEY=abc123\r\n"
                     b"EMPTY=\r\nQUOTED=\"xy z\"\r\n")
        env = _live.load_env(p)
    assert env["GROQ_API_KEY"] == "abc123"      # no trailing \r
    assert env["QUOTED"] == "xy z"
    assert "EMPTY" not in env                    # blank values are not keys


def test_load_env_missing_file_is_empty():
    assert _live.load_env("/nonexistent/.env") == {}


def test_extract_final_takes_the_last_one():
    assert _live.extract_final("FINAL: 1\nmore\nFINAL: 2754.00") == "2754.00"
    assert _live.extract_final("no answer here") is None


def test_task_checker_accepts_formatting_variation():
    task = _live.default_live_tasks()[0]
    assert task.check("FINAL: 2754.00")[0] is True
    assert task.check("FINAL: 2,754.00")[0] is True
    assert task.check("FINAL: 3240.00")[0] is False


def test_live_task_answers_are_not_the_naive_answers():
    for t in _live.default_live_tasks():
        assert t.answer != t.naive_answer
        assert t.constraints, f"{t.name} has no constraint to lose"


def test_complete_without_a_key_raises_cleanly():
    try:
        _live.complete(_live.PROVIDERS["groq"], "s", "u", {})
    except _live.ProviderError as exc:
        assert "GROQ_API_KEY" in str(exc)
    else:
        raise AssertionError("expected ProviderError")


def test_every_interface_produces_a_transfer():
    task = _live.default_live_tasks()[0]
    transcript, ctx = _live.phase_a_transcript(
        task, _live.PROVIDERS["groq"], {}, live=False)
    try:
        for name, fn in _live.INTERFACES.items():
            out = fn(task, transcript, ctx, 1200)
            assert isinstance(out, str) and out.strip(), name
    finally:
        ctx.close()


def test_dry_run_contextos_matches_raw_for_fewer_tokens():
    res = _live.run("groq", "gemini", live=False, env={})
    by = {}
    for r in res:
        by.setdefault(r.interface, []).append(r)
    raw, co = by["raw"], by["contextos"]
    assert all(r.ok for r in co), "contextos must carry the constraints"
    assert sum(r.ok for r in co) >= sum(r.ok for r in raw)
    assert (sum(r.transfer_tokens for r in co) / len(co)) < \
           (sum(r.transfer_tokens for r in raw) / len(raw))


def test_dry_run_traj_drop_loses_non_file_state():
    """The thesis, as an assertion: traj-drop is the paper's best escalation
    interface, and it loses constraints because constraints are not files."""
    res = [r for r in _live.run("groq", "gemini", live=False, env={})
           if r.interface == "traj_drop"]
    assert not any(r.ok for r in res)
    assert all(r.naive for r in res)

def test_strip_reasoning_removes_think_blocks():
    s = _live.strip_reasoning
    assert s("<think>chain of thought</think>The answer is 42.") == "The answer is 42."
    assert s("<THINK>x</THINK>ok") == "ok"
    assert s("<thinking>a</thinking>b") == "b"
    assert s("<think>a</think>mid<think>b</think>end") == "midend"
    assert s("plain text") == "plain text"


def test_strip_reasoning_handles_unclosed_tag():
    """A truncated reply leaves an open <think>; everything after it is thought,
    not answer, and must not reach the user or the store."""
    assert _live.strip_reasoning("before <think>cut off mid thou") == "before"


def test_check_reports_every_provider():
    rows = _live.check({})           # no keys: must not make any network call
    assert {r["provider"] for r in rows} == set(_live.PROVIDERS)
    assert all(r["status"] == "no key" for r in rows)


def test_read_timeout_becomes_provider_error():
    """A slow provider must trigger fallback, not crash the turn."""
    import urllib.request
    real = urllib.request.urlopen

    def slow(*a, **k):
        raise TimeoutError("The read operation timed out")
    urllib.request.urlopen = slow
    try:
        with raises(_live.ProviderError):
            _live.complete(_live.PROVIDERS["groq"], "s", "u", {"GROQ_API_KEY": "x"})
    finally:
        urllib.request.urlopen = real


from contextos import router as _router  # noqa: E402


def test_router_sends_easy_prompts_to_fast_lane():
    for p in ("hi", "thanks!", "What is the capital of France?",
              "rephrase: we shipped it", "translate hello to hindi",
              "what does API stand for"):
        assert _router.decide(p).lane == "fast", p


def test_router_sends_hard_prompts_to_smart_lane():
    for p in ("Design a database schema for a hostel booking app",
              "Debug this:\n```python\ndef f(x): return x/0\n```",
              "A shirt costs 800, gets 10% off, then 18% tax. Final price?",
              "write a python function to reverse a linked list",
              "Compare Raft and Paxos and explain why one is easier to implement"):
        assert _router.decide(p).lane == "smart", p


def test_router_override_prefix_and_mode():
    mode, rest = _router.parse_override("/fast Design a compiler")
    assert (mode, rest) == ("fast", "Design a compiler")
    assert _router.parse_override("no prefix") == (None, "no prefix")
    d = _router.decide("hi", mode="smart")
    assert d.lane == "smart" and d.forced


def test_router_chain_spills_to_other_lane_and_benches_cooling():
    chain = _router.build_chain("fast", ["a", "b"], ["c", "a"],
                                cooling=lambda n: n == "c")
    assert chain == ["a", "b", "c"]          # fast first, dedup, cooling last
    assert _router.build_chain("smart", ["a"], ["b"]) == ["a", "b"]


def test_cooldown_lengths_match_error_kind():
    t = [100.0]
    cd = _router.Cooldown(clock=lambda: t[0])
    assert cd.hit("x", "HTTP 404: model_not_found") == 3600
    assert cd.hit("y", "HTTP 429: Rate limit exceeded") == 60
    assert cd.hit("z", "HTTP 503: high demand") == 30
    assert cd.active("y")
    t[0] += 61
    assert not cd.active("y") and cd.active("x")
    cd.clear(["x"])
    assert not cd.active("x")


def _offline_engine():
    from contextos.server import Engine
    return Engine(tempfile.mkdtemp(), {}, offline=True)


def test_engine_routes_by_difficulty():
    e = _offline_engine()
    easy = e.chat("hi there")
    assert easy["route"]["lane"] == "fast" and easy["provider"] == "offline-b"
    hard = e.chat("Design a schema and explain why it avoids the N+1 bug")
    assert hard["route"]["lane"] == "smart" and hard["provider"] == "offline-a"
    forced = e.chat("/smart hi")
    assert forced["route"]["forced"] and forced["provider"] == "offline-a"


def test_engine_falls_back_across_lanes_and_benches_failed_route():
    e = _offline_engine()
    e.toggle_failure("offline-a")
    r = e.chat("Design a schema and explain why it avoids the N+1 bug")
    assert r["provider"] == "offline-b"
    assert r["switched"][0]["direction"] == "downshift"
    assert e.cooldown.active("offline-a")
    e.toggle_failure("offline-a")                 # restoring it lifts the bench
    assert not e.cooldown.active("offline-a")
    assert e.chat("Design a new schema for the booking table")["provider"] == "offline-a"


def test_commit_ignores_think_aloud_and_reads_every_block():
    e = _offline_engine()
    text = ("We need to output a <context> block with entries: fact | /x | y.</context>\n"
            "<context>\nfact | /task/inputs/price | 800\n</context>\nAnswer here.")
    written = e._commit(text, "m")
    assert [w["address"] for w in written] == ["/task/inputs/price"]


def test_engine_saves_user_input_when_model_saves_nothing():
    e = _offline_engine()
    e._offline_reply = lambda name, user: "Here is an answer with no context block."
    r = e.chat("Beds cost 450 rupees and checkout is at 10am")
    assert r["written"][0]["address"] == "/task/inputs/turn-1"
    assert "450" in e.ctx.get("/task/inputs/turn-1").value
    assert e.chat("thanks")["written"] == []      # small talk is not stored


def test_degenerate_reply_detection():
    from contextos.server import is_degenerate
    assert is_degenerate("!" * 200)
    assert is_degenerate("ok " + "!" * 120)
    assert not is_degenerate("The final price is **849.6**.")
    assert not is_degenerate("```\n" + "-" * 40 + "\n| a | b |\n```\nTable above shows the "
                             "columns, keys, and constraints for the bookings table.")


def test_chat_store_crud_search_and_truncate():
    from contextos.chats import ChatStore, title_from
    cs = ChatStore(os.path.join(tempfile.mkdtemp(), "c.db"))
    c = cs.create()
    a = cs.add(c["id"], "user", "hello there")
    cs.add(c["id"], "assistant", "hi", {"provider": "groq"})
    b = cs.add(c["id"], "user", "tell me about PostgreSQL")
    cs.add(c["id"], "assistant", "it is a database")
    assert [m["seq"] for m in cs.messages(c["id"])] == [1, 2, 3, 4]
    assert cs.messages(c["id"])[1]["meta"] == {"provider": "groq"}
    assert cs.list("postgres")[0]["id"] == c["id"] and cs.list("nope") == []
    removed = cs.truncate_from(c["id"], b["id"])
    assert len(removed) == 2 and len(cs.messages(c["id"])) == 2
    assert cs.rename(c["id"], "  My chat ") and cs.get(c["id"])["title"] == "My chat"
    assert cs.delete(c["id"]) and cs.get(c["id"]) is None
    assert cs.messages(c["id"]) == []                  # cascade removed messages
    assert title_from("one two three four five six seven eight") == \
        "one two three four five six seven…"
    assert a["id"]


def test_visible_text_hides_blocks_and_partial_tags():
    from contextos.server import visible_text as v
    assert v("<context>\nfact | /task/x | 1\n</context>\nHello") == "Hello"
    assert v("<context>\nfact | /task/x | 1") == ""          # block still arriving
    assert v("Hello <con") == "Hello"                          # tag may be starting
    assert v("<think>plan</think>Answer") == "Answer"
    assert v("a < b and c") == "a < b and c"                  # ordinary text survives


def test_stream_events_and_persisted_transcript():
    e = _offline_engine()
    evs = list(e.chat_stream(None, "Design a schema for bookings"))
    kinds = [x["type"] for x in evs]
    assert kinds[:3] == ["start", "route", "model"] and kinds[-1] == "done"
    text = "".join(x["text"] for x in evs if x["type"] == "delta")
    done = evs[-1]["message"]
    assert text == done["content"] and "<context>" not in text
    cid = evs[0]["conversation"]["id"]
    conv = e.chats.get(cid)
    assert conv["title"] == "Design a schema for bookings"
    assert [m["role"] for m in conv["messages"]] == ["user", "assistant"]
    assert done["meta"]["written"] and e.memory(cid)["units"]


def test_midstream_failure_resets_and_hands_off():
    e = _offline_engine()
    real = e._stream

    def flaky(name, system, user):
        if name == "offline-a":
            yield "Partial answer that"
            raise _live.ProviderError("HTTP 503: high demand")
        yield from real(name, system, user)
    e._stream = flaky
    evs = list(e.chat_stream(None, "Design a schema and explain why"))
    kinds = [x["type"] for x in evs]
    assert "reset" in kinds and "switch" in kinds
    assert kinds.index("reset") < kinds.index("switch")
    assert evs[-1]["message"]["meta"]["provider"] == "offline-b"


def test_garbled_stream_falls_back():
    e = _offline_engine()
    real = e._stream

    def junk(name, system, user):
        if name == "offline-a":
            yield "!" * 80
            return
        yield from real(name, system, user)
    e._stream = junk
    evs = list(e.chat_stream(None, "Design a schema and explain why"))
    assert evs[-1]["message"]["meta"]["provider"] == "offline-b"
    assert "garbled" in evs[-1]["message"]["meta"]["attempts"][0]["error"]


def test_regenerate_and_edit_undo_store_writes():
    e = _offline_engine()
    first = list(e.chat_stream(None, "Design the booking schema"))
    cid = first[0]["conversation"]["id"]
    reply = first[-1]["message"]
    added = [w["address"] for w in reply["meta"]["written"] if w["created"]]
    assert added and all(e.ctx_for(cid).get(a) for a in added)
    regen = list(e.chat_stream(cid, regenerate=reply["id"]))
    assert regen[-1]["type"] == "done"
    assert [m["role"] for m in e.chats.messages(cid)] == ["user", "assistant"]
    user_id = e.chats.messages(cid)[0]["id"]
    list(e.chat_stream(cid, "Design the invoices schema instead", edit=user_id))
    msgs = e.chats.messages(cid)
    assert [m["content"] for m in msgs if m["role"] == "user"] == \
        ["Design the invoices schema instead"]
    addrs = {u["address"] for u in e.memory(cid)["units"]}
    assert not any("booking" in a for a in addrs if a.startswith("/project/notes"))
    assert e.ctx_for(cid).get("/task/goal").value == "Design the invoices schema instead"
    assert e.chats.get(cid)["title"] == "Design the invoices schema instead"
    e.chats.rename(cid, "My own title")                  # a title the user chose stays
    list(e.chat_stream(cid, "Design the payments schema",
                       edit=e.chats.messages(cid)[0]["id"]))
    assert e.chats.get(cid)["title"] == "My own title"


def test_stop_keeps_partial_reply_without_committing():
    e = _offline_engine()
    gen = e.chat_stream(None, "Design a schema for bookings")
    cid = None
    for ev in gen:
        cid = cid or ev.get("conversation", {}).get("id")
        if ev["type"] == "delta":
            break
    gen.close()                                     # what a browser Stop does
    last = e.chats.messages(cid)[-1]
    assert last["role"] == "assistant" and last["meta"]["stopped"]
    assert not any(u["address"].startswith("/project/notes")
                   for u in e.memory(cid)["units"])


def test_conversations_have_separate_memory_and_delete_cleanly():
    e = _offline_engine()
    a = list(e.chat_stream(None, "Design the booking schema"))[0]["conversation"]["id"]
    b = list(e.chat_stream(None, "Plan the invoice module"))[0]["conversation"]["id"]
    goals = {e.ctx_for(c).get("/task/goal").value for c in (a, b)}
    assert goals == {"Design the booking schema", "Plan the invoice module"}
    assert e.delete_conversation(a)
    assert not (e.data / "ctx" / f"{a}.db").exists()
    assert [c["id"] for c in e.chats.list()] == [b]


def test_reasoning_streams_as_thinking_events():
    import time as _t
    e = _offline_engine()

    def thinker(name, system, user):
        yield ("think", "Let me work this out. ")
        _t.sleep(0.02)
        yield ("think", "Discount first, then tax.")
        yield "The answer is **849.6**."
    e._stream = thinker
    evs = list(e.chat_stream(None, "hi"))
    thinking = "".join(x["text"] for x in evs if x["type"] == "thinking")
    assert thinking == "Let me work this out. Discount first, then tax."
    done = evs[-1]["message"]
    assert done["content"] == "The answer is **849.6**." and done["meta"]["thought_ms"] > 0


def test_route_pin_goes_first():
    e = _offline_engine()
    r = e.chat("hi", route="offline-c")
    assert r["provider"] == "offline-c"


def _http_server():
    import threading
    from http.server import ThreadingHTTPServer
    from contextos.server import Engine, Handler
    Handler.engine = Engine(tempfile.mkdtemp(), {}, offline=True)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    httpd.daemon_threads = True
    Handler.port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{Handler.port}"


def _req(url, body=None, headers=None):
    import json as _j
    import urllib.request
    h = {"Content-Type": "application/json", **(headers or {})}
    data = None if body is None else _j.dumps(body).encode()
    return urllib.request.urlopen(urllib.request.Request(url, data=data, headers=h),
                                  timeout=10)


def test_http_chat_flow_end_to_end():
    import json as _j
    httpd, base = _http_server()
    try:
        assert b"<html" in _req(base + "/").read().lower()
        cid = _j.load(_req(base + "/api/conversations", {}))["id"]
        lines = _req(base + "/api/chat", {"conversation_id": cid,
                                          "prompt": "Design a bookings schema"}).read()
        evs = [_j.loads(x) for x in lines.decode().splitlines()]
        assert evs[0]["type"] == "start" and evs[-1]["type"] == "done"
        conv = _j.load(_req(base + f"/api/conversations/{cid}"))
        assert len(conv["messages"]) == 2 and conv["title"] == "Design a bookings schema"
        assert _j.load(_req(base + f"/api/conversations/{cid}/memory"))["units"]
        _req(base + f"/api/conversations/{cid}/rename", {"title": "Bookings"})
        listed = _j.load(_req(base + "/api/conversations?q=Book"))["conversations"]
        assert listed[0]["title"] == "Bookings"
        assert b"**You:**" in _req(base + f"/api/conversations/{cid}/export").read()
        assert _j.load(_req(base + f"/api/conversations/{cid}/delete", {}))["deleted"]
    finally:
        httpd.shutdown()


def test_http_rejects_other_origins_and_non_json():
    import urllib.error
    httpd, base = _http_server()
    try:
        for hdrs in ({"Origin": "https://evil.example"},
                     {"Content-Type": "text/plain"}):
            try:
                _req(base + "/api/conversations", {}, hdrs)
                raise AssertionError(f"accepted {hdrs}")
            except urllib.error.HTTPError as e:
                assert e.code == 403
    finally:
        httpd.shutdown()


def test_http_stop_mid_stream_saves_partial():
    import json as _j
    import socket
    import time as _t
    from contextos.server import Handler
    httpd, base = _http_server()
    try:
        cid = _j.load(_req(base + "/api/conversations", {}))["id"]
        body = _j.dumps({"conversation_id": cid,
                         "prompt": "Design a bookings schema"}).encode()
        s = socket.create_connection(("127.0.0.1", Handler.port))
        s.sendall(b"POST /api/chat HTTP/1.0\r\nHost: 127.0.0.1\r\n"
                  b"Content-Type: application/json\r\nContent-Length: "
                  + str(len(body)).encode() + b"\r\n\r\n" + body)
        buf = b""
        while b'"delta"' not in buf:
            buf += s.recv(4096)
        s.close()                                   # the browser's Stop button
        for _ in range(100):
            msgs = Handler.engine.chats.messages(cid)
            if len(msgs) == 2:
                break
            _t.sleep(0.05)
        assert msgs[-1]["meta"].get("stopped")
    finally:
        httpd.shutdown()


def test_engine_reads_lane_order_from_env():
    from contextos.server import Engine
    d = tempfile.mkdtemp()
    env = {"GROQ_API_KEY": "k", "MISTRAL_API_KEY": "k",
           "LLM_SMART_ORDER": "mistral,groq,nvidia", "LLM_FAST_ORDER": "groq-fast"}
    e = Engine(d, env, offline=False)
    assert e.smart == ["mistral", "groq"]        # nvidia has no key: skipped
    assert e.fast == ["groq-fast"]


if __name__ == "__main__":
    raise SystemExit(_run())
