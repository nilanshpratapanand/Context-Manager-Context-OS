# ContextOS

**The state layer that makes trajectory-drop handoffs work.**

When a task moves from one model to another, what should travel — and what should not?
ContextOS is an addressable, bi-temporal context store that answers that question with a
direction-aware handoff packet, and tells the receiving model what it deliberately left out.

## Dashboard

```bash
python -m contextos.server --offline     # no API keys needed
python -m contextos.server               # uses the keys in .env
```

Opens at `http://127.0.0.1:8000`. You type a prompt; the server does the rest:

1. selects **only the relevant context** from the store — never the whole history
2. calls the current provider with it
3. if that provider is rate-limited or fails, classifies the direction, builds a
   handoff packet, switches provider, and retries — the task carries on
4. extracts durable state from the reply (`decision`, `constraint`, `blocker`, …) and
   commits it back to an address

Conversation history is capped at a few turns on purpose. **History is not the memory —
the store is.** The sidebar shows the store filling up live, the token budget actually
being spent, and every model switch as it happens.

Click any provider in the sidebar to force it to fail, then send another message. That's
the demo: the model changes, the task doesn't.

---

**On Windows, double-click `RUN.bat`** — a menu for everything, dashboard first, and it
tells you what to do if Python is missing.

```bash
python -m contextos.demo      # end-to-end story
python -m contextos.bench     # the benchmark
python tests/test_contextos.py
```

No dependencies. Python 3.9+. SQLite and FTS5 only. `tiktoken` is used for exact token
counts if it happens to be installed, and a `chars/4` estimate otherwise.

---

## Why this is not another memory system

The original idea was *"never lose context — carry everything to the next model."*
The research says that is the worst available option.

**arXiv 2608.24358, *The Handoff Tax*** — 58,000 agent runs, 2M API calls, 36B tokens,
on SWE-bench Verified with Claude Haiku 4.5 / Opus 4.7 and GPT-5.6 Luna / Sol:

| Interface | Escalation (weak → strong) | Downshift (strong → weak) |
|---|---|---|
| Raw full trajectory | **47%** / 36% quality-gap recovery, at **4.0×** / 6.1× cost | 50–79% |
| `traj-drop` — no trajectory, working-tree edits kept | **64%** / **84%** | collapses to 28% / 53% |
| `compact_pre` — departing model summarises first | cost $1.61 → $0.75, quality 47% → 60% | — |

Two things follow, and they are the whole design:

1. **Dropping the trajectory beats carrying it — when escalating.** It works because the
   *working tree* survives. Nobody had built the state layer that lets non-file state —
   decisions, constraints, blockers — survive the same way. That is this project.
2. **The right packet depends on direction.** A system that ships one format is provably
   wrong in one of the two directions.

**arXiv 2608.21208, *Specification Portability*** — 1,802 Oracle→PostgreSQL scripts across
Kiro, Gemini, Copilot: worst cross-agent transfer scored Token F1 **0.035** and 2.33% SQL
validity. Agent-authored artifacts are *not* agent-neutral. Retrieval-augmented ingestion
sat on the Pareto frontier; **compression gave no universal benefit**. So ContextOS
*selects* verbatim units rather than summarising them, and renders packets into a neutral
schema rather than one agent's idiom.

### Against the prior art

| System | What it does | Benchmarked on | Cross-model handoff |
|---|---|---|---|
| MemGPT / Letta | OS-style paging between main and external context | DMR, doc QA | no |
| Mem0 | LLM extraction + ADD/UPDATE/DELETE/NOOP consolidation | LOCOMO | no |
| Zep / Graphiti | bi-temporal knowledge graph, 115k → 1.6k tokens | DMR, LongMemEval | no |

All three solve conversational recall and are measured on it. ContextOS borrows Zep's
bi-temporal invalidation and Mem0's NOOP-on-identical-write, and points them at a
different problem.

---

## The benchmark

Replicating the handoff-tax study needs SWE-bench and a five-figure API budget. This
harness measures the **necessary condition** underneath it, offline and deterministically:

> **State sufficiency @ budget** — after a handoff at step *k*, does the transferred
> context contain every unit the remaining steps provably depend on?

A required unit that is missing must be re-derived or guessed, which is the mechanism
behind the measured quality drop. Sufficiency is an upper bound on handoff quality and
needs no model calls. A unit counts as delivered only if its **content survives verbatim**
— counting a mention of its address would let a lossy summariser score full marks while
having thrown the fact away.

`python -m contextos.bench --direction lateral`, 36 runs per strategy across 3 tasks,
every interior handoff point, budgets 800/1500/3000:

| strategy | recall | precision | sufficiency | mean tokens | over budget |
|---|---|---|---|---|---|
| full_replay | 63.0% | 7.0% | 38.9% | 1614 | 5.6% |
| recency | 27.0% | 2.2% | 13.9% | 1609 | 2.8% |
| summary | 86.8% | 66.4% | 75.0% | 1422 | 0.0% |
| **contextos** | **100.0%** | 64.6% | **100.0%** | **464** | 0.0% |

Full sufficiency at roughly a third of the tokens of the best baseline.

The direction asymmetry reproduces, which is the point of the design:

| direction | recall | + loss manifest | precision | mean tokens |
|---|---|---|---|---|
| escalate | 90.0% | **100.0%** | 62.3% | 465 |
| downshift | 100.0% | 100.0% | 19.1% | 1602 |
| lateral | 100.0% | 100.0% | 64.6% | 464 |

Escalation drops low-importance residue and loses 10% of required units outright — and the
loss manifest recovers all of it, because the receiving model is *told* what is missing and
can fetch it by address. That gap is the manifest earning its place.

---

## Design

| # | Decision | Grounded in |
|---|---|---|
| D1 | Direction-aware packets (`escalate` / `downshift` / `lateral`) | traj-drop wins escalation, harms downshift |
| D2 | Write-time commit — agents `put()` as they work | `compact_pre` beat post-hoc summarising |
| D3 | Retrieval, not compression — units stored verbatim | "compression gave no universal benefit" |
| D4 | Artifacts first-class: path + sha256, never a prose summary | traj-drop works *because* the working tree survives |
| D5 | Loss manifest — every omitted address is declared, with a fetch handle | silent omission is selective retrieval's dominant failure |
| D6 | Bi-temporal provenance + conflict detection | Zep's invalidation model |
| D7 | Difficulty-gated migration | "on easy tasks all escalation interfaces underperform" |
| D8 | Packets render to a neutral schema | specs are not agent-neutral (F1 0.035) |

**Budgets are enforced on the rendered packet**, not on the sum of unit tokens. Headings,
notes and the manifest are real context the receiver pays for; budgeting the units alone
and then rendering is how a system quietly ships 2× its stated budget. Structural units
(goal, constraints, open blockers, pinned) are never evicted — if they alone exceed the
budget the packet says so instead of silently truncating.

### Address space

```
/user/…      preferences, profile          /agent/…     per-agent scratch state
/project/…   architecture, decisions        /tool/…      tool results (chatter)
/task/…      goal, progress, blockers       /artifact/…  real files: path + sha256
```

### Retrieval

Four independent rankers fused by Reciprocal Rank Fusion (k=60) — BM25 scores and
address-overlap scores live on incompatible scales, so fusing by *rank* avoids
calibration entirely:

1. **address router** — query terms matched against address segments (deterministic)
2. **BM25** — SQLite FTS5 over address + value
3. **intent** — query→kind mapping, so *"what is blocking us"* surfaces blockers even
   though no blocker's text contains the word "blocking"
4. **prior** — importance × confidence × recency (7-day half-life)

Embeddings are deliberately out of scope for v1: address routing covers the exact-match
path a vector index handles badly, and BM25 covers the rest. Add them only if the
benchmark shows a gap.

---

## Usage

```python
from contextos import ContextOS

ctx = ContextOS("project.db")
ctx.put("/task/goal", "Replace cookie sessions with OAuth2", kind="goal", pinned=True)
ctx.put("/project/decisions/oauth-lib", "authlib, already vendored in the monorepo",
        kind="decision", source="architect", importance=0.9)
ctx.put_artifact("/artifact/auth/token", "src/auth/token.py", source_code)

packet = ctx.handoff(direction="escalate", budget_tokens=1200,
                     from_model="haiku-4.5", to_model="opus-4.7", difficulty=0.75)
print(packet.render())         # 13,373 stored tokens -> 634 sent
```

Conflicts are surfaced, never silently resolved:

```python
ctx.put("/project/decisions/oauth-lib", "oauthlib", source="second-opinion-agent")
ctx.conflicts()   # architect vs second-opinion-agent, both values preserved
```

The packet then carries a warning rather than picking a winner.

### CLI

```bash
contextos put /project/decisions/db "PostgreSQL 16" --kind decision --importance .9
contextos search "what database did we pick"
contextos handoff --direction escalate --budget 2000
contextos bench
```

### As an MCP server (spec 2026-07-28)

```json
{ "mcpServers": { "contextos": {
    "command": "python", "args": ["-m", "contextos.mcp_server", "--db", "./project.db"] } } }
```

Seven tools: `context_put`, `context_get`, `context_search`, `context_select`,
`context_handoff`, `context_stats`, `context_conflicts`. Raw JSON-RPC over stdio, no SDK
dependency.

---

## Live cross-model evaluation

`bench.py` measures state sufficiency — a necessary condition, offline. `live.py`
measures what sufficiency only bounds: **does the task still come out right after the
model changes?**

Each task is built so a constraint established early *changes the correct answer*. Lose
it at the handoff and the model produces a specific, predictable wrong number — so "did
the context survive" is directly observable in the output, with no LLM judge.

| task | correct | answer when a constraint was lost |
|---|---|---|
| invoice-total | 2754.00 (discount then tax) | 3240.00 (discount dropped) |
| capacity-plan | 6 nodes (headroom + standby) | 4 nodes |
| release-date | 2026-03-09 (soak, no-Friday) | 2026-03-07 |

```bash
python -m contextos.live --list-providers        # which keys .env actually has
python -m contextos.live --dry-run               # verify the harness, no API calls
python -m contextos.live --a groq --b gemini --repeats 3
```

Providers: `groq`, `cerebras`, `together`, `mistral`, `gemini`, `openrouter`,
`anthropic`. Keys are read from `.env` in the working directory; `--dry-run` needs none.
Nothing that contains a key is ever printed or logged.

Harness self-test (`--dry-run`, model B simulated as one that applies exactly the
constraints it receives — **not** a result, a check that the experiment is wired right):

| interface | solved | transfer tokens |
|---|---|---|
| raw (full trajectory) | 100% | 842 |
| traj_drop | **0%** | 28 |
| summary | 100% | 835 |
| contextos | 100% | **385** |

`traj_drop` scoring zero is the thesis in one line: it is the paper's *best* escalation
interface, and it fails here because it preserves only the working tree — and a
constraint is not a file. ContextOS matches full-trajectory correctness at 46% of the
tokens by carrying the constraint as addressable state instead.

Real numbers need real models — run the command above with your own keys.

## Honest limits

- Sufficiency is a **necessary, not sufficient** condition for handoff quality. It does not
  prove a task completes; it proves the next model was not starved of state.
- The three benchmark tasks are hand-authored with declared ground-truth dependencies.
  They are auditable but synthetic, and the suite is small.
- The live evaluation ships but **has not been run against real models here** — this
  sandbox's egress policy blocks the provider APIs. Only the `--dry-run` self-test has
  executed. Treat the live table as unmeasured until you run it yourself.
- The three live tasks are arithmetic/date reasoning, not software engineering. They
  isolate constraint survival cleanly, which is the point, but they are not SWE-bench.
- Some state genuinely does not survive a handoff: the departing model's half-formed plan,
  its implicit commitments, its reasoning trace. ContextOS carries declared state. It does
  not carry intent, and does not claim to.
- Retrieval is lexical. It will miss a paraphrase that shares no vocabulary with the stored
  unit.

## Files

```
contextos/units.py       addresses, Unit, token accounting
contextos/store.py       SQLite + FTS5, versioning, conflicts, TTL
contextos/retrieval.py   4-ranker hybrid search fused by RRF
contextos/budget.py      tiered packing + loss accounting
contextos/handoff.py     direction-aware packets, difficulty gate
contextos/bench.py       sufficiency benchmark + 3 baselines
contextos/mcp_server.py  MCP stdio server
contextos/server.py      dashboard backend (stdlib http.server)
contextos/dashboard.html the dashboard UI, one file
contextos/cli.py         command line
contextos/demo.py        end-to-end demo
tests/                   36 tests, stdlib only
PLAN.md                  the research and the reasoning behind each decision
```

## References

- [The Handoff Tax: Continuing Non-Native Trajectories in LLM Agents](https://arxiv.org/abs/2608.24358) — arXiv 2608.24358
- [Specification Portability Across LLM Development Agents](https://arxiv.org/abs/2608.21208) — arXiv 2608.21208
- [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560) — arXiv 2310.08560
- [Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory](https://arxiv.org/abs/2504.19413) — arXiv 2504.19413
- [Zep: A Temporal Knowledge Graph Architecture for Agent Memory](https://arxiv.org/abs/2501.13956) — arXiv 2501.13956
- [Model Context Protocol specification 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)
