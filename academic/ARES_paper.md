# ARES: A Stateful, Human-in-the-Loop Multi-Agent Architecture for Grounded Autonomous Research Synthesis

**Udit Sharma** · `uditsharma9981@gmail.com` · *Independent Research*

> This is a readable Markdown rendering of the paper. The typeset, citation-ready
> version - with architecture figure, algorithm, tables, and bibliography - is
> [`ARES_paper.tex`](ARES_paper.tex). Build instructions are in
> [`README.md`](README.md).

---

## Abstract

Large Language Model (LLM) "research assistants" are typically implemented as
single-shot, linear pipelines: a query is expanded, a few documents are
retrieved, and one prompt produces a summary. Such pipelines are brittle (a
single tool failure loses the entire run), opaque (the user cannot steer the line
of inquiry), and prone to hallucination (generation is only loosely tied to
retrieved evidence). We present **ARES** (*Autonomous Research & Multi-Agent
Evaluation Engine*), a stateful, graph-orchestrated system that reframes
automated literature synthesis as the coordinated behaviour of a small *simulated
research team*. ARES (i) generates diverse analyst personas and exposes a
human-in-the-loop checkpoint to steer them before any expensive work begins; (ii)
fans out an independent, retrieval-grounded expert *interview* per persona using a
map–reduce execution pattern; and (iii) synthesises the resulting cited memos into
a structured report. The system is built on LangGraph and is engineered for
resilience: bounded exponential-backoff retries, schema-validated structured
outputs with graceful fallbacks, custom channel reducers that tolerate concurrent
state writes, and SQLite-backed checkpointing that makes long-horizon runs
pausable and resumable. We describe a message *re-roling* technique that enables
robust multi-turn dialogue when both interlocutors are instantiated from the same
LLM — a common failure mode for small and locally-hosted models — and a strict
citation-grounding scheme with stable inline numbering.

**Keywords:** multi-agent systems, LLM orchestration, retrieval-augmented
generation, human-in-the-loop, LangGraph, agentic workflows, research automation.

---

## 1. Introduction

The dominant implementation pattern for automated research remains a *single
agent in a straight line*: expand the query, retrieve a handful of passages, and
emit one long generation. This pattern inherits three structural weaknesses:

1. **Fragility.** A linear script has no notion of partial progress. If a search
   API rate-limits, a model call times out, or a structured-output parse fails,
   the whole run typically aborts and its intermediate work is discarded.
2. **Opacity and lack of control.** The user provides a topic and receives an
   answer; there is no principled point at which they can inspect and redirect
   *how* the topic will be investigated before compute is spent.
3. **Weak grounding.** When retrieval and generation are only loosely coupled,
   the model freely interpolates unsupported claims, and citations — if present —
   often do not map to the passages actually used.

ARES is a response to these weaknesses. Rather than a single agent, it
orchestrates a *team*: a set of analyst personas, each of which interviews a
retrieval-grounded "expert" from a distinct angle, after which their findings are
reduced into a coherent report. The orchestration is expressed as an explicit
state graph, giving every intermediate artefact a durable home and every control
decision an inspectable edge.

### Contributions

1. **A phase-structured, graph-orchestrated architecture** unifying human-in-the-
   loop *persona steering*, a reusable interview sub-graph, and a parallel
   map–reduce fan-out over personas within a single compiled, checkpointed graph.
2. **A message re-roling technique** (`build_dialogue`) that lets a single LLM
   sustain a coherent multi-turn interview *with itself* by reconstructing a
   strict, role-alternating transcript per call — addressing an empirical failure
   mode of small/local models.
3. **A resilience layer** combining bounded-retry invocation, structured outputs
   with typed fallbacks, and a concurrency-tolerant channel reducer that together
   enable fault-tolerant, resumable long-horizon runs over an SQLite checkpoint
   store.
4. **A strict retrieval-grounding and citation scheme** with stable inline
   numbering that constrains the expert to retrieved evidence and preserves a
   faithful source list into the final memo.

ARES is provider-agnostic (cloud NVIDIA NIM endpoints or fully-local Ollama),
retrieval-agnostic (Wikipedia plus DuckDuckGo or Tavily), and ships as a single,
containerised, dependency-light Python application.

---

## 2. Background and Related Work

**Agentic LLM workflows.** The ReAct paradigm interleaves reasoning traces with
tool actions; reflection-style methods add self-critique to improve robustness.
ARES adopts the *explicit-graph* view of agent control rather than a free-form
scratchpad: control flow is a compiled directed graph, which makes checkpointing,
interruption, and resumption first-class rather than emergent.

**Multi-agent collaboration.** Generative-agent societies and role-playing
frameworks (CAMEL, AutoGen) show that decomposing a task across specialised roles
can outperform a single monolithic agent. ARES follows this intuition but
constrains it: personas are not free-roaming chat participants but *interviewers*
whose counterpart (the expert) is hard-limited to retrieved context, trading
open-ended emergence for grounding and reproducibility.

**Retrieval-augmented generation (RAG).** RAG couples a parametric model with a
non-parametric retriever to reduce hallucination and inject fresh knowledge. ARES
applies RAG *per interview turn*: each analyst question is rewritten into a search
query, evidence is retrieved and merged into a stably-numbered context window, and
the expert is instructed to answer using only that context with inline citations.

**Human-in-the-loop (HITL) control.** ARES places the human at the highest-
leverage point — *after* personas are proposed but *before* any interviews run —
so a small amount of human judgement steers a large amount of downstream compute.

**Graph orchestration substrate.** ARES is implemented on LangGraph, which
provides a `StateGraph` abstraction with typed state channels, conditional edges,
the `Send` primitive for dynamic fan-out, compiled interruption points, and
pluggable checkpointers. Our contribution is not the substrate but the
*architecture* composed on top of it and the engineering techniques that make it
robust in practice.

---

## 3. System Architecture

ARES is organised into four phases. Phases 1–3 are compiled into a single master
graph; Phase 4 is the interactive driver that streams the graph to a terminal UI.

```
             ┌────────────────┐      edits ↺
   START ──► │ create_analysts│ ◄─────────────┐
             └───────┬────────┘               │
                     ▼                         │
             ⟨ human_feedback? ⟩ ──────────────┘
                     │ accept
                     ▼
        ┌─────────────────────────────┐        Interview sub-graph (per persona)
        │  conduct_interview  (MAP)   │        ┌───────────────────────────────┐
        │  parallel, one per persona  │───────►│ ask_question                  │
        └──────────────┬──────────────┘        │   │            ↺ under cap     │
             ┌─────────┼─────────┐             │   ▼                           │
             ▼         ▼         ▼             │ ⟨sign-off?⟩──► search_context │
        write_report write_intro write_concl   │   │no             │           │
             └─────────┼─────────┘             │   │               ▼           │
                       ▼                        │   │        answer_question    │
             finalize_report (REDUCE)          │   │               │           │
                       ▼                        │   yes         ⟨turn cap?⟩     │
                      END                        │   └────┬──────────┘ cap      │
                                                 │        ▼                     │
                                                 │  save_interview ► write_section
                                                 └───────────────────────────────┘
```

### 3.1 State model

Data flows through three typed state objects, each a LangGraph channel schema:
`GenerateAnalystsState` (topic, budget, feedback, personas), `InterviewState`
(the message transcript, accumulated sources, the persona, and the emitted memo),
and the top-level `ResearchGraphState` (personas plus the collected memos and the
report parts). Channels declare *reducers* that define how concurrent or repeated
writes combine.

### 3.2 Phase 1 - Analyst persona generation with HITL steering

Given a topic and a budget *N* of analysts, ARES prompts the LLM under a
*structured-output* constraint to return exactly *N* personas, each with a name,
affiliation, role, and description of focus and motives. Personas are the
pipeline's most consequential artefact — every downstream interview inherits one —
so this call fails loudly (no silent fallback) after its retries are exhausted.

Immediately after generation, the compiled graph *interrupts before* a
deliberately empty `human_feedback` node. This pause surfaces the proposed
personas to the user, who may accept them (proceed) or supply free-text feedback,
which routes the graph back to `create_analysts` to regenerate *conditioned on*
that feedback. Any number of refinement rounds is supported. The interrupt is a
property of the compiled graph (`interrupt_before=["human_feedback"]`), not of the
driver loop, so the same checkpoint semantics hold whether the run is interactive,
scripted, or resumed from disk.

### 3.3 Phase 2 - The interview sub-graph

Each persona conducts an independent interview with an "expert" instantiated from
the same LLM. Five nodes:

- **`ask_question`** — the analyst, in character, asks one specific question,
  building on the prior answer; a sign-off sentinel signals it is done.
- **`search_context`** — the latest analyst question is deterministically located
  and rewritten into a clean query; evidence is retrieved from Wikipedia and
  (optionally) the web.
- **`answer_question`** — the expert answers using *only* the numbered retrieved
  context, citing sources inline. A two-stage safety net guarantees a non-empty
  answer.
- **`save_interview`** — the transcript is serialised into a role-labelled form.
- **`write_section`** — the interview is synthesised into a focused, cited
  markdown memo with a guaranteed *Sources* block.

Two conditional edges govern the loop. After a question, `continue_or_finish`
ends the interview early if the analyst has signed off (and at least one answer
already exists, preventing a wasted final answer to a goodbye); otherwise it
proceeds to retrieval. After an answer, `route_messages` ends the interview once
the number of expert answers reaches the turn cap, else loops back. Turn counting
is robust: it counts *tagged expert answers*, not raw message length.

### 3.4 Phase 3 - Map–reduce over personas

Once personas are accepted, `initiate_all_interviews` performs the **Map** step by
emitting one LangGraph `Send` per persona, each launching a fresh instance of the
Phase-2 sub-graph with its own seeded transcript and turn budget. These instances
execute concurrently and write memos into a shared, append-reduced `sections`
channel. The **Reduce** step fans the collected memos into three parallel writers
— body, introduction, conclusion — and a final join node stitches them, with the
topic as title, into the delivered markdown report. Using `Send` (rather than
static edges) means the parallel width is determined *at runtime* by the number of
accepted personas.

### 3.5 Phase 4 — Execution and terminal UI

The driver streams the master graph and renders progress with the `rich` library:
a banner, a persona table, live question/answer panels, search indicators, and a
final rendered-markdown report. Every UI helper degrades gracefully to plain
`print`, and the console is forced to UTF-8. CLI flags (topic, persona count,
turns, output path, thread id, `--no-feedback`) each fall back to an `ARES_*`
environment variable, making the same entry point usable interactively and in
headless pipelines.

---

## 4. Message Re-roling for Single-Model Dialogue

Because the analyst and the expert are the *same* underlying LLM playing two
roles, a subtle problem arises. LangGraph's `MessagesState` stores the whole
interview in one `messages` list. To keep speaker attribution unambiguous, ARES
stores *both* sides as `AIMessage`s and tags expert turns with `name="expert"`;
only the initial seed is a `HumanMessage`.

This internal representation is not what a chat model expects at inference time.
Chat models are trained on strictly alternating Human/Assistant turns and,
empirically, small and locally-hosted models often emit *empty* completions when
handed a history that does not end on a human turn or that contains consecutive
same-role messages. ARES resolves this with `build_dialogue`, which reconstructs a
fresh, per-call transcript from the appropriate perspective without mutating
stored state:

- **Expert perspective:** analyst questions → `Human`, own answers → `AI`, seed
  dropped so the list *ends on the question to be answered*.
- **Analyst perspective:** expert answers → `Human`, own questions → `AI`, seed
  retained as the opening human turn.

Finally, `merge_message_runs` collapses accidental consecutive same-role turns
into a strictly alternating sequence. Regardless of which role is being generated,
the model always receives a clean conversation that terminates on a human turn —
the configuration under which instruction-tuned chat models are most reliable.
This is what makes ARES runnable on modest local models, not only large hosted
endpoints.

---

## 5. Retrieval Grounding and Citation

ARES treats the expert as a *closed-book-turned-open-book* agent: it may only use
what retrieval provides. Three mechanisms enforce and preserve this.

- **Deterministic query construction.** Because the transcript is all-`AIMessage`,
  asking the model to "search the last question" is ambiguous. ARES instead walks
  the message list in reverse to *deterministically* select the latest untagged
  (analyst) message, then rewrites it into a query under a structured-output
  schema, with the raw question text as a fallback.
- **Stable numbered context.** Retrieved passages are de-duplicated by URL
  (preserving first-seen order), capped to a maximum count, and truncated to a
  per-source character budget, then rendered as `[1] Source: …` blocks. The same
  numbering is used both in the expert's context and in the *Sources* listing
  emitted with the memo, so an inline `[2]` maps to a real, listed source.
- **Instructional and structural guarantees.** The expert prompt forbids external
  facts and requires inline `[n]` citations; the section writer synthesises from
  the transcript rather than inventing facts; and if the writer omits a *Sources*
  block, ARES appends the numbered listing programmatically. Grounding is enforced
  at the prompt level and *backstopped* at the code level.

---

## 6. Resilience, Concurrency, and Persistence

- **Bounded-retry invocation.** All model calls route through two wrappers.
  `safe_invoke` retries plain generations with exponential backoff and re-raises
  only after exhausting attempts. `structured_invoke` does the same for
  schema-constrained calls but additionally accepts a *typed fallback*:
  non-critical calls (e.g. query rewriting) degrade to a safe default instead of
  aborting the run, whereas critical calls (persona generation) fail loudly. A
  transient-error classifier gives users actionable messages ("is Ollama
  running?", "check your API key") when the backend is unreachable.
- **Concurrency-tolerant reducers.** The map step runs *N* interview sub-graphs in
  one super-step. Each echoes the same scalar `max_num_turns` back to the parent,
  and a naive last-value channel rejects *N* simultaneous writes with an
  `InvalidUpdateError`. ARES defines a custom reducer, `_keep_last`, that folds
  concurrent identical writes into one value. Accumulating channels (`sources`,
  `sections`) use additive reducers so parallel contributions *merge* rather than
  clobber.
- **Checkpointing and resumption.** The master graph is compiled with an SQLite
  checkpointer keyed by a `thread_id`. Every super-step persists state to disk, so
  a run can be interrupted — by the HITL pause, a crash, or the user — and later
  *resumed* by supplying the same thread id. If the SQLite extra is unavailable,
  ARES falls back to an in-memory saver. A fresh random thread id per run prevents
  accidentally resuming a completed run.
- **Provider and retrieval abstraction.** Only the active provider's package is
  imported and only its credential validated, so a local Ollama run never requires
  a cloud key. Web search is pluggable and degrades to an empty result set rather
  than raising.

```
Algorithm - Map–reduce over analyst personas
  Require: topic t, personas A = {a1..aN}, turn cap τ
  -- Map (parallel) --
  for each a_i in A concurrently:
      seed transcript with "Start the interview: t"
      m_i ← InterviewSubgraph(a_i, τ)        # cited memo
      append m_i to sections                 # additive reducer
  -- Reduce (parallel writers, then join) --
  B ← WriteBody(sections);  I ← WriteIntro(t);  C ← WriteConclusion(t)
  return Stitch(t, I, B, C)
```

---

## 7. Implementation

ARES is a single, self-documenting Python module (~1.1k lines) depending on
LangGraph, LangChain community loaders, a provider SDK, and `rich`. Typed state
channels declare reducers that make parallel writes well-defined:

```python
class ResearchGraphState(TypedDict):
    topic: str
    max_analysts: int
    max_num_turns: Annotated[int, _keep_last]   # fold N concurrent writes -> 1
    human_analyst_feedback: str
    analysts: List[Analyst]
    sections: Annotated[list, operator.add]     # parallel memos MERGE
    introduction: str
    content: str
    conclusion: str
    final_report: str
```

**Selected configuration** (environment variables): `LLM_PROVIDER`
(`nvidia`|`ollama`), `NVIDIA_MODEL`, `OLLAMA_MODEL`, `CHECKPOINT_BACKEND`
(`sqlite`|`memory`), `SEARCH_BACKEND` (`duckduckgo`|`tavily`|`none`),
`WEB_MAX_RESULTS` (3), `MAX_SOURCES_FOR_EXPERT` (5), `MAX_SOURCE_CHARS` (1500).

The project ships with a slim, non-root Docker image and a Compose stack that can
optionally run a fully-local Ollama sidecar, so a reviewer can reproduce a run
with a single command and no local Python setup.

---

## 8. Qualitative Behaviour and Evaluation Agenda

ARES is a systems contribution; a full quantitative study is future work.

**Observed behaviour.** On representative topics (e.g. renewable-energy
microgrids), ARES reliably (i) produces personas spanning complementary angles
(technical, financial, regulatory); (ii) drives multi-turn interviews in which
each expert answer carries inline citations that resolve to the listed sources;
and (iii) synthesises a structured report whose claims trace back to retrieved
evidence. The HITL step demonstrably changes the run: feedback such as "add an
economist" regenerates the persona set before any interview cost is incurred.

**Proposed metrics.**

1. **Citation faithfulness** — fraction of inline `[n]` citations whose claim is
   supported by source *n* (human or LLM-as-judge adjudicated).
2. **Coverage / diversity** — semantic spread of persona foci and of the
   questions asked, versus a single-agent baseline.
3. **Robustness** — completion rate under injected faults (search timeouts, empty
   generations, parse failures) with and without the resilience layer.
4. **Steerability** — effect of HITL feedback on final-report relevance, via A/B
   runs with and without the checkpoint.

**Baselines.** (a) A single-shot RAG summariser; (b) ARES with one persona and one
turn (ablating the team); (c) ARES with the resilience layer disabled.

---

## 9. Limitations

Retrieval quality bounds answer quality: a keyless web backend and short
per-source budgets can miss or truncate relevant evidence. The expert and analyst
share one model, so systematic biases are correlated across roles rather than
independent. Grounding is enforced by instruction and structure but not *proven* —
a determined model can still paraphrase beyond its sources, which is why
citation-faithfulness measurement is prioritised. Parallel fan-out multiplies
token and latency cost roughly linearly in the number of personas. Finally, the
synthesis stage writes intro/body/conclusion independently; it does not yet
perform a global consistency or contradiction-resolution pass across memos.

---

## 10. Future Work

Priorities: (i) the quantitative evaluation above, including an automated
citation-faithfulness harness; (ii) a cross-memo reconciliation node that detects
and resolves contradictions before finalisation; (iii) heterogeneous role models
(a stronger model for synthesis, a cheaper one for interviews); (iv) richer
retrieval (vector stores, PDF/arXiv ingestion, recency-aware ranking); and (v) an
evaluation/critique agent — the "E" in ARES's name — that scores each memo and can
trigger targeted re-interviews, closing an outer quality-control loop.

---

## 11. Conclusion

ARES reframes automated research synthesis from a fragile linear script into the
coordinated behaviour of a small, steerable, resilient research team. By
expressing the workflow as an explicit, checkpointed state graph, it earns durable
intermediate state, first-class interruption and resumption, and a principled
human-in-the-loop steering point. Its engineering contributions — a single-model
dialogue re-roling technique, concurrency-tolerant channel reducers, a
retry-and-fallback resilience layer, and a strict numbered-citation grounding
scheme — are individually small but collectively transform a demo-grade agent into
a system that survives real-world failures and remains transparent to its user.

---

## References

1. S. Yao et al. "ReAct: Synergizing Reasoning and Acting in Language Models." *ICLR*, 2023.
2. N. Shinn et al. "Reflexion: Language Agents with Verbal Reinforcement Learning." *NeurIPS*, 2023.
3. J. S. Park et al. "Generative Agents: Interactive Simulacra of Human Behavior." *UIST*, 2023.
4. G. Li et al. "CAMEL: Communicative Agents for 'Mind' Exploration of LLM Society." *NeurIPS*, 2023.
5. Q. Wu et al. "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation." *arXiv:2308.08155*, 2023.
6. P. Lewis et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *NeurIPS*, 2020.
7. LangChain. "LangGraph: Building Stateful, Multi-Actor Applications with LLMs." 2024.
8. F. Petroni et al. "KILT: a Benchmark for Knowledge Intensive Language Tasks." *NAACL*, 2021.
9. S. Es et al. "RAGAS: Automated Evaluation of Retrieval Augmented Generation." *arXiv:2309.15217*, 2023.
