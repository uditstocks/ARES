# ARES — Complete Code Documentation

> A plain-language but technically deep walkthrough of [`ARES.py`](../ARES.py) —
> written for an engineering student who knows Python but may be new to
> LangGraph and multi-agent LLM systems.

---

## Table of Contents

1. [What is ARES?](#1-what-is-ares)
2. [Concepts You Need First](#2-concepts-you-need-first)
3. [The Big Picture — Architecture](#3-the-big-picture--architecture)
4. [Startup: Imports & Environment Setup](#4-startup-imports--environment-setup)
5. [Configuration Layer](#5-configuration-layer)
6. [Reliability Helpers — Retries & Structured Output](#6-reliability-helpers--retries--structured-output)
7. [Phase 1 — Creating Analyst Personas (Human-in-the-Loop)](#7-phase-1--creating-analyst-personas-human-in-the-loop)
8. [Phase 2 — The Interview Sub-Graph](#8-phase-2--the-interview-sub-graph)
9. [Phase 3 — The Master Graph (Map → Reduce)](#9-phase-3--the-master-graph-map--reduce)
10. [Phase 4 — The Interactive CLI](#10-phase-4--the-interactive-cli)
11. [Key Engineering Decisions (and Why They Matter)](#11-key-engineering-decisions-and-why-they-matter)
12. [Configuration Reference](#12-configuration-reference)
13. [Glossary](#13-glossary)

---

## 1. What is ARES?

**ARES (Autonomous Research & Multi-Agent Evaluation Engine)** takes one input —
a research *topic* typed by you — and produces one output — a structured,
citation-backed markdown *report* saved to disk.

In between, it simulates a small research team:

1. It invents a panel of **analyst personas** (e.g., "a grid architect from
   NREL", "a policy lead"), each looking at the topic from a different angle.
2. Each analyst **interviews an "expert"** — which is the *same* LLM playing a
   different role — over several question–answer turns. Before every answer,
   the system fetches real material from **Wikipedia and web search**, and the
   expert is only allowed to answer from that material, citing it as `[1]`, `[2]`.
3. Each interview is condensed into a **memo** (a cited markdown section).
4. All memos are merged into a final report with an introduction, body, and
   conclusion.

The entire orchestration is a **state machine** built with
[LangGraph](https://github.com/langchain-ai/langgraph), so the run can pause
(for human feedback), resume after a crash (via SQLite checkpoints), and run
the interviews **in parallel**.

Everything lives in a single file, `ARES.py` (~1100 lines), organised into four
clearly-bannered phases.

---

## 2. Concepts You Need First

If you already know LangGraph, skip to [Section 3](#3-the-big-picture--architecture).

### 2.1 LLM chat models and "roles"

A chat LLM consumes a list of messages, each with a role:

- **System** — instructions that set behaviour ("You are an analyst…").
- **Human** — what the user said.
- **AI** — what the model previously said.

The model then generates *the next AI message*. This matters enormously in
ARES because both the "analyst" and the "expert" are AI-generated — so the code
has to *re-label* who is "Human" and who is "AI" depending on whose turn it is
(see [Section 8.4](#84-the-re-roling-trick-build_dialogue)).

### 2.2 LangGraph in 60 seconds

LangGraph lets you build a program as a **directed graph**. You assemble it by
adding nodes and edges to a *builder* object (`StateGraph`), then call
`.compile()` to turn it into a runnable app — options like the checkpointer
and `interrupt_before` are passed at this compile step, which is what phrases
like "the graph is compiled with…" mean later in this doc. The building
blocks:

- **State** — a typed dictionary (a `TypedDict` class) that flows through the
  graph. It is the *only* way data moves between steps.
- **Node** — a plain Python function. It receives the current state and
  returns a *partial* update (a dict with only the keys it changed).
- **Edge** — "after node A, run node B". Added with `add_edge`.
- **Conditional edge** — a function that inspects the state and *returns the
  name* of the next node. This is how loops and branching work.
- **Reducer** — an annotation on a state key that says *how* to merge a node's
  update into the existing value. `Annotated[list, operator.add]` means
  "append to the list" instead of "replace the list". Reducers are what make
  parallel writes safe.
- **Checkpointer** — a storage backend that saves the state after every step,
  so a run can be paused, inspected, and resumed by `thread_id`.
- **`interrupt_before`** — compile-time flag: "pause the graph *before*
  executing this node". This is LangGraph's human-in-the-loop mechanism.
- **`Send`** — a special return value from a conditional edge that says "launch
  this node with *this specific state*". Returning a *list* of `Send` objects
  launches N copies of a node in parallel — LangGraph's **map** primitive
  (the "map" of Map–Reduce, explained in Section 2.4 below).

### 2.3 Structured output

LLMs return free text by default. `llm.with_structured_output(SomePydanticModel)`
forces the model to return JSON matching a [Pydantic](https://docs.pydantic.dev/)
schema, and parses/validates it into a Python object. ARES uses this wherever
it needs *data* (a list of analysts, a search query) rather than *prose*.

### 2.4 Map–Reduce

A classic parallel-computing pattern:

- **Map** — apply the same function to many inputs independently (here: run
  one interview per analyst, all in parallel).
- **Reduce** — combine all the results into one output (here: merge all memos
  into a single report).

---

## 3. The Big Picture — Architecture

### 3.1 The four phases

```
 PHASE 1                PHASE 2 (sub-graph, ×N in parallel)         PHASE 3
 ────────               ────────────────────────────────────        ────────
 create_analysts   ┌──> ask_question ─> search_context ─┐           write_report
       │           │                                    │           write_introduction
       v           │         ┌──────────────────────────┘           write_conclusion
 human_feedback ───┤         v                                        ^     │
 (pause for user)  │    answer_question ──loop back──> ask_question   │     v
       │           │         │ (turn cap reached)                     │ finalize_report
       └─feedback──┘         v                                        │     │
        (regenerate)    save_interview ─> write_section ──memos───────┘     v
                                                                        report.md

 PHASE 4: an interactive CLI (rich terminal UI) drives all of the above.
```

### 3.2 The three state objects

Data flows through three `TypedDict` state classes, one per graph:

| State class            | Used by                    | Key fields                                                                 |
|------------------------|----------------------------|----------------------------------------------------------------------------|
| `GenerateAnalystsState`| standalone analyst graph   | `topic`, `max_analysts`, `human_analyst_feedback`, `analysts`              |
| `InterviewState`       | interview sub-graph        | `messages`, `max_num_turns`, `sources`, `analyst`, `interview`, `sections` |
| `ResearchGraphState`   | master graph               | `topic`, `max_analysts`, `max_num_turns`, `human_analyst_feedback`, `analysts`, `sections` (accumulated from all interviews), `introduction`, `content`, `conclusion`, `final_report` |

Note what the master state does **not** contain: the interview-local fields
`messages`, `sources`, `analyst`, and `interview`. Each parallel interview
gets its own private `InterviewState`, which the master graph must construct
and hand over explicitly via `Send` (see
[Section 9.2](#92-the-map-step-initiate_all_interviews)). Results flow back
the other way through the two key names the states *share* — `sections` and
`max_num_turns` — a mechanism explained in
[Section 9.1](#91-researchgraphstate--how-sub-graph-results-flow-back-up).

---

## 4. Startup: Imports & Environment Setup

Startup does three defensive things. The first two are environment tweaks
that must run **before** any LangChain import; the third — the optional
`rich` UI — is imported afterwards, alongside the other third-party packages:

### 4.1 Disabling TensorFlow (`USE_TF=0`)

```python
os.environ.setdefault("USE_TF", "0")
```

Importing `langchain_community` can transitively import the `transformers`
library, which by default eagerly imports TensorFlow. ARES never uses
TensorFlow, and a broken TF/protobuf installation would crash the program at
*import time*. Setting `USE_TF=0` tells `transformers` not to touch TF at all.
The ordering is critical — an environment variable read at import time must be
set *before* that import happens.

### 4.2 Forcing UTF-8 console output

```python
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
```

On older Windows configurations the console uses a legacy code page (cp1252)
that cannot encode emoji like 🔬 — printing one raises `UnicodeEncodeError`.
Reconfiguring the streams to UTF-8 fixes this. It is wrapped in
`try/except Exception: pass` because some environments (redirected output,
embedded interpreters) don't support `reconfigure`, and failing there would be
worse than losing emoji.

### 4.3 Optional `rich` UI

The `rich` library (panels, tables, spinners, markdown rendering in the
terminal) is imported inside a `try/except`. If it's missing, a module-level
flag `_RICH = False` is set, and every UI helper in Phase 4 falls back to plain
`print()`. **The program never hard-depends on its prettiness.**

Finally, `load_dotenv()` loads a local `.env` file so all the configuration
below can be read from environment variables.

---

## 5. Configuration Layer

### 5.1 LLM provider selection

```python
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "nvidia").lower()
```

Three backends are supported, chosen at startup:

- **`nvidia`** (default) — `ChatNVIDIA` from `langchain_nvidia_ai_endpoints`,
  model `meta/llama-3.3-70b-instruct` by default. Requires `NVIDIA_API_KEY`
  (an `assert` fails fast if it's missing).
- **`ollama`** — `ChatOllama` for a fully local model, `llama3.1:8b` by default.
- **`openrouter`** — `ChatOpenAI` from `langchain_openai` pointed at
  OpenRouter's OpenAI-compatible endpoint (`OPENROUTER_BASE_URL`, default
  `https://openrouter.ai/api/v1`). Requires `OPENROUTER_API_KEY`; the model
  defaults to `openai/gpt-4o-mini` and can be any OpenRouter-hosted id
  (e.g. `openai/gpt-4o`). Prefer models with function-calling support,
  since structured output rides on tool calling.

Note the **lazy import** pattern: only the *active* provider's package is
imported, and only its credential is validated. Running locally with Ollama
never requires an NVIDIA key, and vice-versa. An unknown provider raises
`ValueError` immediately rather than failing mysteriously later.

The resulting `llm` object is a module-level global used by every node.

### 5.2 Search backend and context caps

```python
SEARCH_BACKEND = os.getenv("SEARCH_BACKEND", "duckduckgo").lower()
WEB_MAX_RESULTS = int(os.getenv("WEB_MAX_RESULTS", "3"))
MAX_SOURCES_FOR_EXPERT = int(os.getenv("MAX_SOURCES_FOR_EXPERT", "5"))
MAX_SOURCE_CHARS = int(os.getenv("MAX_SOURCE_CHARS", "1500"))
```

The last two exist because of a subtle problem: **sources accumulate every
interview turn** (the state uses an appending reducer, Section 8.1). After a
few turns the raw source text could exceed the model's context window. So when
sources are rendered into a prompt, only the first `MAX_SOURCES_FOR_EXPERT`
are shown, each truncated to `MAX_SOURCE_CHARS` characters. The *full* list
still lives in state — only the prompt view is capped.

---

## 6. Reliability Helpers — Retries & Structured Output

LLM calls fail in practice: networks flake, local models time out, small
models produce malformed JSON. ARES wraps *every* LLM call in one of two
helpers.

### 6.1 `safe_invoke(messages, retries=2, label="llm")`

Calls `llm.invoke(messages)`. On any exception it waits and retries with
**exponential backoff** — the wait doubles each attempt (`2**attempt`:
1 s, then 2 s). After `retries + 1` total attempts (default 3), it re-raises
the last exception. Exponential backoff is the standard way to avoid hammering
an already-struggling backend.

### 6.2 `structured_invoke(schema, messages, retries=2, fallback=None)`

Same retry loop, but wraps the model with `llm.with_structured_output(schema)`
so the return value is a validated Pydantic object. The important extra is the
**`fallback` parameter**:

- `fallback` given → if *all* attempts fail, return the fallback instead of
  raising. Used for the search-query rewrite (Section 8.3), where a failure
  should merely degrade the search — the code falls back to searching with
  the raw question text — not kill a 10-minute run.
- `fallback=None` → re-raise. Used for analyst generation, because
  *analysts are essential* — continuing with none would be pointless.

This asymmetry — *crash on essential failures, degrade on optional ones* — is a
recurring design theme in ARES.

### 6.3 `is_connection_error(exc)`

A heuristic that checks the exception text against substrings like
`"connection"`, `"timeout"`, `"refused"`. It's used only at the very bottom of
the file to print a friendly "is Ollama running?" hint instead of a raw
traceback when the LLM backend is unreachable.

---

## 7. Phase 1 — Creating Analyst Personas (Human-in-the-Loop)

### 7.1 Data models

```python
class Analyst(BaseModel):
    affiliation: str
    name: str
    role: str
    description: str

    @property
    def persona(self) -> str: ...   # formatted text block for prompts

class Perspectives(BaseModel):
    analysts: List[Analyst]
```

`Analyst` is a Pydantic model describing one persona. `Perspectives` is a thin
wrapper whose only job is to be the **structured-output schema** — the LLM is
forced to return `{"analysts": [...]}` which Pydantic validates into real
`Analyst` objects. The `persona` property renders the analyst as a text block
that gets injected into interview prompts, keeping the questioner "in
character".

### 7.2 The `create_analysts` node

Formats the `analyst_instructions` prompt with the topic, the requested count,
and **any human feedback already in state**, then calls `structured_invoke`
with `Perspectives` as the schema and *no fallback* (failure raises — see
Section 6.2). It returns `{"analysts": [...]}` as its state update.

### 7.3 The `human_feedback` node — a deliberate no-op

```python
def human_feedback(state):
    pass
```

This looks pointless but is the heart of the human-in-the-loop design. The
graph is compiled with:

```python
interrupt_before=["human_feedback"]
```

meaning LangGraph **pauses the run just before executing this node** and
returns control to the caller. The CLI (Phase 4) then shows the analysts to
the user, collects feedback, and writes it into the paused state with
`update_state(..., as_node="human_feedback")` — i.e., "pretend the
`human_feedback` node produced this update". When the graph resumes, a
conditional edge reads that feedback and decides where to go next:

- feedback present → back to `create_analysts` (regenerate with the feedback
  in the prompt);
- feedback empty → proceed to the interviews.

### 7.4 The checkpointer (`make_checkpointer`)

Pausing and resuming requires persistence. `make_checkpointer()` returns:

- **SQLite** (default) — `SqliteSaver` on `ares_checkpoints.sqlite`, opened
  with `check_same_thread=False` because LangGraph may touch the connection
  from multiple threads. Runs survive process restarts.
- **Memory** — `MemorySaver`, RAM-only, for quick experiments.

It degrades gracefully: if the optional `langgraph-checkpoint-sqlite` package
is missing or the DB can't open, it *warns and falls back* to in-memory rather
than crashing. One shared checkpointer instance (`memory`) is used by both
compiled graphs.

### 7.5 The standalone analyst graph

`analyst_only_graph` wires just `create_analysts → human_feedback` with a
feedback loop. **The full pipeline does not use it** — it exists for debugging
and LangGraph Studio. The real pipeline embeds the same two nodes inside the
master graph (Phase 3).

---

## 8. Phase 2 — The Interview Sub-Graph

This is the most intricate part of the file. One instance of this graph runs
**per analyst**, and the master graph launches all instances in parallel.

### 8.1 `InterviewState`

```python
class InterviewState(MessagesState):
    max_num_turns: int
    sources: Annotated[list, operator.add]
    analyst: Analyst
    interview: str
    sections: list
```

- It **inherits from `MessagesState`**, which provides a `messages` key with
  LangGraph's message-appending reducer — every node that returns
  `{"messages": [msg]}` *appends* rather than overwrites.
- `sources` uses `Annotated[list, operator.add]`, so retrieved documents
  **accumulate across turns** (turn 3's answer can still cite turn 1's
  sources).
- A crucial convention: **both** the analyst's questions and the expert's
  answers are stored as `AIMessage`s in the same `messages` list. They are
  distinguished by a name tag — expert answers get `.name = "expert"`,
  analyst questions stay untagged. The single `HumanMessage` in the list is
  the seed: *"Start the interview regarding: {topic}"*.

### 8.2 Node `ask_question` (`generate_question`)

The analyst asks the next question, in character. The system prompt injects
the analyst's persona and imposes strict output rules (only the question, no
speaker labels, no stage directions) — necessary because small models love to
add "*(nodding)* Aaradhya Jain:" prefixes. The prompt also defines the
**termination sentence**: when the analyst feels done, it must reply exactly
*"Thank you so much for your help!"* — which the router (Section 8.6) detects.

Before invoking the LLM, the transcript is passed through `build_dialogue`
(Section 8.4) with `perspective="analyst"`.

### 8.3 Node `search_context` — retrieval

Two steps:

1. **Find the question.** `_latest_analyst_question` walks `messages`
   *backwards* and returns the first `AIMessage` whose name is **not**
   `"expert"` — i.e., the newest analyst question — falling back to the seed
   `HumanMessage`. This is done *deterministically in Python* instead of
   asking the LLM to "find the last question", which is unreliable when the
   history is all AI messages.
2. **Rewrite it into a search query.** A `structured_invoke` call with the
   tiny `SearchQuery` schema turns a long conversational question into a
   short, clean query. Its fallback is an *empty* query — and the code then
   falls back further to the first 100 characters of the raw question. Only
   if both are empty is the search skipped.

Then it retrieves from two places:

- **Wikipedia** — `WikipediaLoader(query, load_max_docs=2)`, wrapped in
  try/except so a Wikipedia hiccup costs nothing but a log line.
- **Web search** (`search_web`) — behind `SEARCH_BACKEND`:
  - `duckduckgo` (default, no API key) via `DuckDuckGoSearchResults`;
  - `tavily` (needs `TAVILY_API_KEY`; skips politely if unset);
  - `none`/`off`/empty → disabled, returns `[]`.

  Every failure path returns `[]` — **search is an enhancement, never a
  dependency**. Results are normalised to `{"source": url, "content": text}`
  dicts regardless of backend.

Three details worth noticing:

- `SEARCH_BACKEND=none` disables only the **web** half — the Wikipedia lookup
  always runs (and its `load_max_docs=2` is hardcoded, not configurable).
- If everything fails and no sources exist at all, the expert's prompt gets
  the literal text *"No source material was retrieved."* — and since the
  expert may only answer from context, its replies degrade into explicit
  non-answers rather than hallucinations.
- In the DuckDuckGo path, if the tool returns a raw string instead of a list,
  the whole blob is stored under the generic source label `"web"`. Because
  de-duplication later keys on that label, any subsequent `"web"` item is
  dropped as a "duplicate", and the citation renders as `[n] web` with no
  real URL.

The node returns `{"sources": [...]}`; the reducer appends them to state.

### 8.4 The re-roling trick: `build_dialogue`

This is the cleverest function in the file, and it solves a real problem:

> Chat models expect a strict Human/AI alternation ending on a **Human** turn.
> But ARES stores both sides of the interview as **AI** messages. If you feed
> a model a transcript that ends with its "own" AI message, small models often
> return empty output.

`build_dialogue(messages, perspective)` builds a **new**, re-labelled copy of
the transcript from one participant's point of view:

| Message in state                 | `perspective="expert"`     | `perspective="analyst"`     |
|----------------------------------|----------------------------|-----------------------------|
| Seed `HumanMessage`              | **dropped**                | kept as opening Human turn  |
| Analyst question (untagged AI)   | becomes `HumanMessage`     | stays `AIMessage`           |
| Expert answer (`name="expert"`)  | stays `AIMessage`          | becomes `HumanMessage`      |

For the expert, the seed is dropped so the re-roled list **ends on the latest
question** (a Human turn — exactly what a chat model wants to respond to).
Finally, `merge_message_runs` collapses any accidental consecutive same-role
messages into one, guaranteeing strict alternation. The original `messages`
list in state is never mutated.

### 8.5 Node `answer_question` (`generate_answer`)

The expert answers using **only** the retrieved context:

1. `_format_numbered_sources` renders the accumulated sources as numbered
   blocks — `[1] Source: <url>\n<content>` — after de-duplicating
   (`_dedup_sources` keys on the URL, falling back to the first 60 characters
   of content when a source has no URL; first-seen order preserved), capping
   the count at `MAX_SOURCES_FOR_EXPERT`, and truncating each to
   `MAX_SOURCE_CHARS`. Because the cap keeps the *first* N sources in stable
   order, the numbering `[1]`, `[2]` stays consistent across turns — so
   citations in early answers don't silently change meaning later.
   The flip side: a single turn can already retrieve up to 5 sources
   (2 Wikipedia + 3 web — exactly the default cap), so sources fetched in
   *later* turns often never make it into the expert's prompt at all. The
   expert may answer turn 3's question grounded mostly in turn 1's material,
   and the memo's Sources list (Section 8.7) inherits the same capped view.
2. The system prompt embeds the numbered context and hard rules: *answer only
   from the context, cite inline as `[1]`,`[2]`, no speaker labels*.
3. The transcript is re-roled with `perspective="expert"` and the model is
   invoked via `safe_invoke`.
4. **Empty-answer safety net:** if the model returns an empty string, it is
   nudged once with an explicit "Please answer the question above now…"
   message; if it's *still* empty, a placeholder answer is inserted. This
   guarantees the transcript and the turn counter never carry an empty answer.
5. The answer is tagged `answer.name = "expert"` — this tag is what the
   routing logic and re-roling depend on.

### 8.6 Routing — how the interview loop ends

Two conditional edges control the loop:

- **`continue_or_finish`** (after `ask_question`): if the analyst's latest
  message contains the termination phrase *"Thank you so much for your help"*
  **and** at least one expert answer already exists, jump straight to
  `save_interview` — this avoids wasting a search + answer round on a goodbye.
  The "at least one answer" guard (`_num_expert_answers(...) >= 1`) forces a
  minimum of one real Q&A round even if the model signs off immediately.
  Otherwise → `search_context`.
- **`route_messages`** (after `answer_question`): count expert answers
  (`_num_expert_answers` counts `AIMessage`s tagged `"expert"`); if the count
  has reached `max_num_turns`, go to `save_interview`; otherwise loop back to
  `ask_question`.

Note a deliberate asymmetry: the *prompt* demands the exact sentence "Thank
you so much for your help!" and nothing else, but the *code* only checks that
the phrase appears somewhere in the last message (a substring match). Strict
prompt, lenient check — small models rarely comply exactly, so the code meets
them halfway.

So an interview ends either by **voluntary sign-off** or by hitting the
**turn cap**, whichever comes first.

### 8.7 Nodes `save_interview` and `write_section`

- **`save_interview`** converts the message list into a plain-text transcript
  with explicit `Analyst:` / `Expert:` labels (skipping the seed message).
  This gives the section writer an unambiguous document to work from.
- **`write_section`** asks the LLM (as a "technical writer") to produce a
  memo in a fixed markdown structure — `## title`, `### Summary` (~300 words),
  `### Sources` — synthesising **from the transcript**, citing the same
  numbered sources. A post-processing guard appends a real `### Sources`
  block if the model forgot one, so downstream steps can rely on it existing.

### 8.8 Graph wiring

```python
START → ask_question
ask_question  → (continue_or_finish) → search_context | save_interview
search_context → answer_question
answer_question → (route_messages)  → ask_question | save_interview
save_interview → write_section → END
```

The sub-graph is compiled **without its own checkpointer** — when it runs as a
node inside the master graph, the parent's checkpointer persists the whole
tree. (Giving a child graph its own checkpointer is a known LangGraph
anti-pattern.)

---

## 9. Phase 3 — The Master Graph (Map → Reduce)

### 9.1 `ResearchGraphState` — how sub-graph results flow back up

Start with the mechanism, because it explains everything else here: when a
compiled sub-graph runs as a node, LangGraph maps the sub-graph's output onto
the parent state **by matching key names**. `InterviewState` and
`ResearchGraphState` share exactly two keys — `sections` and `max_num_turns` —
so those two values flow up to the parent when an interview finishes, while
the interview's `messages`, `sources`, `analyst`, and `interview` are simply
discarded at the parent level.

One more piece of vocabulary: internally, every state key is backed by a
*channel*. A key with no reducer uses the default channel type (`LastValue`),
which allows exactly **one write per execution round** — LangGraph runs
graphs in lock-step rounds called *super-steps*. With that, the two special
keys make sense:

- `sections: Annotated[list, operator.add]` — each finishing interview writes
  its memo here; the appending reducer is what makes the **fan-in** (the
  point where parallel branches merge back together) work: N parallel writes
  concatenate instead of colliding.
- `max_num_turns: Annotated[int, _keep_last]` — a subtle fix. Because
  `max_num_turns` is a shared key, each interview *echoes* it back to the
  parent when it finishes. With N interviews finishing in the same
  super-step, the parent receives N simultaneous writes to one scalar key —
  which a default `LastValue` channel rejects with `InvalidUpdateError`. The
  `_keep_last(current, new)` reducer simply returns `new`, folding the N
  identical values down to one and keeping LangGraph happy.

### 9.2 The Map step: `initiate_all_interviews`

This conditional edge (out of `human_feedback`) does two jobs:

1. **Feedback loop** — if `human_analyst_feedback` is set, return the string
   `"create_analysts"`: regenerate the personas.
2. **Fan-out** — otherwise return a **list of `Send` objects**, one per
   analyst:

```python
return [
    Send("conduct_interview", {
        "analyst": analyst,
        "messages": [HumanMessage(content=f"Start the interview regarding: {topic}")],
        "max_num_turns": max_num_turns,
        "sources": [],
        "sections": []
    }) for analyst in state["analysts"]
]
```

Each `Send` launches one copy of the interview sub-graph **with its own
private input state** — its own analyst, its own fresh seed message, its own
empty `sources`. LangGraph runs all of them concurrently. This is the "Map"
in Map–Reduce. (`max_num_turns` defaults to 3 if absent.)

### 9.3 The Reduce step: writing the report

Once **all** interviews complete, three writer nodes run **in parallel** —
this works because `conduct_interview` has an edge to each of them:

- `write_report` — synthesises the main body from all memos (with the
  explicit instruction *not* to write "Analyst 1 said…").
- `write_introduction` — a ~100-word intro from the topic.
- `write_conclusion` — a ~100-word conclusion from the topic.

They join at `finalize_report` via a **list edge**:

```python
master_builder.add_edge(["write_conclusion", "write_report", "write_introduction"], "finalize_report")
```

A list edge means "run the destination only after *all* listed sources have
completed" — a synchronisation barrier. `finalize_report` then does no LLM
work at all: it just concatenates strings into the final markdown
(`# topic`, `## Introduction`, `## Insights`, `## Conclusion`).

### 9.4 Full master-graph wiring

```python
START → create_analysts → human_feedback  (⏸ interrupt happens here)
human_feedback → (initiate_all_interviews) → create_analysts | Send×N conduct_interview
conduct_interview → write_report, write_introduction, write_conclusion  (parallel)
[all three] → finalize_report → END
```

Compiled with `interrupt_before=["human_feedback"]` and the shared
checkpointer — both are required for the pause/resume cycle in Phase 4.

---

## 10. Phase 4 — The Interactive CLI

### 10.1 UI helpers

`display_analysts`, `ui_banner`, `ui_status`, `ui_print`, `ui_rule`, `ui_ask`,
`ui_qa`, `ui_node_done` — every one checks the `_RICH` flag and falls back to
plain `print()`/`input()` if `rich` is unavailable, stripping `[bold cyan]`-style
markup with a regex so plain output stays clean. `ui_status` is a context
manager: with rich it shows a live spinner while a blocking step runs; without
it, it just prints the message once.

### 10.2 CLI arguments (`parse_args`)

| Flag              | Env fallback        | Default               | Meaning                                  |
|-------------------|---------------------|-----------------------|------------------------------------------|
| `--topic`         | `ARES_TOPIC`        | *(interactive prompt)*| Research topic                            |
| `--max-analysts`  | `ARES_MAX_ANALYSTS` | 3                     | Number of personas                        |
| `--max-turns`     | `ARES_MAX_TURNS`    | 3                     | Expert answers per interview              |
| `--thread-id`     | `ARES_THREAD_ID`    | random 8-hex chars    | Checkpoint thread; reuse to resume a run  |
| `--output`        | `ARES_OUTPUT`       | `research_report.md`  | Where the report is written               |
| `--no-feedback`   | —                   | off                   | Accept first analysts without prompting   |

Every flag except `--no-feedback` falls back to an `ARES_*` environment
variable, so the same script works interactively and in scripted/CI runs
(`--no-feedback` is a pure command-line switch — there is no
`ARES_NO_FEEDBACK`).

### 10.3 `main()` — step 1: the feedback loop

A fresh `thread_id` is generated per run (`uuid.uuid4().hex[:8]`) unless one
is passed — deliberately, so a *persistent* SQLite checkpointer doesn't
accidentally resume last week's finished run. Passing `--thread-id` is the
explicit way to resume — with a caveat: `main()` always sends a fresh input
on its first stream call and assumes the run is paused at the
analyst-feedback checkpoint. Resuming a run that already *finished* yields no
new `finalize_report` event, so the CLI ends with the "Report was empty"
warning instead of re-printing the old report.

Then a `while True` loop drives analyst generation:

1. **First iteration** streams the graph with the real input
   (`{"topic", "max_analysts", "max_num_turns"}`); **later iterations** stream
   with `input=None`, which means *"resume from the checkpoint"*.
2. Streaming uses `stream_mode="values"` — each event is the **full state
   snapshot** — and the loop grabs the latest `analysts` value to display.
3. The graph pauses at `human_feedback` (the interrupt). The CLI asks the user
   for feedback (`Enter` = accept).
4. The decision is recorded with:

   ```python
   master_graph.update_state(thread, {"human_analyst_feedback": user_feedback},
                             as_node="human_feedback")
   ```

   `as_node="human_feedback"` makes LangGraph treat the update *as if the
   interrupted node produced it*, so the conditional edge
   `initiate_all_interviews` fires next with the fresh value. Note that this
   write happens even when the user *accepts* (value `None`) — and it is
   load-bearing, not cosmetic: state persists across feedback rounds, so
   after a regeneration the previous feedback string is still sitting in
   `human_analyst_feedback`. Without overwriting it with `None`,
   `initiate_all_interviews` would see the stale feedback and loop back to
   `create_analysts` forever.
5. Feedback given → loop again (regenerate). Feedback empty → break.
   With `--no-feedback`, the loop records `None` and breaks immediately.

### 10.4 `main()` — step 2: streaming the interviews live

```python
for namespace, event in master_graph.stream(None, thread,
                                            stream_mode="updates", subgraphs=True):
```

Two details matter:

- `stream_mode="updates"` — events are **per-node deltas** (`{node_name:
  update_dict}`), not full state snapshots. Right choice for a live progress
  feed.
- `subgraphs=True` — without this, everything inside `conduct_interview`
  would be invisible (you'd see one opaque node run for minutes). With it,
  the inner nodes (`ask_question`, `search_context`, `answer_question`)
  surface, and each event comes as a `(namespace, event)` tuple. The
  `namespace` identifies *which* parallel interview an event belongs to —
  but `main()` ignores it, so with several interviews running at once the
  question/answer panels from different analysts interleave in the terminal
  with no label saying which analyst is speaking.

The event loop pattern-matches on the node name: questions render in a cyan
panel, answers in green, searches as a dim log line, `finalize_report`
captures the final markdown, and anything else prints a ✓-done line.
`__interrupt__` pseudo-events and non-dict updates are handled defensively.

### 10.5 Final output & error handling

The finished report is rendered (rich `Markdown` in a panel, or raw text) and
written to `args.output` in UTF-8. An empty report prints a warning instead
of writing an empty file.

The `if __name__ == "__main__"` block catches:

- `KeyboardInterrupt` → clean "Interrupted by user" line.
- Any exception matching `is_connection_error` → a friendly, provider-specific
  hint ("Is Ollama running? Try `ollama serve`…" / "Check NVIDIA_API_KEY…").
- Everything else → re-raised, because unexpected bugs *should* show a
  traceback.

---

## 11. Key Engineering Decisions (and Why They Matter)

1. **Both interview roles are AI messages, disambiguated by a `name` tag.**
   Alternative designs (two separate message lists, or custom message types)
   complicate LangGraph's built-in message reducer. One list + a tag keeps
   state simple; `build_dialogue` pays the re-roling cost only at invoke time.

2. **Deterministic code over LLM judgement wherever possible.** Finding the
   latest question, counting turns, deduplicating sources, stitching the
   final report — all plain Python. The LLM is only used where language
   generation is actually needed. This makes the control flow testable and
   predictable.

3. **Graceful degradation with an explicit criticality hierarchy.** Search
   failing → empty list, run continues. Rich missing → plain text. SQLite
   missing → in-memory. But analyst generation failing → hard crash, because
   nothing downstream makes sense without analysts.

4. **Stable citation numbering.** Sources are deduped and capped *keeping
   first-seen order*, so `[2]` in turn 1 still means the same document in
   turn 3. Sorting or re-ranking sources between turns would silently corrupt
   earlier citations.

5. **Prompt-size discipline.** Accumulating sources are capped (count and
   per-source length) only in the *rendered prompt*, never in state — the data
   is preserved; the context window is protected. The trade-off: with the
   default caps, later-turn sources may never be shown to the expert at all
   (Section 8.5).

6. **Interrupt + `update_state(as_node=…)` for human-in-the-loop.** The
   pause point is a *no-op node*, which keeps the graph topology explicit:
   you can see in the wiring exactly where a human sits in the loop.

7. **A reducer as a concurrency fix (`_keep_last`).** The N-parallel-writes
   `InvalidUpdateError` is one of the most common real-world LangGraph
   pitfalls; folding identical scalar writes with a keep-last reducer is the
   canonical fix.

8. **Fresh `thread_id` per run.** With persistent checkpoints, reusing a
   thread id silently resumes an old (possibly finished) run. Random ids make
   resumption *opt-in* (`--thread-id`), never accidental.

---

## 12. Configuration Reference

All settings come from environment variables (a local `.env` is auto-loaded).

| Variable                 | Default                        | Purpose                                              |
|--------------------------|--------------------------------|------------------------------------------------------|
| `LLM_PROVIDER`           | `nvidia`                       | `nvidia`, `ollama`, or `openrouter`                  |
| `NVIDIA_MODEL`           | `meta/llama-3.3-70b-instruct`  | NVIDIA model id                                      |
| `NVIDIA_API_KEY`         | — (required for nvidia)        | NVIDIA NIM credential                                |
| `OLLAMA_MODEL`           | `llama3.1:8b`                  | Ollama model id                                      |
| `OPENROUTER_MODEL`       | `openai/gpt-4o-mini`           | OpenRouter model id (e.g. `openai/gpt-4o`)           |
| `OPENROUTER_API_KEY`     | — (required for openrouter)    | OpenRouter credential                                |
| `OPENROUTER_BASE_URL`    | `https://openrouter.ai/api/v1` | OpenRouter endpoint override                         |
| `CHECKPOINT_BACKEND`     | `sqlite`                       | `sqlite` or `memory`                                 |
| `CHECKPOINT_DB`          | `ares_checkpoints.sqlite`      | SQLite file path                                     |
| `SEARCH_BACKEND`         | `duckduckgo`                   | `duckduckgo`, `tavily`, or `none` (gates *web* search only — Wikipedia always runs) |
| `WEB_MAX_RESULTS`        | `3`                            | Web results fetched per query                        |
| `TAVILY_API_KEY`         | — (required for tavily)        | Tavily credential                                    |
| `MAX_SOURCES_FOR_EXPERT` | `5`                            | Max sources rendered into the expert's prompt        |
| `MAX_SOURCE_CHARS`       | `1500`                         | Per-source character cap in the prompt               |
| `ARES_TOPIC`             | —                              | Default for `--topic`                                |
| `ARES_MAX_ANALYSTS`      | `3`                            | Default for `--max-analysts`                         |
| `ARES_MAX_TURNS`         | `3`                            | Default for `--max-turns`                            |
| `ARES_THREAD_ID`         | —                              | Default for `--thread-id`                            |
| `ARES_OUTPUT`            | `research_report.md`           | Default for `--output`                               |

---

## 13. Glossary

| Term | Meaning |
|------|---------|
| **Node** | A Python function in a LangGraph graph; takes state, returns a partial state update. |
| **Edge / Conditional edge** | Fixed vs. state-dependent transition between nodes. |
| **State** | The typed dict flowing through a graph; the only inter-node data channel. |
| **Reducer** | Merge rule for a state key (e.g., `operator.add` appends lists) — makes parallel writes safe. |
| **Checkpointer** | Backend that persists state after each step, enabling pause/resume by `thread_id`. |
| **Thread id** | Identifier for one graph run's checkpoint history. |
| **Interrupt** | A compiled-in pause point (`interrupt_before`) used for human-in-the-loop. |
| **`Send`** | LangGraph primitive to launch a node with explicit private state; a list of Sends = parallel fan-out. |
| **Map–Reduce** | Run N independent tasks in parallel (map), then combine results (reduce). |
| **Structured output** | Forcing an LLM to return schema-validated JSON instead of free text. |
| **Persona** | A generated analyst identity (name, role, affiliation, focus) injected into prompts. |
| **Re-roling** | Re-labelling stored messages as Human/AI from one participant's perspective before an LLM call. |
| **Exponential backoff** | Retrying with doubling wait times (1 s, 2 s, 4 s…) to avoid overwhelming a failing service. |
| **Human-in-the-loop** | A workflow that pauses for human review/feedback before continuing. |
| **Grounding** | Restricting an LLM's answer to supplied source material to prevent hallucination. |
