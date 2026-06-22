"""
ARES — Autonomous Research & Multi-Agent Evaluation Engine
==========================================================

ARES turns a single research *topic* into a structured, cited markdown *report*
by simulating a small research team with LangGraph. It is organised into four
phases, each clearly marked with a banner further down this file:

    PHASE 1 · Create analysts
        Generate N "analyst personas", each focused on a different angle of the
        topic. A human-in-the-loop checkpoint lets you accept or refine them.

    PHASE 2 · Interview system  (a reusable sub-graph, run once per analyst)
        ask_question → search_context → answer_question  (looped a few turns)
        → save_interview → write_section  → a single cited markdown memo.

    PHASE 3 · Master research graph
        Fan OUT the Phase-2 sub-graph over every analyst in parallel (the "Map"
        step), then fan IN their memos and write the introduction / body /
        conclusion and stitch them into the final report (the "Reduce" step).

    PHASE 4 · Execution / CLI
        An interactive terminal app (rich UI) that drives the master graph:
        prompt for a topic, show the analysts, collect feedback, stream the
        interviews live, then render and save the final report.

State objects (how data flows through the graphs):
    GenerateAnalystsState  →  InterviewState  →  ResearchGraphState

Configuration — environment variables (a local .env is loaded):
    LLM_PROVIDER        "nvidia" (default) | "ollama"
    NVIDIA_MODEL        NVIDIA model id       (default: "meta/llama-3.3-70b-instruct")
    NVIDIA_API_KEY      Required (default provider is nvidia)
    OLLAMA_MODEL        Ollama model id       (default: "llama3.1:8b"; used when LLM_PROVIDER=ollama)
    CHECKPOINT_BACKEND  "sqlite" (default) | "memory"
    CHECKPOINT_DB       SQLite file path      (default: "ares_checkpoints.sqlite")
    SEARCH_BACKEND      "duckduckgo" (default) | "tavily" | "none"
    WEB_MAX_RESULTS     Web results per query (default: 3)
    TAVILY_API_KEY      Required only when SEARCH_BACKEND=tavily
    ARES_TOPIC / ARES_MAX_ANALYSTS / ARES_MAX_TURNS / ARES_THREAD_ID / ARES_OUTPUT
                        Defaults for the matching CLI flags.

Run it (default provider is NVIDIA — set NVIDIA_API_KEY, or use LLM_PROVIDER=ollama):
    python ARES.py                                  # fully interactive
    python ARES.py --topic "..." --no-feedback      # non-interactive
    python ARES.py --max-analysts 4 --max-turns 2   # tune depth
"""

# =============================================================================
#  IMPORTS & EARLY ENVIRONMENT SETUP
# =============================================================================
# IMPORTANT: two environment tweaks MUST run *before* any langchain import,
# because importing langchain_community can transitively import `transformers`,
# which eagerly imports TensorFlow. ARES never uses TF, so we opt out up front.

# --- Standard library --------------------------------------------------------
import os
import sys
import re
import time
import uuid
import operator
import argparse
import contextlib
from typing import Annotated, List, TypedDict

# Tell `transformers` not to import TensorFlow (a broken TF/protobuf install
# would otherwise crash startup). Must precede the langchain imports below.
os.environ.setdefault("USE_TF", "0")

# Force UTF-8 console output so emoji/Unicode don't raise UnicodeEncodeError on
# legacy Windows code pages (cp1252). Harmless if the stream can't reconfigure.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# --- Third-party -------------------------------------------------------------
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, merge_message_runs
from langchain_community.document_loaders import WikipediaLoader
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Send

# Optional: `rich` powers the interactive terminal UI (banner, tables, spinners,
# markdown rendering). If it isn't installed, every ui_* helper below falls back
# to plain print(), so the program still runs — just without the styling.
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.markdown import Markdown
    from rich.prompt import Prompt
    from rich.rule import Rule
    from rich import box
    console = Console()
    _RICH = True
except Exception:
    console = None
    _RICH = False

# Load variables from a local .env file (if present) before reading any config.
load_dotenv()


# =============================================================================
#  CONFIGURATION (LLM provider + web search backend)
# =============================================================================

# --- LLM provider ------------------------------------------------------------
# Choose the chat-model backend. Only the active provider's package is imported
# and only its credential is validated, so a local Ollama run never requires an
# NVIDIA key (and vice-versa).
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "nvidia").lower()

if LLM_PROVIDER == "nvidia":
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    assert os.getenv("NVIDIA_API_KEY"), "NVIDIA_API_KEY not found in .env file!"
    llm = ChatNVIDIA(model=os.getenv("NVIDIA_MODEL", "meta/llama-3.3-70b-instruct"))
elif LLM_PROVIDER == "ollama":
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"))
else:
    raise ValueError(f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Use 'ollama' or 'nvidia'.")

# --- Web search backend ------------------------------------------------------
# Used during interviews to fetch supporting context (alongside Wikipedia).
SEARCH_BACKEND = os.getenv("SEARCH_BACKEND", "duckduckgo").lower()
WEB_MAX_RESULTS = int(os.getenv("WEB_MAX_RESULTS", "3"))

# Sources accumulate every turn, so cap what gets rendered into a prompt to keep
# it within model context limits (state["sources"] still keeps the full list).
MAX_SOURCES_FOR_EXPERT = int(os.getenv("MAX_SOURCES_FOR_EXPERT", "5"))   # max sources shown
MAX_SOURCE_CHARS = int(os.getenv("MAX_SOURCE_CHARS", "1500"))            # per-source char cap


# =============================================================================
#  LLM INVOCATION HELPERS (retries + structured output)
# =============================================================================
# Small local models and flaky networks intermittently fail generation or
# structured parsing. These wrappers add bounded retries with exponential
# backoff; structured calls can additionally return a fallback instead of
# raising, so a single bad parse doesn't abort an entire run.

_TRANSIENT_HINTS = ("connection", "refused", "timeout", "timed out",
                    "could not connect", "max retries", "connecterror")


def is_connection_error(exc) -> bool:
    """True if the exception text looks like a network/backend connectivity issue."""
    return any(h in str(exc).lower() for h in _TRANSIENT_HINTS)


def safe_invoke(messages, *, retries=2, label="llm"):
    """Invoke the chat model, retrying with exponential backoff on any error.

    Re-raises the last exception if every attempt fails.
    """
    last_exc = None
    for attempt in range(retries + 1):
        try:
            return llm.invoke(messages)
        except Exception as e:
            last_exc = e
            if attempt < retries:
                wait = 2 ** attempt
                print(f"[{label}] ⚠️ attempt {attempt + 1} failed ({e}); retrying in {wait}s...")
                time.sleep(wait)
    raise last_exc


def structured_invoke(schema, messages, *, retries=2, label="structured", fallback=None):
    """Invoke the model with structured (schema-validated) output, with retries.

    Returns `fallback` if all attempts fail and a fallback was provided;
    otherwise re-raises the last exception.
    """
    structured_llm = llm.with_structured_output(schema)
    last_exc = None
    for attempt in range(retries + 1):
        try:
            return structured_llm.invoke(messages)
        except Exception as e:
            last_exc = e
            if attempt < retries:
                wait = 2 ** attempt
                print(f"[{label}] ⚠️ structured attempt {attempt + 1} failed ({e}); retrying in {wait}s...")
                time.sleep(wait)
    if fallback is not None:
        print(f"[{label}] ❌ giving up after {retries + 1} attempts; using fallback.")
        return fallback
    raise last_exc


# =============================================================================
#  PHASE 1 · CREATE ANALYSTS  (human-in-the-loop)
# =============================================================================
# Generate analyst personas for the topic, with an interruptible feedback step.

# --- Data models -------------------------------------------------------------

class Analyst(BaseModel):
    """A single analyst persona that will interview an expert from one angle."""
    affiliation: str = Field(description="Primary affiliation of the analyst.")
    name: str = Field(description="Name of the analyst.")
    role: str = Field(description="Role of the analyst in the context of the topic.")
    description: str = Field(description="Description of the analyst focus, concerns, and motives.")

    @property
    def persona(self) -> str:
        """Human-readable persona block injected into the interview prompts."""
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"


class Perspectives(BaseModel):
    """Structured-output wrapper: the list of analysts the LLM must return."""
    analysts: List[Analyst] = Field(
        description="Comprehensive list of analysts with their roles and affiliations.",
    )


class GenerateAnalystsState(TypedDict):
    """State for the standalone analyst-creation graph."""
    topic: str                    # Research topic
    max_analysts: int             # Number of analysts to generate
    human_analyst_feedback: str   # Optional editorial feedback from the user
    analysts: List[Analyst]       # The generated analyst personas


# --- Checkpointer ------------------------------------------------------------

def make_checkpointer():
    """Create the LangGraph checkpointer used to persist/resume graph runs.

    CHECKPOINT_BACKEND=sqlite (default) persists to disk so runs can resume
    across restarts; =memory keeps everything in RAM. Falls back to in-memory
    if the optional SQLite extra (`langgraph-checkpoint-sqlite`) isn't installed.
    """
    backend = os.getenv("CHECKPOINT_BACKEND", "sqlite").lower()
    if backend == "memory":
        return MemorySaver()
    if backend == "sqlite":
        try:
            import sqlite3
            from langgraph.checkpoint.sqlite import SqliteSaver
            db_path = os.getenv("CHECKPOINT_DB", "ares_checkpoints.sqlite")
            conn = sqlite3.connect(db_path, check_same_thread=False)
            print(f"[Checkpoint] Using SQLite checkpointer at '{db_path}'.")
            return SqliteSaver(conn)
        except Exception as e:
            print(f"[Checkpoint] ⚠️ SQLite unavailable ({e}); falling back to in-memory.")
            return MemorySaver()
    print(f"[Checkpoint] ⚠️ Unknown CHECKPOINT_BACKEND '{backend}'; using in-memory.")
    return MemorySaver()


# Shared checkpointer for the master graph (and the standalone analyst graph).
memory = make_checkpointer()


# --- Prompt + nodes ----------------------------------------------------------

analyst_instructions = """You are tasked with creating a set of AI analyst personas. Follow these instructions carefully:
                        1. First, review the research topic: {topic}
                        2. Examine any editorial feedback that has been optionally provided to guide creation of the analysts: {human_analyst_feedback}
                        3. Determine the most interesting themes based upon documents and / or feedback above.
                        4. Pick the top {max_analysts} themes.
                        5. Assign one analyst to each theme."""


def create_analysts(state: GenerateAnalystsState):
    """Node: ask the LLM to generate analyst personas for the topic.

    Honours any human feedback already in state. Uses structured output so we
    get back a validated `Perspectives` object. Analysts are critical to the
    whole pipeline, so a total failure is allowed to raise (no silent fallback).
    """
    topic = state["topic"]
    max_analysts = state["max_analysts"]
    human_analyst_feedback = state.get("human_analyst_feedback", "")

    system_message = analyst_instructions.format(
        topic=topic,
        human_analyst_feedback=human_analyst_feedback,
        max_analysts=max_analysts)

    # Enforce structured output (with retries). Analysts are critical to the whole
    # pipeline, so we let a total failure raise rather than continue with none.
    analysts = structured_invoke(
        Perspectives,
        [SystemMessage(content=system_message),
         HumanMessage(content="Generate the set of analysts.")],
        label="create-analysts",
    )

    return {"analysts": analysts.analysts}


def human_feedback(state: GenerateAnalystsState):
    """Node: deliberate no-op. The graph is compiled to interrupt *before* this
    node, which is how we pause to collect feedback from the user."""
    pass


def should_continue(state: GenerateAnalystsState):
    """Conditional edge (standalone graph only): regenerate if feedback was
    given, otherwise end."""
    if state.get("human_analyst_feedback", None):
        return "create_analysts"
    return END


# --- Standalone analyst graph (OPTIONAL) -------------------------------------
# A self-contained graph for *just* the analyst-creation step. Handy for
# debugging or LangGraph Studio. The full pipeline below uses `master_graph`
# and does NOT depend on this object.
analyst_builder = StateGraph(GenerateAnalystsState)
analyst_builder.add_node("create_analysts", create_analysts)
analyst_builder.add_node("human_feedback", human_feedback)
analyst_builder.add_edge(START, "create_analysts")
analyst_builder.add_edge("create_analysts", "human_feedback")
analyst_builder.add_conditional_edges("human_feedback", should_continue, ["create_analysts", END])
analyst_only_graph = analyst_builder.compile(interrupt_before=["human_feedback"], checkpointer=memory)


# =============================================================================
#  PHASE 2 · INTERVIEW SYSTEM  (reusable sub-graph, one run per analyst)
# =============================================================================
# A single analyst interviews an "expert" (the same LLM in a different role).
# Flow: ask_question → search_context → answer_question  (looped) →
#       save_interview → write_section.

# --- Interview state ---------------------------------------------------------

class InterviewState(MessagesState):
    """State for one interview. Inherits `messages` from MessagesState.

    The whole conversation lives in `messages`: the analyst's questions and the
    expert's answers are BOTH AIMessages, distinguished by `.name == "expert"`.
    """
    max_num_turns: int
    # Retrieved sources accumulate across turns; each item is {"source", "content"}.
    sources: Annotated[list, operator.add]
    analyst: Analyst    # Which analyst is conducting this interview
    interview: str      # Final labelled transcript (filled by save_interview)
    sections: list      # The written memo (filled by write_section)


class SearchQuery(BaseModel):
    """Structured-output wrapper for the rewritten search query."""
    search_query: str = Field(description="Search query to run on Wikipedia")


# --- Question generation -----------------------------------------------------

question_instructions = """You are an analyst conducting an interview, staying in character as:
{goals}

Ask ONE insightful, specific question about your topic, building on the expert's
previous answer when there is one.

Output rules — follow exactly:
- Output ONLY the question itself. No preamble, greetings, or commentary.
- Do NOT prefix your name or any speaker label (e.g. do not write "Aaradhya Jain:").
- Do NOT include stage directions or actions in parentheses (e.g. "(nodding)").
- Do NOT restate these instructions.

When you have gathered enough information and the interview is complete, reply with
exactly this sentence and nothing else:
Thank you so much for your help!
"""


def generate_question(state: InterviewState):
    """Node `ask_question`: the analyst asks the next question, in character.

    The transcript is re-roled from the analyst's perspective (expert answers
    become Human turns) so the model always responds to a Human turn.
    """
    analyst = state["analyst"]

    system_msg = SystemMessage(content=question_instructions.format(goals=analyst.persona))
    dialogue = build_dialogue(state["messages"], perspective="analyst")
    response = safe_invoke([system_msg] + dialogue, label="ask_question")

    return {"messages": [response]}


# --- Context retrieval (Wikipedia + optional web search) ---------------------

def search_web(query):
    """Run a web search and return a list of {"source", "content"} dicts.

    Backend is chosen by SEARCH_BACKEND. Degrades gracefully (returns []) if the
    backend is disabled, a dependency is missing, or the search errors out.
    """
    if SEARCH_BACKEND in ("none", "off", ""):
        return []
    try:
        if SEARCH_BACKEND == "tavily":
            if not os.getenv("TAVILY_API_KEY"):
                print("[Web] ⚠️ TAVILY_API_KEY not set; skipping web search.")
                return []
            from langchain_community.tools.tavily_search import TavilySearchResults
            results = TavilySearchResults(max_results=WEB_MAX_RESULTS).invoke({"query": query})
            items = [(r.get("url", "web"), r.get("content", "")) for r in results]
        else:  # duckduckgo (default, keyless)
            from langchain_community.tools import DuckDuckGoSearchResults
            raw = DuckDuckGoSearchResults(output_format="list", num_results=WEB_MAX_RESULTS).invoke(query)
            if isinstance(raw, str):
                items = [("web", raw)]
            else:
                items = [(r.get("link", r.get("url", "web")),
                          r.get("snippet", r.get("content", ""))) for r in raw]
        items = [(src, txt) for src, txt in items if txt]
        print(f"[Web] ✅ {SEARCH_BACKEND} returned {len(items)} results for: '{query}'")
        return [{"source": src, "content": txt} for src, txt in items]
    except Exception as e:
        print(f"[Web] ❌ {SEARCH_BACKEND} search failed for '{query}': {e}")
        return []


def _dedup_sources(sources):
    """Drop duplicate sources (by URL), preserving first-seen order."""
    seen, ordered = set(), []
    for s in sources:
        key = s.get("source") or s.get("content", "")[:60]
        if key not in seen:
            seen.add(key)
            ordered.append(s)
    return ordered


def _format_numbered_sources(sources):
    """Build stable [n] numbering for the sources gathered so far.

    Returns a tuple of two strings:
      - numbered_context: the sources as "[1] Source: ...\\n<content>" blocks,
        fed to the expert so its [1],[2] citations map to real sources.
      - sources_list:     the "[1] <url>" listing for a section's Sources block.
    """
    sources = _dedup_sources(sources)
    if not sources:
        return "No source material was retrieved.", ""
    sources = sources[:MAX_SOURCES_FOR_EXPERT]   # cap count, keep order -> stable [n]
    ctx, listing = [], []
    for i, s in enumerate(sources, 1):
        src = s.get("source", "unknown")
        content = (s.get("content", "") or "").strip()
        if len(content) > MAX_SOURCE_CHARS:      # truncate long pages to avoid overflow
            content = content[:MAX_SOURCE_CHARS].rstrip() + " …[truncated]"
        ctx.append(f"[{i}] Source: {src}\n{content}")
        listing.append(f"[{i}] {src}")
    return "\n\n---\n\n".join(ctx), "\n".join(listing)


def _latest_analyst_question(messages):
    """Return the text of the most recent analyst question (falling back to the
    seed message). Used to build a reliable search query without relying on the
    model to pick 'the last question' out of an all-AI history."""
    for m in reversed(messages):
        if isinstance(m, AIMessage) and getattr(m, "name", None) != "expert":
            return (m.content or "").strip()
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return (m.content or "").strip()
    return ""


def build_dialogue(messages, perspective):
    """Re-role the shared transcript for a single llm.invoke call.

    state["messages"] stores BOTH sides as AIMessages (expert answers tagged
    name="expert", analyst questions untagged) plus the single seed HumanMessage.
    A chat model needs a clear Human/AI alternation that ENDS on a Human turn,
    or small models return empty output. This returns a NEW re-roled list and
    never mutates `messages`.

      perspective="expert":  analyst questions -> Human, own answers -> AI,
                             seed dropped (list ends on the latest question).
      perspective="analyst": expert answers -> Human, own questions -> AI,
                             seed kept as the opening Human turn.
    """
    out = []
    for m in messages:
        if isinstance(m, HumanMessage):
            # The seed "Start the interview..." message.
            if perspective == "analyst":
                out.append(HumanMessage(content=m.content or ""))
            # expert perspective: drop the seed so the list ends on a question.
            continue

        is_expert = isinstance(m, AIMessage) and getattr(m, "name", None) == "expert"
        content = m.content or ""
        if perspective == "expert":
            out.append(AIMessage(content=content) if is_expert else HumanMessage(content=content))
        else:  # analyst
            out.append(HumanMessage(content=content) if is_expert else AIMessage(content=content))

    # Collapse any accidental consecutive same-role runs -> strict alternation.
    return merge_message_runs(out)


def search_context(state: InterviewState):
    """Node `search_context`: rewrite the latest analyst question into a search
    query, then fetch supporting sources from Wikipedia and (optionally) the web."""
    # Locate the latest analyst question deterministically, then rewrite it into
    # a clean query. This avoids the all-AI-history ambiguity of "the last question".
    latest_q = _latest_analyst_question(state["messages"])
    sq = structured_invoke(
        SearchQuery,
        [SystemMessage(content="Rewrite the following interview question as a short, "
                               "clean web search query. Return ONLY the query string."),
         HumanMessage(content=latest_q)],
        label="search-query",
        fallback=SearchQuery(search_query=""),
    )

    # Use the rewrite, falling back to the raw question text.
    query = (sq.search_query or "").strip() or latest_q[:100]

    if not query:
        print("[Search] No query found, skipping.")
        return {"sources": []}

    sources = []

    # 1) Wikipedia
    try:
        docs = WikipediaLoader(query=query, load_max_docs=2).load()
        print(f"[Wikipedia] ✅ Found {len(docs)} docs for: '{query}'")
        for d in docs:
            sources.append({
                "source": d.metadata.get("source", "wikipedia"),
                "content": d.page_content,
            })
    except Exception as e:
        print(f"[Wikipedia] ❌ Failed for '{query}': {e}")

    # 2) Optional web search
    sources.extend(search_web(query))

    return {"sources": sources}


# --- Answer generation -------------------------------------------------------

answer_instructions = """You are an expert being interviewed by an analyst.

Here is the analyst's profile:
{goals}

Use ONLY the following context to answer their question:
{context}

Rules:
1. Answer the analyst's most recent question directly and immediately.
2. Use ONLY the context above — no external facts.
3. Cite sources inline like [1], [2] matching the numbered context.
4. Tailor the answer to the analyst's focus; keep it focused and specific.
5. Output ONLY the answer. No speaker labels, no preamble, no stage directions.
"""


def generate_answer(state: InterviewState):
    """Node `answer_question`: the expert answers the latest question using only
    the retrieved context, citing numbered sources.

    The transcript is re-roled from the expert's perspective (analyst questions
    become Human turns, seed dropped) so the list ends on the question to answer.
    """
    analyst = state["analyst"]

    # Build a numbered context so the expert's [1],[2] citations map to real sources.
    numbered_context, _ = _format_numbered_sources(state.get("sources", []))

    sys_msg = SystemMessage(content=answer_instructions.format(
        goals=analyst.persona,
        context=numbered_context,
    ))

    dialogue = build_dialogue(state["messages"], perspective="expert")
    answer = safe_invoke([sys_msg] + dialogue, label="answer")

    # Safety net: if the model still returns nothing, nudge once, then fall back
    # so the transcript/turn-count never carries an empty answer.
    if not (answer.content or "").strip():
        nudge = HumanMessage(content="Please answer the question above now, using only the "
                                     "provided context and citing sources like [1].")
        answer = safe_invoke([sys_msg] + dialogue + [nudge], label="answer-retry")
    if not (answer.content or "").strip():
        answer = AIMessage(content="(The expert could not provide an answer from the available sources.)")

    answer.name = "expert"   # Tag so the router can count expert answers.

    return {"messages": [answer]}


# --- Routing (interview control flow) ----------------------------------------

def _num_expert_answers(messages):
    """Count how many answers the expert has given so far."""
    return len([m for m in messages
                if isinstance(m, AIMessage) and getattr(m, "name", None) == "expert"])


def continue_or_finish(state: InterviewState):
    """Conditional edge after `ask_question`.

    Finish early if the analyst signed off (avoids generating a wasted answer to
    a goodbye); otherwise go retrieve context for the question. Always allows at
    least one Q&A round before honouring an early sign-off.
    """
    messages = state["messages"]
    last = (messages[-1].content or "") if messages else ""
    if _num_expert_answers(messages) >= 1 and "Thank you so much for your help" in last:
        return "save_interview"
    return "search_context"


def route_messages(state: InterviewState):
    """Conditional edge after `answer_question`: stop at the turn cap, else ask
    another question."""
    if _num_expert_answers(state["messages"]) >= state["max_num_turns"]:
        return "save_interview"
    return "ask_question"


# --- Saving the interview + writing the section ------------------------------

def save_interview(state: InterviewState):
    """Node `save_interview`: build a clearly-labelled (Analyst vs Expert)
    transcript for the section writer to synthesise from."""
    lines = []
    for m in state["messages"]:
        if isinstance(m, HumanMessage):
            continue  # skip the seed "Start the interview..." message
        role = "Expert" if (isinstance(m, AIMessage) and getattr(m, "name", None) == "expert") else "Analyst"
        lines.append(f"{role}: {m.content}")
    return {"interview": "\n\n".join(lines)}


section_writer_instructions = """
You are a technical writer. Using the INTERVIEW TRANSCRIPT and the NUMBERED SOURCES
provided by the user, write a focused markdown section.

Use exactly this structure:

## <a short, descriptive title>
### Summary
<~300 words synthesizing the expert's answers through the analyst's specific angle>
### Sources
<list the numbered sources you used>

Rules:
- Cite claims inline as [1], [2], ... matching the numbered sources.
- Base the summary on the transcript; do not invent facts or sources.
"""


def write_section(state: InterviewState):
    """Node `write_section`: turn one interview into a cited markdown memo,
    synthesising from the transcript (not just the raw sources)."""
    analyst = state["analyst"]
    interview = state.get("interview", "")
    numbered_context, sources_list = _format_numbered_sources(state.get("sources", []))

    sys_msg = SystemMessage(content=section_writer_instructions)
    human_msg = HumanMessage(content=(
        f"ANALYST PERSPECTIVE:\n{analyst.persona}\n\n"
        f"INTERVIEW TRANSCRIPT:\n{interview}\n\n"
        f"NUMBERED SOURCES:\n{numbered_context}"
    ))

    result = safe_invoke([sys_msg, human_msg], label="write_section")
    section = result.content

    # Guarantee a real Sources list even if the model omitted one.
    if sources_list and "### Sources" not in section:
        section = section.rstrip() + "\n\n### Sources\n" + sources_list

    return {"sections": [section]}


# --- Build the interview sub-graph -------------------------------------------

interview_builder = StateGraph(InterviewState)

interview_builder.add_node("ask_question", generate_question)
interview_builder.add_node("search_context", search_context)
interview_builder.add_node("answer_question", generate_answer)
interview_builder.add_node("save_interview", save_interview)
interview_builder.add_node("write_section", write_section)

interview_builder.add_edge(START, "ask_question")
# After a question: retrieve context for it, or finish if the analyst signed off.
interview_builder.add_conditional_edges("ask_question", continue_or_finish, ["search_context", "save_interview"])
interview_builder.add_edge("search_context", "answer_question")
# After an answer: loop for another question, or finish at the turn cap.
interview_builder.add_conditional_edges("answer_question", route_messages, ["ask_question", "save_interview"])
interview_builder.add_edge("save_interview", "write_section")
interview_builder.add_edge("write_section", END)

# No checkpointer here: when this graph is used as a node inside master_graph,
# the parent graph's checkpointer handles persistence for the whole tree.
interview_graph = interview_builder.compile()


# =============================================================================
#  PHASE 3 · MASTER RESEARCH GRAPH  (Map → Reduce)
# =============================================================================
# Create analysts → (feedback loop) → run all interviews in parallel →
# write intro/body/conclusion → stitch the final report.

# --- Master state ------------------------------------------------------------

class ResearchGraphState(TypedDict):
    """Top-level state for the full research pipeline."""
    topic: str
    max_analysts: int
    max_num_turns: int
    human_analyst_feedback: str
    analysts: List[Analyst]
    sections: Annotated[list, operator.add]   # memos collected from all interviews
    introduction: str
    content: str
    conclusion: str
    final_report: str


# --- Map step: fan out one interview per analyst -----------------------------

def initiate_all_interviews(state: ResearchGraphState):
    """Conditional edge out of `human_feedback`, with two jobs:

    1. If the user left feedback, route back to `create_analysts` to regenerate.
    2. Otherwise, fan OUT: return one Send() per analyst to launch the interview
       sub-graph for each of them in parallel (the "Map" step).
    """
    # 1) Feedback present → regenerate analysts.
    human_analyst_feedback = state.get("human_analyst_feedback")
    if human_analyst_feedback:
        return "create_analysts"

    # 2) Otherwise kick off an interview sub-graph for every analyst.
    topic = state["topic"]
    max_num_turns = state.get("max_num_turns", 3)
    return [
        Send("conduct_interview", {
            "analyst": analyst,
            # Seed each interview with a HumanMessage (cast explicitly for Ollama stability).
            "messages": [HumanMessage(content=f"Start the interview regarding: {topic}")],
            "max_num_turns": max_num_turns,
            "sources": [],
            "sections": []
        }) for analyst in state["analysts"]
    ]


# --- Reduce step: write the report from the collected memos -------------------

report_writer_instructions = """You are a technical writer.
Topic: {topic}
Based on the analyst memos provided, write a comprehensive section.
Do not use "Analyst 1 said..." just synthesize the facts.
Memos: {context}"""


def write_report(state: ResearchGraphState):
    """Node: synthesise the main body of the report from all analyst memos."""
    sections = state["sections"]
    topic = state["topic"]
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

    system_message = report_writer_instructions.format(topic=topic, context=formatted_str_sections)
    report = safe_invoke([SystemMessage(content=system_message),
                          HumanMessage(content="Write the main body of the report.")], label="write_report")
    return {"content": report.content}


def write_introduction(state: ResearchGraphState):
    """Node: write a short introduction for the report."""
    topic = state["topic"]
    instructions = f"Write a catchy 100-word introduction for a report on: {topic}"
    intro = safe_invoke([SystemMessage(content=instructions),
                         HumanMessage(content="Write the introduction.")], label="write_introduction")
    return {"introduction": intro.content}


def write_conclusion(state: ResearchGraphState):
    """Node: write a short conclusion for the report."""
    topic = state["topic"]
    instructions = f"Write a strong 100-word conclusion for a report on: {topic}"
    conclusion = safe_invoke([SystemMessage(content=instructions),
                              HumanMessage(content="Write the conclusion.")], label="write_conclusion")
    return {"conclusion": conclusion.content}


def finalize_report(state: ResearchGraphState):
    """Node: stitch introduction + body + conclusion into the final markdown."""
    final_report = (
        "# " + state["topic"] + "\n\n" +
        "## Introduction\n" + state["introduction"] + "\n\n" +
        "## Insights\n" + state["content"] + "\n\n" +
        "## Conclusion\n" + state["conclusion"]
    )
    return {"final_report": final_report}


# --- Build the master graph --------------------------------------------------

master_builder = StateGraph(ResearchGraphState)

# Nodes
master_builder.add_node("create_analysts", create_analysts)
master_builder.add_node("human_feedback", human_feedback)
master_builder.add_node("conduct_interview", interview_graph)   # the Phase-2 sub-graph
master_builder.add_node("write_report", write_report)
master_builder.add_node("write_introduction", write_introduction)
master_builder.add_node("write_conclusion", write_conclusion)
master_builder.add_node("finalize_report", finalize_report)

# Edges
master_builder.add_edge(START, "create_analysts")
master_builder.add_edge("create_analysts", "human_feedback")
# Conditional: loop back for feedback OR fan out to the interviews.
master_builder.add_conditional_edges("human_feedback", initiate_all_interviews,
                                     ["create_analysts", "conduct_interview"])
# Fan-in: once all interviews finish, write the three report parts in parallel.
master_builder.add_edge("conduct_interview", "write_report")
master_builder.add_edge("conduct_interview", "write_introduction")
master_builder.add_edge("conduct_interview", "write_conclusion")
# Join the three parts, then finalize.
master_builder.add_edge(["write_conclusion", "write_report", "write_introduction"], "finalize_report")
master_builder.add_edge("finalize_report", END)

# Compile. interrupt_before=human_feedback is what pauses the run for feedback.
master_graph = master_builder.compile(interrupt_before=["human_feedback"], checkpointer=memory)


# =============================================================================
#  PHASE 4 · EXECUTION (interactive CLI)
# =============================================================================

# --- Analyst display ---------------------------------------------------------

def display_analysts(analysts):
    """Pretty-print the generated analysts (rich table, or plain text fallback)."""
    if _RICH:
        table = Table(title=f"🧑‍🔬 {len(analysts)} Analyst(s) Generated",
                      box=box.ROUNDED, show_lines=True,
                      title_style="bold cyan", header_style="bold magenta")
        table.add_column("#", justify="right", style="cyan", no_wrap=True)
        table.add_column("Name", style="bold")
        table.add_column("Role")
        table.add_column("Affiliation", style="dim")
        table.add_column("Focus")
        for i, a in enumerate(analysts, 1):
            table.add_row(str(i), a.name, a.role, a.affiliation, a.description)
        console.print(table)
    else:
        print(f"\n--> Generated {len(analysts)} analysts:\n")
        for analyst in analysts:
            print(f"Name: {analyst.name}")
            print(f"Affiliation: {analyst.affiliation}")
            print(f"Role: {analyst.role}")
            print(f"Description: {analyst.description}")
            print("-" * 50)


# --- Generic UI helpers (rich, with plain-print fallbacks) -------------------
# Every helper checks `_RICH`. When rich is missing, [tag] markup is stripped so
# plain output stays clean.

def ui_banner():
    """Print the ARES title banner."""
    if _RICH:
        console.print(Panel(
            "[bold cyan]🔬 ARES[/]\n"
            "[dim]Autonomous Research & Multi-Agent Evaluation Engine[/]",
            border_style="cyan", box=box.DOUBLE, expand=False, padding=(1, 4)))
    else:
        print("\n🚀 LAUNCHING AUTONOMOUS RESEARCH AGENT...\n")


@contextlib.contextmanager
def ui_status(message):
    """Context manager: show a spinner while a blocking step runs (rich), or
    just print a one-line notice (plain)."""
    if _RICH:
        with console.status(message, spinner="dots"):
            yield
    else:
        print(re.sub(r"\[/?[^\]]*\]", "", message))
        yield


def ui_print(msg, style=None):
    """Print a (possibly styled) line."""
    if _RICH:
        console.print(msg, style=style)
    else:
        print(re.sub(r"\[/?[^\]]*\]", "", msg))


def ui_rule(title, style="cyan"):
    """Print a horizontal section divider with a title."""
    if _RICH:
        console.print(Rule(title, style=style))
    else:
        plain = re.sub(r"\[/?[^\]]*\]", "", title)
        print("\n" + "=" * 70 + f"\n{plain}\n" + "=" * 70 + "\n")


def ui_ask(prompt_text, default=None):
    """Prompt the user for input (rich Prompt, or input() fallback)."""
    if _RICH:
        return Prompt.ask(prompt_text) if default is None else Prompt.ask(prompt_text, default=default)
    suffix = f" [{default}]" if default else ""
    val = input(f"{prompt_text}{suffix}: ")
    return val if val.strip() != "" else (default if default is not None else "")


def ui_qa(title, content, color):
    """Print a question/answer panel."""
    if _RICH:
        console.print(Panel(str(content), title=title, title_align="left",
                            border_style=color, box=box.ROUNDED))
    else:
        print(f"\n{title}: {content}\n")


def ui_node_done(node):
    """Print a 'finished node X' acknowledgement."""
    if _RICH:
        console.print(f"[green]✓[/] finished [bold]{node}[/]")
    else:
        print(f"✅ Finished Node: {node}")


# --- CLI argument parsing ----------------------------------------------------

def parse_args():
    """Parse CLI flags. Each flag falls back to an ARES_* environment variable."""
    p = argparse.ArgumentParser(
        description="ARES - Autonomous Research & Multi-Agent Evaluation Engine"
    )
    p.add_argument("--topic", default=os.getenv("ARES_TOPIC"),
                   help="Research topic (prompted interactively if omitted).")
    p.add_argument("--max-analysts", type=int, default=int(os.getenv("ARES_MAX_ANALYSTS", "3")),
                   help="Number of analyst personas to generate (default: 3).")
    p.add_argument("--max-turns", type=int, default=int(os.getenv("ARES_MAX_TURNS", "3")),
                   help="Q&A turns per interview (default: 3).")
    p.add_argument("--thread-id", default=os.getenv("ARES_THREAD_ID"),
                   help="Checkpoint thread id. Default: random per run. "
                        "Reuse a previous id to resume that run.")
    p.add_argument("--output", default=os.getenv("ARES_OUTPUT", "research_report.md"),
                   help="Path to write the final report (default: research_report.md).")
    p.add_argument("--no-feedback", action="store_true",
                   help="Accept the first set of analysts without prompting for feedback.")
    return p.parse_args()


# --- Main driver -------------------------------------------------------------

def main():
    """Drive the master graph end-to-end with the interactive CLI."""
    args = parse_args()

    ui_banner()

    # Resolve the topic (CLI flag → env → interactive prompt).
    topic = args.topic
    if not topic:
        topic = ui_ask("🔍 Research topic")
    topic = (topic or "").strip()
    if not topic:
        ui_print("[yellow]⚠️ No topic provided. Exiting.[/]")
        return

    max_analysts = args.max_analysts
    max_num_turns = args.max_turns
    # A fresh thread id per run avoids accidentally resuming a finished run from
    # the persistent checkpointer. Pass --thread-id to deliberately resume one.
    thread_id = args.thread_id or uuid.uuid4().hex[:8]
    thread = {"configurable": {"thread_id": thread_id}}
    ui_print(f"[dim]Run · thread_id={thread_id} · analysts={max_analysts} · turns/interview={max_num_turns}[/]")

    # -- Step 1: generate analysts, looping on human feedback until accepted. --
    first_run = True
    while True:
        # First pass kicks off the graph with the topic; later passes resume
        # (input=None) after we record feedback at the human_feedback node. This
        # single loop supports any number of feedback rounds.
        stream_input = (
            {"topic": topic, "max_analysts": max_analysts, "max_num_turns": max_num_turns}
            if first_run else None
        )
        first_run = False

        latest_analysts = None
        with ui_status("[bold cyan]🧠 Generating analysts..."):
            for event in master_graph.stream(stream_input, thread, stream_mode="values"):
                found = event.get("analysts")
                if found:
                    latest_analysts = found
        if latest_analysts:
            display_analysts(latest_analysts)

        # Non-interactive mode: accept the first batch and move on.
        if args.no_feedback:
            master_graph.update_state(
                thread, {"human_analyst_feedback": None}, as_node="human_feedback"
            )
            break

        # Ask for feedback (press Enter to accept the current analysts).
        feedback = ui_ask("✏️  Feedback for analysts (Enter to accept)", default="")
        user_feedback = (feedback or "").strip() or None

        # Record the decision at the interrupted human_feedback node so the graph
        # either loops back to create_analysts (feedback) or fans out to interviews.
        master_graph.update_state(
            thread, {"human_analyst_feedback": user_feedback}, as_node="human_feedback"
        )

        if not user_feedback:
            break  # User accepted the analysts — proceed to interviews.

        ui_rule("♻️  Regenerating analysts...", style="yellow")

    # -- Step 2: run the interviews and synthesise the report. ----------------
    ui_rule("🎙️  Interviews & Report Synthesis", style="cyan")

    final_output = ""

    # subgraphs=True surfaces node updates from *inside* the conduct_interview
    # sub-graph (ask_question / answer_question / search_context). Without it the
    # interview progress below would be invisible.
    for namespace, event in master_graph.stream(
        None, thread, stream_mode="updates", subgraphs=True
    ):
        for node, value in event.items():
            if node == "__interrupt__":
                continue

            # Non-dict updates (e.g. interrupt payloads) — just acknowledge.
            if not isinstance(value, dict):
                ui_node_done(node)
                continue

            if node == "ask_question":
                msgs = value.get("messages", [])
                if msgs:
                    ui_qa("🎤 Analyst Question", msgs[-1].content, "cyan")

            elif node == "answer_question":
                msgs = value.get("messages", [])
                if msgs:
                    ui_qa("💬 Expert Answer", msgs[-1].content, "green")

            elif node == "search_context":
                ui_print("[dim]🔍 Searching sources (Wikipedia + web)...[/]")

            elif node == "finalize_report":
                final_output = value.get("final_report", "")
                ui_print("[bold green]✅ Report finalized![/]")

            else:
                ui_node_done(node)

    # -- Final output: render and save the report. ----------------------------
    ui_rule("📝 Final Report", style="magenta")

    if final_output:
        if _RICH:
            console.print(Panel(Markdown(final_output), border_style="magenta", box=box.ROUNDED))
        else:
            print(final_output)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(final_output)
        ui_print(f"[bold green]✅ Full report saved to '{args.output}'[/]")
    else:
        ui_print("[yellow]⚠️ Report was empty. Check if interviews completed successfully.[/]")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        ui_print("\n[bold]⏹️  Interrupted by user.[/]")
    except Exception as e:
        # Give a friendly hint for the common "backend not reachable" case.
        if is_connection_error(e):
            ui_print(f"[bold red]❌ Could not reach the LLM backend (provider: {LLM_PROVIDER}).[/]")
            if LLM_PROVIDER == "ollama":
                ui_print("   Is Ollama running?  Try:  [bold]ollama serve[/]  then  [bold]ollama pull <model>[/]")
            else:
                ui_print("   Check NVIDIA_API_KEY and your network connection.")
            ui_print(f"[dim]   Details: {e}[/]")
        else:
            raise
