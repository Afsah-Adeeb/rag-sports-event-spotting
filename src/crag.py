"""
Corrective RAG: the branching flow, wired with LangGraph.

WHAT THIS CHANGES
-------------------
Plain RAG (generate.answer_traced) is a straight line:

    question -> retrieve -> generate -> answer

Nothing inspects what retrieval returned. If four of five chunks are junk,
the model sees all five and the only defence is a line in the prompt asking
it to notice.

CRAG turns the line into a flowchart:

    question -> retrieve -> grade -+-> generate            (enough is relevant)
                              ^    |
                              |    +-> deepen -> grade     (too thin: look further)
                              |    |
                              |    +-> refuse              (nothing relevant)
                              +----+

The judgement lives in grader.py -- how a chunk is graded and what the grades
mean. This file only wires the pieces together and carries state between them.
That split is deliberate: the interesting part should be code you wrote, not
behaviour hidden inside a framework.

WHY LANGGRAPH RATHER THAN AN IF-STATEMENT
-------------------------------------------
Honestly, the control flow here is small enough to write by hand. What
LangGraph buys is the *cycle*: `deepen` routes back into `grade`, so the graph
revisits a node it already ran. Ordinary chains (and LangChain's LCEL pipes)
are directed and acyclic -- they cannot express "go back and try again", which
is the defining shape of every corrective and agentic pattern. Building on a
graph now means Stage 2 (query rewriting, self-critique) is another node and
another edge rather than a rewrite.

It also makes the flow inspectable: `python crag.py --graph` prints the
diagram above as Mermaid, generated from the actual wiring, so the picture in
this docstring cannot quietly drift away from the code.

WHAT "DEEPEN" IS, AND IS NOT
------------------------------
Deepening re-runs the SAME query against the SAME index with a larger top_k
and re-grades. It is not query rewriting -- that needs an extra model call to
invent a new question and belongs to Stage 2. Retrieval here is a local FAISS
search costing nothing, so this is the cheapest correction available and the
right one to try first. It also surfaces more distinct papers, which is the
only lever Stage 1 has against the measured multi-paper coverage problem.

COST
------
    refused          1 call   (grade only -- no generation)
    normal           2 calls  (grade, generate)
    deepened         3 calls  (grade, re-grade, generate)

Plain RAG is always 1. The extra call is what buys the correction, and
crag_answer() reports `api_calls` so the trade is measured rather than assumed.
"""

from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

import config
import grader
from generate import generate_with_usage
from retrieve import retrieve, search


class CragState(TypedDict, total=False):
    """Everything the flow carries between nodes.

    LangGraph merges whatever dict a node returns into this state, so running
    totals (timings, token counts, call counts) are read out and added back
    rather than overwritten -- a node that returns {"api_calls": 1} would reset
    the count instead of incrementing it.
    """
    question: str
    top_k: int
    index: Any            # optional caller-supplied FAISS index
    metadata: Any         # its matching chunk metadata
    model: Optional[str]

    chunks: List[dict]         # everything retrieved, for scoring
    sent_chunks: List[dict]    # what generation actually saw
    labels: List[str]
    reasons: List[str]
    graded_ids: List[str]
    attempts: int
    decision: str

    answer: str
    retrieval_ms: float
    grading_ms: float
    generation_ms: float
    grading_input_tokens: int
    grading_output_tokens: int
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    api_calls: int
    grader_failed: bool


def _fetch(state: CragState, top_k: int) -> List[dict]:
    """Retrieve, honouring a caller-supplied index (the app's session uploads)."""
    if state.get("index") is not None:
        return search(state["question"], state["index"], state["metadata"], top_k=top_k)
    return retrieve(state["question"], top_k=top_k)


# --- Nodes -------------------------------------------------------------------

def retrieve_node(state: CragState) -> Dict[str, Any]:
    """First pass: the same retrieval plain RAG does, at the same top_k."""
    import time
    started = time.perf_counter()
    chunks = _fetch(state, state.get("top_k") or config.DEFAULT_TOP_K)
    return {
        "chunks": chunks,
        "attempts": 1,
        "retrieval_ms": (time.perf_counter() - started) * 1000,
    }


def grade_node(state: CragState) -> Dict[str, Any]:
    """Grade the chunks that do not have a grade yet. The corrective signal.

    Only the NEW chunks are sent. After deepening, the first five chunks are
    byte-identical to the ones already graded -- FAISS returns the top-15 with
    the top-5 as its exact prefix -- so re-grading them costs tokens for an
    answer already known. Measured on one question: re-grading all 15 spent
    6,070 input tokens where grading the 10 new ones spends roughly a third of
    that, and the extra call was over half of CRAG's entire cost.

    The prefix assumption is checked rather than trusted: if the already-graded
    chunks are not still sitting at the front, everything is re-graded.
    """
    chunks = state["chunks"]
    old_labels = list(state.get("labels") or [])
    old_reasons = list(state.get("reasons") or [])
    graded_ids = state.get("graded_ids") or []

    # Only reuse grades if those exact chunks are still the leading ones.
    prefix_intact = (
        len(old_labels) <= len(chunks)
        and graded_ids == [c.get("chunk_id") for c in chunks[:len(graded_ids)]]
    )
    if not prefix_intact:
        old_labels, old_reasons = [], []

    todo = chunks[len(old_labels):]
    if not todo:
        return {}

    result = grader.grade_chunks(state["question"], todo, model=state.get("model"))
    return {
        "labels": old_labels + result["labels"],
        "reasons": old_reasons + result["reasons"],
        "graded_ids": [c.get("chunk_id") for c in chunks],
        "grading_ms": state.get("grading_ms", 0.0) + result["grading_ms"],
        "grading_input_tokens": (state.get("grading_input_tokens", 0)
                                 + (result["grading_input_tokens"] or 0)),
        "grading_output_tokens": (state.get("grading_output_tokens", 0)
                                  + (result["grading_output_tokens"] or 0)),
        "api_calls": state.get("api_calls", 0) + (1 if result["graded"] else 0),
        "grader_failed": not result["parsed"],
    }


def deepen_node(state: CragState) -> Dict[str, Any]:
    """Too little was relevant: look further down the same ranked list.

    Free -- this is a local FAISS search, not an API call. The re-grade that
    follows is what costs a request.
    """
    import time
    started = time.perf_counter()
    chunks = _fetch(state, config.CRAG_DEEP_TOP_K)
    return {
        "chunks": chunks,
        "attempts": state.get("attempts", 1) + 1,
        "retrieval_ms": state.get("retrieval_ms", 0.0)
                        + (time.perf_counter() - started) * 1000,
    }


def generate_node(state: CragState) -> Dict[str, Any]:
    """Answer from the chunks that survived grading, best-graded first.

    Irrelevant chunks are dropped rather than passed through, and the rest are
    reordered so the strongest evidence leads. Both matter: everything in the
    prompt costs tokens, and material the grader just called irrelevant is
    exactly what a model confabulates around.
    """
    labels = state.get("labels") or [grader.RELEVANT] * len(state["chunks"])
    order = grader.rank(labels)
    ranked = [state["chunks"][i] for i in order]
    ranked_labels = [labels[i] for i in order]
    kept = grader.keep_useful(ranked, ranked_labels)[:config.DEFAULT_TOP_K]

    result = generate_with_usage(state["question"], kept, model=state.get("model"))
    return {
        # `chunks` deliberately NOT overwritten. It stays the full retrieved
        # set so retrieval metrics (hit rate, MRR, coverage) measure the same
        # thing for CRAG as for plain RAG. Scoring CRAG on its filtered set
        # while scoring plain RAG on its full set would make the headline
        # comparison meaningless -- CRAG would look better purely because
        # grading removed chunks before anyone counted them.
        "sent_chunks": kept,
        "answer": result["answer"],
        "generation_ms": result["generation_ms"],
        "input_tokens": result.get("input_tokens"),
        "output_tokens": result.get("output_tokens"),
        "api_calls": state.get("api_calls", 0) + 1,
        "decision": "answer",
    }


def refuse_node(state: CragState) -> Dict[str, Any]:
    """Decline without calling the model at all.

    The saving is the point: plain RAG spends a generation call producing an
    answer from chunks that do not support it. Here the grader has already
    established there is nothing to answer from, so the call is skipped.
    """
    return {
        "answer": grader.REFUSAL_TEXT,
        "sent_chunks": [],      # nothing was good enough to show or answer from
        "generation_ms": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "decision": "refuse",
    }


# --- Routing -----------------------------------------------------------------

def route_after_grading(state: CragState) -> str:
    """The one conditional edge. Pure -- all logic lives in grader.decide()."""
    already_retried = state.get("attempts", 1) >= config.CRAG_MAX_ATTEMPTS
    return grader.decide(state.get("labels") or [], already_retried)


_GRAPH = None


def crag_graph():
    """Build (once) and return the compiled LangGraph flow."""
    global _GRAPH
    if _GRAPH is None:
        builder = StateGraph(CragState)
        builder.add_node("retrieve", retrieve_node)
        builder.add_node("grade", grade_node)
        builder.add_node("deepen", deepen_node)
        builder.add_node("generate", generate_node)
        builder.add_node("refuse", refuse_node)

        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "grade")
        builder.add_conditional_edges(
            "grade",
            route_after_grading,
            {"answer": "generate", "deepen": "deepen", "refuse": "refuse"},
        )
        # The cycle that makes this a graph rather than a chain.
        builder.add_edge("deepen", "grade")
        builder.add_edge("generate", END)
        builder.add_edge("refuse", END)

        _GRAPH = builder.compile()
    return _GRAPH


# --- Public entry point ------------------------------------------------------

def crag_answer(question, top_k=None, index=None, metadata=None, model=None):
    """Answer one question through the corrective flow.

    Returns a superset of what generate.answer_traced() returns, so every
    caller and every evaluation script can treat the two interchangeably and
    the plain-RAG-vs-CRAG comparison scores identically on both sides.

    Shared with plain RAG:
        answer, chunks, retrieval_ms, generation_ms, input_tokens, output_tokens

    CRAG-specific:
        decision            "answer" | "refuse"
        labels              one grade per originally-retrieved chunk
        grade_counts        {relevant: n, partial: n, irrelevant: n}
        attempts            retrievals performed (1, or 2 if it deepened)
        deepened            whether it went looking further
        kept_chunks         how many chunks survived grading
        grading_ms          time spent grading
        grading_input_tokens / grading_output_tokens
        total_input_tokens / total_output_tokens   including grading
        api_calls           1 (refused), 2 (normal), or 3 (deepened)
        grader_failed       the grader errored or returned unparseable JSON,
                            so it fell back to plain-RAG behaviour

    `input_tokens` deliberately counts GENERATION only, matching plain RAG's
    meaning, so existing evaluation code comparing the two is not silently
    comparing different quantities. The true cost is in total_input_tokens.
    """
    final = crag_graph().invoke({
        "question": question,
        "top_k": top_k or config.DEFAULT_TOP_K,
        "index": index,
        "metadata": metadata,
        "model": model,
        "attempts": 0,
        "api_calls": 0,
        "grading_ms": 0.0,
        "grading_input_tokens": 0,
        "grading_output_tokens": 0,
    })

    labels = final.get("labels") or []
    gen_in = final.get("input_tokens") or 0
    gen_out = final.get("output_tokens") or 0

    return {
        # --- same shape as answer_traced() ---
        "answer": final.get("answer", ""),
        "chunks": final.get("chunks", []),
        "retrieval_ms": final.get("retrieval_ms", 0.0),
        "generation_ms": final.get("generation_ms", 0.0),
        "input_tokens": final.get("input_tokens"),
        "output_tokens": final.get("output_tokens"),
        # --- CRAG additions ---
        "decision": final.get("decision", "answer"),
        "labels": labels,
        "grade_counts": grader.count_labels(labels) if labels else {},
        "attempts": final.get("attempts", 1),
        "deepened": final.get("attempts", 1) > 1,
        "sent_chunks": final.get("sent_chunks", []),
        "kept_chunks": len(final.get("sent_chunks", [])),
        "grading_ms": final.get("grading_ms", 0.0),
        "grading_input_tokens": final.get("grading_input_tokens", 0),
        "grading_output_tokens": final.get("grading_output_tokens", 0),
        "total_input_tokens": gen_in + final.get("grading_input_tokens", 0),
        "total_output_tokens": gen_out + final.get("grading_output_tokens", 0),
        "api_calls": final.get("api_calls", 0),
        "grader_failed": final.get("grader_failed", False),
    }


if __name__ == "__main__":
    import sys

    if "--graph" in sys.argv:
        # The diagram in this module's docstring, generated from the real
        # wiring so the two cannot drift apart. Mermaid rather than ASCII
        # because ASCII needs an extra dependency (grandalf) and Mermaid
        # renders directly in the README on GitHub.
        print(crag_graph().get_graph().draw_mermaid())
        raise SystemExit

    question = " ".join(a for a in sys.argv[1:] if not a.startswith("--")) \
        or "What is the E2E-Spot architecture?"
    result = crag_answer(question)

    print(f"Question: {question}\n")
    print(f"Answer:\n{result['answer']}\n")
    print(f"  decision   : {result['decision']}"
          f"{'  (deepened)' if result['deepened'] else ''}")
    print(f"  grades     : {result['grade_counts']}")
    print(f"  kept       : {result['kept_chunks']} chunks")
    print(f"  api calls  : {result['api_calls']}")
    print(f"  tokens     : {result['total_input_tokens']} in "
          f"({result['grading_input_tokens']} of it grading)")
    print(f"  time       : {result['retrieval_ms']:.0f}ms retrieval + "
          f"{result['grading_ms']:.0f}ms grading + "
          f"{result['generation_ms']:.0f}ms generation")
    if result["sent_chunks"]:
        print("\n  Sources used:")
        for c in result["sent_chunks"]:
            print(f"    - {c['source_paper'][:60]} (p.{c['page_start']})")
    else:
        print(f"\n  Nothing shown: all {len(result['chunks'])} retrieved chunks "
              f"were graded irrelevant.")
