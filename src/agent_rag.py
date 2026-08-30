"""
Agentic retrieval: answer by searching the papers, not by embedding them.

THE COMPARISON THIS EXISTS FOR
--------------------------------
The main pipeline (retrieve.py + generate.py) is *semantic* RAG:

    chunk everything -> embed everything -> embed the question ->
    return the k nearest chunks -> answer from those

One retrieval step, decided by vector similarity, fixed at k chunks.

This module is *agentic* RAG, the pattern behind coding agents like Claude
Code and Cursor:

    give the model grep/list/read tools -> it searches, reads results,
    decides what to search next -> repeats until it has enough -> answers

No embeddings, no index, no chunking, nothing precomputed. Retrieval is a
loop the model drives, so it can recover from a bad search by trying a
different pattern -- something one-shot vector search structurally cannot do.

Which is better is an empirical question, and compare_rag.py answers it on
this corpus with the same labelled questions used by evaluate.py.

WHY THE LOOP IS WRITTEN OUT BY HAND
-------------------------------------
The Gemini SDK can run this loop for you (`automatic function calling`), and
frameworks like Pydantic AI wrap it further. It is deliberately disabled here
and the loop written explicitly, for the same reason the project chose FAISS
over Chroma: the mechanism being studied should be visible, not hidden inside
a library. Every step below -- the model asking for a tool, Python executing
it, the result going back as a message -- is the thing the comparison is
about, and it is about fifteen lines of code.

Writing it out also makes the cost model obvious. Each pass through the loop
is a full API round trip that resends the entire conversation so far,
including every tool result. That is exactly why agentic retrieval costs
several times more than semantic retrieval, and you can see it happen.

TURN CAP
----------
config.AGENT_MAX_TURNS bounds the loop. Without a cap, a model that keeps
searching without converging bills indefinitely; every production agent
harness has this control (Claude Code calls it max_turns). On hitting the
cap the loop asks for a final answer from what it has, rather than returning
nothing for work already paid for.
"""

import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

import agent_tools
import config

load_dotenv()

SYSTEM_INSTRUCTION = """\
You answer questions about a collection of academic papers on sports video \
event-spotting and temporal action localization, by searching them with the \
provided tools.

How to search well:
- Start with grep_papers using SHORT, distinctive keywords. Long exact phrases \
usually fail: papers hyphenate and rephrase, so "temporal discriminability" may \
appear as "Temporal-Discriminability". If a search returns nothing, retry with a \
shorter pattern, a single keyword, or a synonym before giving up.
- Use context=3 on grep_papers to read around matches without a separate call.
- Use read_paper to read the full passage once grep shows you where to look.
- Search more than once when a question spans several papers.

Answering:
- Answer ONLY from what the tools returned. Never use outside knowledge.
- If the papers genuinely do not cover it, say so explicitly.
- Write a clean, natural answer with no inline citations or filenames in the prose.
- Be concise and technical -- the reader is a researcher.
"""

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not set. Copy .env.example to .env and add your key."
            )
        _client = genai.Client(api_key=api_key)
    return _client


def _tool_config():
    """Describe the three tools to Gemini, with automatic calling turned off."""
    declarations = [
        types.FunctionDeclaration(
            name=d["name"],
            description=d["description"],
            parameters_json_schema=d["parameters"],
        )
        for d in agent_tools.TOOL_DECLARATIONS
    ]
    return types.GenerateContentConfig(
        tools=[types.Tool(function_declarations=declarations)],
        # Off on purpose: we run the loop ourselves so it stays visible and
        # so turn count, tool calls, and tokens can be measured per question.
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        system_instruction=SYSTEM_INSTRUCTION,
    )


class DailyQuotaExhausted(RuntimeError):
    """The free tier's per-day request quota is gone. Waiting will not help."""


def _request_with_retry(contents, cfg, model=None):
    """One API call, retrying per-minute rate limits with backoff.

    Free-tier Gemini enforces two different 429s and they need opposite
    responses, so they are told apart here rather than blindly retried:

      PerMinute quota -- transient. An agentic run spends several requests per
        question and will trip this constantly. Sleeping a few seconds fixes
        it, and that is what makes an unattended benchmark survive to the end.
      PerDay quota    -- not transient. Sleeping 20s, 40s, 60s just wastes
        minutes before failing anyway. Raise immediately with an explanation,
        because the only real fixes are waiting for reset or switching model.

    Treating both the same was the original implementation, and it burned five
    backoff sleeps against a daily cap that had hours left on it.
    """
    client = _get_client()
    model = model or config.GEMINI_MODEL_NAME

    for attempt in range(config.API_RETRY_ATTEMPTS):
        try:
            return client.models.generate_content(
                model=model, contents=contents, config=cfg
            )
        except Exception as exc:  # noqa: BLE001
            message = str(exc)
            is_quota = "429" in message or "RESOURCE_EXHAUSTED" in message
            if is_quota and "PerDay" in message:
                raise DailyQuotaExhausted(
                    f"Free-tier daily request quota for {model!r} is exhausted. "
                    f"Wait for the daily reset or point config.BENCHMARK_MODEL_NAME "
                    f"at a different model."
                ) from exc
            if not is_quota or attempt == config.API_RETRY_ATTEMPTS - 1:
                raise
            wait = config.API_RETRY_BASE_SECONDS * (attempt + 1)
            print(f"    rate limited, waiting {wait}s...", flush=True)
            time.sleep(wait)


def agentic_answer(question, max_turns=None, verbose=False, model=None):
    """Answer `question` by letting the model search the papers with tools.

    Returns a dict shaped to line up with generate.answer_traced(), so
    compare_rag.py can measure both arms the same way:

        {"answer", "tool_calls", "turns", "papers_used", "hit_cap",
         "total_ms", "input_tokens", "output_tokens"}

    `papers_used` is the set of paper filenames the model actually opened or
    matched -- the agentic equivalent of "which chunks were retrieved", and
    what makes Hit Rate comparable between the two approaches.
    """
    max_turns = max_turns or config.AGENT_MAX_TURNS
    cfg = _tool_config()

    contents = [types.Content(role="user", parts=[types.Part(text=question)])]
    tool_calls = []
    papers_used = set()
    input_tokens = output_tokens = 0
    started = time.perf_counter()
    hit_cap = False
    answer = None

    for turn in range(max_turns):
        response = _request_with_retry(contents, cfg, model=model)

        usage = getattr(response, "usage_metadata", None)
        if usage:
            input_tokens += usage.prompt_token_count or 0
            output_tokens += usage.candidates_token_count or 0

        candidate = response.candidates[0]
        parts = candidate.content.parts or []
        calls = [p.function_call for p in parts if getattr(p, "function_call", None)]

        # No tool call means the model is done searching and has answered.
        if not calls:
            answer = response.text
            break

        # Record the model's turn verbatim, then run each tool it asked for.
        contents.append(candidate.content)
        results = []
        for call in calls:
            args = dict(call.args or {})
            func = agent_tools.TOOL_FUNCTIONS.get(call.name)
            if func is None:
                output = f"Error: unknown tool {call.name!r}."
            else:
                try:
                    output = func(**args)
                except Exception as exc:  # noqa: BLE001
                    # Returned to the model, not raised: it can read the error
                    # and try a different call instead of the run dying.
                    output = f"Error running {call.name}: {exc}"

            tool_calls.append({"tool": call.name, "args": args, "chars": len(output)})
            papers_used.update(_papers_mentioned(output))

            if verbose:
                preview = str(args)[:70]
                print(f"    turn {turn + 1}: {call.name}({preview}) -> {len(output)} chars")

            results.append(
                types.Part.from_function_response(
                    name=call.name, response={"result": output}
                )
            )
        contents.append(types.Content(role="user", parts=results))
    else:
        # Loop exhausted without the model volunteering an answer. Rather than
        # discard everything it gathered, ask once more for a final answer.
        hit_cap = True
        contents.append(types.Content(role="user", parts=[types.Part(
            text="Stop searching now and answer from what you have already found. "
                 "If it is not enough, say so explicitly."
        )]))
        final = _request_with_retry(contents, types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION
        ), model=model)
        usage = getattr(final, "usage_metadata", None)
        if usage:
            input_tokens += usage.prompt_token_count or 0
            output_tokens += usage.candidates_token_count or 0
        answer = final.text

    return {
        "answer": answer or "",
        "tool_calls": tool_calls,
        "turns": len(tool_calls) and (turn + 1) or 1,
        "papers_used": sorted(papers_used),
        "hit_cap": hit_cap,
        "total_ms": (time.perf_counter() - started) * 1000,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def _papers_mentioned(tool_output):
    """Which paper files a tool result refers to.

    grep_papers prefixes every hit with 'filename.txt:line:', and read_paper is
    called with a filename, so scanning tool output for known paper names
    recovers what the agent actually looked at -- the agentic equivalent of the
    retrieved-chunk list, needed to score Hit Rate the same way for both arms.
    """
    found = set()
    for path in config.PAPERS_TEXT_DIR.glob("*.txt"):
        if path.name in tool_output:
            found.add(path.stem + ".pdf")  # map back to the PDF name evaluate.py uses
    return found


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "How does T-DEED improve temporal discriminability?"
    print(f"Question: {question}\n")
    result = agentic_answer(question, verbose=True)

    print(f"\nAnswer:\n{result['answer']}\n")
    print(f"Papers used: {', '.join(result['papers_used']) or 'none'}")
    print(f"Turns: {result['turns']}  Tool calls: {len(result['tool_calls'])}"
          f"{'  (hit turn cap)' if result['hit_cap'] else ''}")
    print(f"Tokens: {result['input_tokens']} in / {result['output_tokens']} out")
    print(f"Time: {result['total_ms'] / 1000:.1f}s")
