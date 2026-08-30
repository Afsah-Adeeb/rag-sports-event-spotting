"""
The three tools an agentic retriever searches with: list, grep, read.

WHAT THIS IS FOR
------------------
This is the retrieval half of the *comparison arm* of the project. The main
pipeline retrieves by meaning: embed everything up front, embed the question,
return the nearest chunks (retrieve.py). This one retrieves by searching:
give the model tools and let it look things up itself, the way you would with
Ctrl+F -- no embeddings, no index, no chunking.

That is the approach coding agents (Claude Code, Cursor, Codex) use over
codebases, and the point of implementing it here is to find out whether it
also works over research papers, measured rather than assumed. compare_rag.py
runs both against the same labelled questions.

WHY PLAIN PYTHON REGEX AND NOT ripgrep
----------------------------------------
Production agent harnesses shell out to `rg` because it is dramatically
faster on large repositories. This corpus is 9 papers and 600KB of text --
Python's `re` scans that in milliseconds, and requiring users to install a
Rust binary to run a portfolio project is a worse trade than a few
milliseconds. The tool *interface* is what matters for the comparison, not
the search binary behind it.

WHAT IS KEPT FROM PRODUCTION PRACTICE
---------------------------------------
The things that actually matter for correctness and cost are kept:

  - Every output is bounded (max_results, max_lines). An unbounded tool
    result goes straight into the model's context window and is billed.
  - Paths are validated against the papers directory, so a model that asks
    for "../../.env" gets an error string instead of the file. The model
    supplies the path; Python decides if it is allowed.
  - Errors are returned as readable strings, never raised. A raise would
    kill the agent loop; a string lets the model read "no matches, try a
    different pattern" and recover on its own -- which is the entire
    advantage of an agentic loop over one-shot retrieval.
"""

import re

import config


def _paper_paths():
    return sorted(config.PAPERS_TEXT_DIR.glob("*.txt"))


def _safe_path(name):
    """Resolve a model-supplied filename inside the papers directory, or None.

    The model chooses the path, so it is untrusted input: resolve it and
    confirm it did not escape the directory before opening anything.
    """
    target = (config.PAPERS_TEXT_DIR / name).resolve()
    if not target.is_relative_to(config.PAPERS_TEXT_DIR.resolve()):
        return None
    return target


def list_papers():
    """List every paper available to search, with its size in lines."""
    paths = _paper_paths()
    if not paths:
        return ("Error: no text files found. Run export_text.py to generate them "
                "from the PDFs in data/papers/.")
    rows = []
    for path in paths:
        lines = path.read_text(encoding="utf-8").count("\n") + 1
        rows.append(f"{path.name} ({lines} lines)")
    return "\n".join(rows)


def grep_papers(pattern, max_results=None, context=0):
    """Search every paper for a case-insensitive regex, returning file:line: text."""
    max_results = max_results or config.AGENT_GREP_MAX_RESULTS

    try:
        rx = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        # Returned, not raised: the model reads this and tries another pattern.
        return f"Error: invalid regex {pattern!r}: {exc}"

    def clip(text):
        """One match preview. Cap each line so one long sentence can't eat the budget."""
        text = text.strip()
        if len(text) > config.AGENT_GREP_LINE_CHARS:
            return text[:config.AGENT_GREP_LINE_CHARS] + " ...[truncated]"
        return text

    # Collect matches per paper first, then spend the budget round-robin across
    # papers rather than draining it in filename order.
    #
    # This matters more than it looks. The first implementation walked files
    # alphabetically and stopped the moment the budget ran out, so a broad
    # pattern was answered entirely from whichever papers sort first. Measured:
    # grep("E2E-Spot") filled 11,449 of its 12,000 characters on papers 1-5 and
    # never reached "Spotting Temporally Precise..." -- the paper that actually
    # introduces E2E-Spot, 6th alphabetically. The model could not cite the
    # right source because it was never shown it, which looks like a failure of
    # agentic retrieval but is really a failure of this function.
    #
    # Round-robin guarantees every matching paper is represented before any
    # paper gets a second match, so breadth survives truncation.
    per_paper = {}
    for path in _paper_paths():
        lines = path.read_text(encoding="utf-8").splitlines()
        blocks = []
        for i, line in enumerate(lines):
            if not rx.search(line):
                continue
            if context > 0:
                lo, hi = max(0, i - context), min(len(lines), i + context + 1)
                blocks.append("\n".join(
                    f"{path.name}:{n + 1}: {clip(lines[n])}" for n in range(lo, hi)
                ) + "\n--")
            else:
                blocks.append(f"{path.name}:{i + 1}: {clip(line)}")
        if blocks:
            per_paper[path.name] = blocks

    if not per_paper:
        return (f"No matches for pattern: {pattern}. Try a shorter or more "
                f"general pattern, or a synonym.")

    total_matches = sum(len(v) for v in per_paper.values())
    hits = []
    used_chars = 0
    exhausted = False
    round_index = 0
    while not exhausted:
        added_this_round = False
        for name, blocks in per_paper.items():
            if round_index >= len(blocks):
                continue
            block = blocks[round_index]
            if (len(hits) >= max_results
                    or used_chars + len(block) > config.AGENT_GREP_MAX_CHARS):
                exhausted = True
                break
            hits.append(block)
            used_chars += len(block)
            added_this_round = True
        if not added_this_round:
            break
        round_index += 1

    header = (f"{total_matches} match(es) across {len(per_paper)} paper(s)"
              f"{'; showing a sample from each' if len(hits) < total_matches else ''}.")
    out = header + "\n" + "\n".join(hits)
    if len(hits) < total_matches:
        out += ("\n... result truncated. Narrow the pattern, or use read_paper "
                "to read around a specific match.")
    return out


def read_paper(name, offset=1, limit=None):
    """Read a bounded line range from one paper, with line numbers.

    Bounded because papers run to thousands of lines: an unbounded read would
    dump an entire paper into the context window on every call.
    """
    limit = limit or config.AGENT_READ_MAX_LINES
    limit = min(limit, config.AGENT_READ_MAX_LINES)

    target = _safe_path(name)
    if target is None:
        return f"Error: {name!r} is outside the papers directory."
    if not target.exists():
        available = ", ".join(p.name for p in _paper_paths())
        return f"Error: no paper named {name!r}. Available: {available}"
    if offset < 1:
        return "Error: offset must be 1 or greater."

    lines = target.read_text(encoding="utf-8").splitlines()
    end = min(offset + limit - 1, len(lines))
    excerpt = lines[offset - 1:end]
    if not excerpt:
        return f"No lines in range. {name} has {len(lines)} lines."

    # Same reasoning as grep: a line here is a whole paragraph, so the line
    # count alone does not bound the size. Stop at the character budget and
    # tell the model exactly where to resume.
    out_lines = []
    used_chars = 0
    stopped_at = offset
    for i, line in enumerate(excerpt, start=offset):
        entry = f"{i}: {line}"
        if used_chars + len(entry) > config.AGENT_READ_MAX_CHARS and out_lines:
            break
        out_lines.append(entry)
        used_chars += len(entry)
        stopped_at = i

    body = "\n".join(out_lines)
    if stopped_at < len(lines):
        body += (f"\n... stopped at line {stopped_at} of {len(lines)} "
                 f"(call again with offset={stopped_at + 1} to continue).")
    return body


# --- Tool declarations handed to Gemini -------------------------------------
# The model only ever sees these descriptions, so they are the real interface.
# They say *when* to reach for each tool, not just what it does -- a model that
# reads a whole paper when it should have grepped burns the turn budget fast.
TOOL_DECLARATIONS = [
    {
        "name": "list_papers",
        "description": (
            "List every research paper available to search, with its length in lines. "
            "Call this first if you are unsure what the collection contains."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "grep_papers",
        "description": (
            "Search across all papers for a case-insensitive regular expression and "
            "return matching lines as 'file:line: text'. This is the main way to find "
            "where a topic is discussed. Prefer short, distinctive keywords over long "
            "phrases, which rarely match verbatim. If a search returns nothing, try a "
            "synonym or a shorter pattern."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Case-insensitive regex to search for."},
                "context": {
                    "type": "integer",
                    "description": "Lines of surrounding context per match (0-5). Use 2-3 to read "
                                   "around a match without a separate read_paper call.",
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "read_paper",
        "description": (
            "Read a bounded range of lines from one paper, given its exact filename "
            "from grep_papers or list_papers. Use after grep to read the full passage "
            "around a promising match. Page markers appear in the text as '[page N]' -- "
            "use them to cite pages."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Exact .txt filename of the paper."},
                "offset": {"type": "integer", "description": "First line to read (1-based)."},
                "limit": {"type": "integer", "description": "How many lines to read."},
            },
            "required": ["name"],
        },
    },
]

# Name -> callable, used by the agent loop to dispatch a model's tool call.
TOOL_FUNCTIONS = {
    "list_papers": list_papers,
    "grep_papers": grep_papers,
    "read_paper": read_paper,
}


if __name__ == "__main__":
    print("--- list_papers() ---")
    print(list_papers())
    print("\n--- grep_papers('temporal discriminability') ---")
    print(grep_papers("temporal discriminability")[:800])
    print("\n--- read_paper(first paper, offset=1, limit=8) ---")
    first = _paper_paths()[0].name
    print(read_paper(first, offset=1, limit=8))
    print("\n--- path traversal is refused ---")
    print(read_paper("../../.env"))
