"""
Step 6: A minimal command-line chat loop over the RAG pipeline.

Deliberately simple: read a question, retrieve + generate an answer,
print it with sources, repeat. No conversation memory across turns (each
question is answered independently) -- keeping it stateless keeps the
retrieval step easy to reason about, which matters more here than a
chatty multi-turn UX.

Questions asked here are recorded by telemetry.py the same way the web app
records them, tagged surface="cli". Monitoring belongs to the pipeline, not
to one front end -- otherwise the metrics only describe whoever used the GUI.
"""

import telemetry
from generate import answer_traced

HELP_TEXT = "Ask a question about the loaded papers. Type 'exit' or 'quit' to stop.\n"


def main():
    print("RAG Q&A over sports video event-spotting papers")
    print(HELP_TEXT)

    session_id = telemetry.new_query_id()  # groups one terminal run's questions

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            print("Goodbye.")
            break

        result = answer_traced(question)
        sources = result["chunks"]

        telemetry.log_query(
            question=question,
            answer=result["answer"],
            retrieved_chunks=sources,
            retrieval_ms=result["retrieval_ms"],
            generation_ms=result["generation_ms"],
            top_k=len(sources),
            usage=result,
            surface="cli",
            session_id=session_id,
        )

        print(f"\n{result['answer']}\n")
        print("Sources:")
        for chunk in sources:
            print(f"  - {chunk['source_paper']} (p.{chunk['page_start']}-{chunk['page_end']}, score={chunk['score']:.3f})")
        print(f"\n[{result['retrieval_ms']:.0f}ms retrieval + "
              f"{result['generation_ms'] / 1000:.1f}s generation]\n")


if __name__ == "__main__":
    main()
