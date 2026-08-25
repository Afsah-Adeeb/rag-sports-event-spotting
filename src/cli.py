"""
Step 6: A minimal command-line chat loop over the RAG pipeline.

Deliberately simple: read a question, retrieve + generate an answer,
print it with sources, repeat. No conversation memory across turns (each
question is answered independently) -- keeping it stateless keeps the
retrieval step easy to reason about, which matters more here than a
chatty multi-turn UX.
"""

from generate import generate_answer

HELP_TEXT = "Ask a question about the loaded papers. Type 'exit' or 'quit' to stop.\n"


def main():
    print("RAG Q&A over sports video event-spotting papers")
    print(HELP_TEXT)

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

        answer, sources = generate_answer(question)

        print(f"\n{answer}\n")
        print("Sources:")
        for chunk in sources:
            print(f"  - {chunk['source_paper']} (p.{chunk['page_start']}-{chunk['page_end']}, score={chunk['score']:.3f})")
        print()


if __name__ == "__main__":
    main()
