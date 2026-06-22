import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.query import query_rag_pipeline

def main():
    print("=" * 60)
    print("        AI ENGINEERING RAG INTERACTIVE TERMINAL APP")
    print("=" * 60)
    print("Type 'exit' or 'quit' to close.\n")

    while True:
        try:
            query = input("\nAsk a question about your documents: ")
            if query.lower() in ['exit', 'quit']:
                break
            if not query.strip():
                continue
                
            print("\nSearching context knowledge base...")
            output = query_rag_pipeline(query)
            print("\n" + "="*10 + " RESPONSE " + "="*10)
            print(output["answer"])
            print("=" * 30)
        except (KeyboardInterrupt, SystemExit):
            break

if __name__ == "__main__":
    main()
