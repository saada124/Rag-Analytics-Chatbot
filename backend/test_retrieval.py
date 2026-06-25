import sys
from rag import retrieve_context

if __name__ == "__main__":
    query = "Recommend actions to reduce business risk."
    print("Testing retrieve_context for:", query)
    try:
        retrieval = retrieve_context(query)
        print("Condensed Query:", retrieval["condensed_query"])
        print("\n--- RETRIEVED CONTEXT ---")
        print(retrieval["context"])
        print("-------------------------")
    except Exception as e:
        import traceback
        traceback.print_exc()
