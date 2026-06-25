import sys
from rag import answer_rag_question
from session import ConversationMemory

if __name__ == "__main__":
    query = "Recommend actions to reduce business risk."
    print("Testing answer_rag_question for:", query)
    memory = ConversationMemory()
    try:
        response = answer_rag_question(query, memory)
        print("Response:", response)
    except Exception as e:
        import traceback
        traceback.print_exc()
