import sys
from models import llm_client

if __name__ == "__main__":
    print("Testing basic invoke...")
    try:
        response = llm_client.invoke("Hello!")
        print("Response:", response.content)
    except Exception as e:
        print("Error:", e)
