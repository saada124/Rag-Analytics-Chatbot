import sys
from models import semantic_cache

if __name__ == "__main__":
    query = "Recommend actions to reduce business risk."
    response = {"answer": "test"}
    print("Testing add_to_cache...")
    try:
        semantic_cache.add_to_cache(query, response)
        print("add_to_cache passed")
    except Exception as e:
        print("Error in add_to_cache:", e)

    print("Testing get_cached_response...")
    try:
        cached = semantic_cache.get_cached_response(query)
        print("get_cached_response passed:", cached)
    except Exception as e:
        print("Error in get_cached_response:", e)
