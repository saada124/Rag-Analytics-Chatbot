import sys
from router import classify_query

if __name__ == "__main__":
    query = "Recommend actions to reduce business risk."
    print("Testing query:", query)
    try:
        route = classify_query(query)
        print("Route:", route)
    except Exception as e:
        print("Error:", e)
