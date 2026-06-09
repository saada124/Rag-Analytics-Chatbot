import sys
import os

# Add backend directory to sys.path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
sys.path.append(backend_dir)

from router import route_query
from analytics import load_dataframes

if __name__ == "__main__":
    print("Loading dataframes...")
    load_dataframes()
    
    # Test query
    q = "show me the user's country with the signup date before 2022 and show the score"
    print(f"\nRouting query: '{q}'\n")
    res = route_query(q)
    print("\nResult:")
    import json
    print(json.dumps(res, indent=2))
