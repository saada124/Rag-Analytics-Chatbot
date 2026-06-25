import requests
import sys

def test_api():
    try:
        response = requests.post(
            "http://127.0.0.1:8000/chat",
            json={"message": "Recommend actions to reduce business risk."},
            timeout=90
        )
        print("Status code:", response.status_code)
        print("Response JSON:", response.json())
    except Exception as e:
        print("Error calling API:", e)

if __name__ == "__main__":
    test_api()
