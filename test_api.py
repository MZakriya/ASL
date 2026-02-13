"""
Test script for Sign Language Recognition API
"""
import requests
import json

def test_api():
    # Test the health endpoint
    print("Testing health endpoint...")
    response = requests.get("http://localhost:8000/")
    print(f"Health check: {response.status_code} - {response.json()}")

    # Test the prediction endpoint with a sample video
    print("\nTesting prediction endpoint...")
    try:
        with open("sample_video.mp4", "rb") as f:  # Replace with actual video file
            files = {"file": f}
            response = requests.post("http://localhost:8000/predict", files=files)

        if response.status_code == 200:
            result = response.json()
            print(f"Prediction: {result['prediction']}")
        else:
            print(f"Error: {response.status_code} - {response.text}")

    except FileNotFoundError:
        print("Sample video file not found. Please provide a video file named 'sample_video.mp4'")
    except Exception as e:
        print(f"Error during prediction test: {str(e)}")

if __name__ == "__main__":
    test_api()