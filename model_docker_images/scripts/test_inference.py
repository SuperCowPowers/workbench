#!/usr/bin/env python
import requests
import json
import argparse
import time


def test_inference_server(host="localhost", port=8080):
    """
    Test the inference server running in the Docker container.
    """
    base_url = f"http://{host}:{port}"

    # Test 1: Check the health endpoint
    print("\n🔍 Testing /ping endpoint (health check)...")
    try:
        response = requests.get(f"{base_url}/ping", timeout=5)
        if response.status_code == 200:
            print("✅ Health check succeeded")
        else:
            print(f"❌ Health check failed with status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check failed with error: {e}")
        print("Is the Docker container running on the specified port?")
        return False

    # Test 2: Test the invocations endpoint with simple data
    print("\n🔍 Testing /invocations endpoint with sample data...")
    sample_data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

    try:
        # Test with JSON data
        response = requests.post(
            f"{base_url}/invocations",
            data=json.dumps(sample_data),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            timeout=5
        )

        if response.status_code == 200:
            print("✅ Inference request succeeded")
            try:
                # Parse the JSON response
                result = response.json()
                print(f"📊 Response: {result}")
                return True
            except json.JSONDecodeError as e:
                print(f"❌ Error parsing response as JSON: {e}")
                print(f"Raw response: {response.text}")
                # Try parsing as CSV
                try:
                    lines = response.text.strip().split('\n')
                    values = [float(line) for line in lines]
                    print(f"📊 CSV Response (converted): {values}")
                    return True
                except Exception:
                    return False
        else:
            print(f"❌ Inference request failed with status code: {response.status_code}")
            print(f"Response text: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Inference request failed with error: {e}")
        return False

    print("\n🎉 All tests passed! Your inference server is working correctly.")
    return True


def run_docker_command():
    """
    Print the docker run command to help the user start the container.
    """
    print("\n📋 To run your Docker container, use the following command:")
    print("docker run -p 8080:8080 aws_model_inference:latest")
    print("\nThis maps port 8080 from the container to port 8080 on your host machine.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the AWS model inference server")
    parser.add_argument("--host", default="localhost", help="Host where the inference server is running")
    parser.add_argument("--port", type=int, default=8080, help="Port where the inference server is running")
    parser.add_argument("--docker-cmd", action="store_true", help="Print the docker run command")

    args = parser.parse_args()

    if args.docker_cmd:
        run_docker_command()

    test_inference_server(args.host, args.port)
