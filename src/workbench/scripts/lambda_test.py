"""
Local test harness for AWS Lambda scripts.

Usage:
    lambda_test <lambda_script.py>

Required: testing/event.json with the event definition
Options: testing/env.json with a set of ENV vars
"""

import sys
import os
import json
import importlib.util


def main():
    if len(sys.argv) != 2:
        print("Usage: lambda_launcher <handler_module_name>")
        print("\nOptional: testing/event.json with test event")
        print("Optional: testing/env.json with environment variables")
        sys.exit(1)

    handler_file = sys.argv[1]

    # Add .py if not present
    if not handler_file.endswith(".py"):
        handler_file += ".py"

    # Check if file exists
    if not os.path.exists(handler_file):
        print(f"Error: File '{handler_file}' not found")
        sys.exit(1)

    # Load environment variables from env.json if it exists
    if os.path.exists("testing/env.json"):
        print("Loading environment variables from testing/env.json")
        with open("testing/env.json") as f:
            env_vars = json.load(f)
            for key, value in env_vars.items():
                os.environ[key] = value
                print(f"  Set {key} = {value}")
        print()

    # Load event configuration
    if os.path.exists("testing/event.json"):
        print("Loading event from testing/event.json")
        with open("testing/event.json") as f:
            event = json.load(f)
    else:
        print("No testing/event.json found, using empty event")
        event = {}

    # Load the module dynamically
    spec = importlib.util.spec_from_file_location("lambda_module", handler_file)
    lambda_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lambda_module)

    # Call the lambda_handler
    print(f"Invoking lambda_handler from {handler_file}...")
    print("-" * 50)
    print(f"Event: {json.dumps(event, indent=2)}")
    print("-" * 50)

    result = lambda_module.lambda_handler(event, {})

    print("-" * 50)
    print("Result:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
