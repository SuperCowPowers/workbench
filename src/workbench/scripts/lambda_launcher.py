import sys
import os
import importlib.util


def main():
    if len(sys.argv) != 2:
        print("Usage: lambda_launcher <handler_module_name>")
        sys.exit(1)

    handler_file = sys.argv[1]

    # Add .py if not present
    if not handler_file.endswith(".py"):
        handler_file += ".py"

    # Check if file exists
    if not os.path.exists(handler_file):
        print(f"Error: File '{handler_file}' not found")
        sys.exit(1)

    # Load the module dynamically
    spec = importlib.util.spec_from_file_location("lambda_module", handler_file)
    lambda_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lambda_module)

    # Call the lambda_handler
    print(f"Invoking lambda_handler from {handler_file}...")
    print("-" * 50)

    result = lambda_module.lambda_handler({}, {})

    print("-" * 50)
    print("Result:")
    print(result)


if __name__ == "__main__":
    main()
