import json
import time

# Get the boto3 session from the SageWorks Account Clamp
from sageworks.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp

session = AWSAccountClamp().boto3_session

# Create EventBridge client
eventbridge_client = session.client("events")

# Define the event bus name
event_bus_name = "sageworks"


# Function to monitor events
def monitor_events():
    while True:
        try:
            response = eventbridge_client.list_rules(EventBusName=event_bus_name)
            print(f"Monitoring events from EventBus: {event_bus_name}")

            for rule in response.get("Rules", []):
                print(f"\nRule: {rule['Name']}")
                events_response = eventbridge_client.describe_rule(Name=rule["Name"], EventBusName=event_bus_name)
                print(json.dumps(events_response, indent=2))

            # Wait for a short period before checking again
            time.sleep(10)

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            time.sleep(10)


if __name__ == "__main__":
    monitor_events()
