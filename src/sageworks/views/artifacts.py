"""AWSServiceBroker pulls and collects metadata from a bunch of AWS Services"""
import sys
import argparse

# Local Imports
from sageworks.views.view import View


class Artifacts(View):
    def __init__(self):
        """Artifacts: Helper Class for AWS SageMaker Artifacts"""
        # Call SuperClass Initialization
        super().__init__()

        # Get AWS Service information for ALL the categories (data_source, feature_set, endpoints, etc)
        self.service_info = self.aws_broker.get_all_metadata()

    def check(self) -> bool:
        """Can we connect to this view/service?"""
        True  # I'm great, thx for asking

    def refresh(self) -> bool:
        """Refresh data/metadata associated with this view"""
        self.service_info = self.aws_broker.get_all_metadata()

    def view_data(self) -> dict:
        """Return all the data that's useful for this view"""

        # We're going to simply summarise the first two level of keys (make kewler later)
        summary_data = {}
        for key, value in self.service_info.items():
            summary_data[key] = list(value.keys())
        return summary_data


if __name__ == '__main__':
    from pprint import pprint

    # Collect args from the command line
    parser = argparse.ArgumentParser()
    args, commands = parser.parse_known_args()

    # Check for unknown args
    if commands:
        print('Unrecognized args: %s' % commands)
        sys.exit(1)

    # Create the class and get the AWS Model Registry details
    artifact_view = Artifacts()
    artifact_view.refresh()

    # List the Endpoint Names
    print('Artifacts:')
    pprint(artifact_view.view_data())
