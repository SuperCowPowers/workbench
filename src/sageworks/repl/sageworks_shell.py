from IPython.terminal.embed import InteractiveShellEmbed
from IPython.terminal.prompts import Prompts, Token
from IPython.utils.coloransi import TermColors as color
import os
import logging

# SageWorks Imports
import sageworks  # noqa: F401
from sageworks.utils.sageworks_logging import IMPORTANT_LEVEL_NUM
logging.getLogger("sageworks").setLevel(logging.WARNING)
from sageworks.views.artifacts_text_view import ArtifactsTextView
from sageworks.api.data_source import DataSource  # noqa: F401
from sageworks.api.feature_set import FeatureSet  # noqa: F401
from sageworks.api.model import Model, ModelType  # noqa: F401
from sageworks.api.endpoint import Endpoint  # noqa: F401


class SageWorksShell:
    def __init__(self):
        # Set up the Prompt and the IPython shell
        config = InteractiveShellEmbed.instance().config
        config.TerminalInteractiveShell.autocall = 2
        config.InteractiveShellEmbed.colors = 'Linux'
        config.InteractiveShellEmbed.color_info = True
        config.TerminalInteractiveShell.prompts_class = self.SageWorksPrompt
        self.shell = InteractiveShellEmbed(config=config, banner1="Welcome to SageWorks", exit_msg="Goodbye from SageWorks!")

        # Create a class instance of the ArtifactsTextView
        self.artifacts_text_view = ArtifactsTextView()

        # Register our custom commands
        self.commands = dict()
        self.commands["help"] = self.help
        self.commands["incoming_data"] = self.incoming_data
        self.commands["glue_jobs"] = self.glue_jobs
        self.commands["data_sources"] = self.data_sources
        self.commands["feature_sets"] = self.feature_sets
        self.commands["models"] = self.models
        self.commands["endpoints"] = self.endpoints

    class SageWorksPrompt(Prompts):
        def in_prompt_tokens(self, cli=None):
            aws_profile = os.getenv("AWS_PROFILE", "default")
            return [
                (Token, "🍺  "),
                (Token.Prompt, f"SageWorks({aws_profile})> ")
            ]

    def start(self):
        """
        Start the SageWorks IPython shell.
        """
        self.shell(local_ns=self.commands)

    @staticmethod
    def help():
        print("Commands:")
        print("  - incoming_data: List all the incoming S3 data")
        print("  - glue_jobs: List all the Glue Jobs in AWS")
        print("  - data_sources: List all the DataSources in AWS")
        print("  - feature_sets: List all the FeatureSets in AWS")
        print("  - models: List all the Models in AWS")
        print("  - endpoints: List all the Endpoints in AWS")
        print("  - set_log_level <level>: Set the log level to debug or important")
        print("  - broker_refresh: Force a refresh of the AWS broker data")
        print("  - exit: Exit SageWorks REPL")

    def incoming_data(self):
        print(self.artifacts_text_view.incoming_data_summary())

    def glue_jobs(self):
        print(self.artifacts_text_view.glue_jobs_summary())

    def data_sources(self):
        print(self.artifacts_text_view.data_sources_summary())

    def feature_sets(self):
        print(self.artifacts_text_view.feature_sets_summary())

    def models(self):
        print(self.artifacts_text_view.models_summary())

    def endpoints(self):
        print(self.artifacts_text_view.endpoints_summary())

    @staticmethod
    def set_debug(self):
        logging.getLogger("sageworks").setLevel(logging.DEBUG)

    @staticmethod
    def set_info(self):
        logging.getLogger("sageworks").setLevel(logging.INFO)

    @staticmethod
    def set_important(self):
        logging.getLogger("sageworks").setLevel(IMPORTANT_LEVEL_NUM)

    @staticmethod
    def set_warning(self):
        logging.getLogger("sageworks").setLevel(logging.WARNING)

    def broker_refresh(self):
        self.artifacts_text_view.refresh(force_refresh=True)


# Launch Shell Entry Point
def launch_shell():
    shell = SageWorksShell()
    shell.start()


# Start the shell when running the script
if __name__ == "__main__":
    launch_shell()
