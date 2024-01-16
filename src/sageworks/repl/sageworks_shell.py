from IPython.terminal.embed import InteractiveShellEmbed
from IPython.terminal.prompts import Prompts
from pygments.style import Style
from pygments.token import Token
import sys
import logging
import importlib
import botocore
import webbrowser

# SageWorks Imports
from sageworks.utils.repl_utils import cprint
import sageworks  # noqa: F401
from sageworks.utils.sageworks_logging import IMPORTANT_LEVEL_NUM
from sageworks.utils.config_manager import ConfigManager

logging.getLogger("sageworks").setLevel(IMPORTANT_LEVEL_NUM)


class CustomPromptStyle(Style):
    styles = {
        Token.SageWorks: "#ff69b4",  # Pink color for SageWorks
        Token.AWSProfile: "#ffd700",  # Yellow color for AWS Profile
        Token.Lightblue: "#5fd7ff",
        Token.Lightpurple: "#af87ff",
        Token.Lightgreen: "#87ff87",
        Token.Lime: "#afff00",
        Token.Darkyellow: "#ffd787",
        Token.Orange: "#ff8700",
        Token.Red: "#ff5f87",
        Token.Blue: "#4444d7",
        Token.Green: "#22cc22",
    }


# Note: Hack
aws_profile = ConfigManager().get_config("AWS_PROFILE")


class SageWorksShell:
    def __init__(self):
        # Check the SageWorks config
        self.cm = ConfigManager()
        self.cm.load_config()
        if self.cm.is_default_config:
            # Invoke Onboarding Procedure
            cprint("yellow", "Default SageWorks Config Detected...running onboarding procedure...")
            self.onboard()

        # Perform AWS connection test and other checks
        self.commands = dict()
        self.artifacts_text_view = None
        self.check_aws_account()
        self.check_redis()
        self.import_sageworks()

        # Set up the Prompt and the IPython shell
        config = InteractiveShellEmbed.instance().config
        config.TerminalInteractiveShell.autocall = 2
        config.TerminalInteractiveShell.prompts_class = self.SageWorksPrompt
        config.TerminalInteractiveShell.highlighting_style = CustomPromptStyle
        self.shell = InteractiveShellEmbed(config=config, banner1="", exit_msg="Goodbye from SageWorks!")

        # Register our custom commands
        self.commands["hey"] = self.hey
        self.commands["docs"] = self.doc_browser
        self.commands["summary"] = self.summary
        self.commands["incoming_data"] = self.incoming_data
        self.commands["glue_jobs"] = self.glue_jobs
        self.commands["data_sources"] = self.data_sources
        self.commands["feature_sets"] = self.feature_sets
        self.commands["models"] = self.models
        self.commands["endpoints"] = self.endpoints
        self.commands["log_debug"] = self.log_debug
        self.commands["log_info"] = self.log_info
        self.commands["log_important"] = self.log_important
        self.commands["log_warning"] = self.log_warning
        self.commands["broker_refresh"] = self.broker_refresh

    class SageWorksPrompt(Prompts):
        def in_prompt_tokens(self, cli=None):
            lights = SageWorksShell.status_lights()
            aws_profile_prompt = [(Token.Blue, ":"), (Token.AWSProfile, f"{aws_profile}"), (Token.Blue, "> ")]
            return lights + [(Token.SageWorks, "SageWorks")] + aws_profile_prompt

    def start(self):
        """Start the SageWorks IPython shell"""
        cprint("magenta", "Welcome to SageWorks!")
        self.hey()
        self.summary()
        self.shell(local_ns=self.commands)

    @staticmethod
    def check_config():
        """Check the current Configuration Status"""
        return True

    def check_redis(self):
        """Check the Redis Cache"""
        from sageworks.utils.sageworks_cache import SageWorksCache

        # Grab the Redis Host and Port
        host = self.cm.get_config("REDIS_HOST")
        port = self.cm.get_config("REDIS_PORT")

        # Open the Redis connection (class object)
        cprint("lime", f"Checking Redis connection to: {host}:{port}..")
        if SageWorksCache().check():
            cprint("lightgreen", "Redis Cache Check Success...")
        else:
            cprint("yellow", "Redis Cache Check Failed...check your SageWorks Config...")

    @staticmethod
    def check_aws_account():
        """Check if the AWS Account is Set up Correctly"""
        cprint("yellow", "Checking AWS Account Connection...")
        try:
            try:
                aws_clamp = importlib.import_module("sageworks.aws_service_broker.aws_account_clamp").AWSAccountClamp()
                cprint("lightgreen", "AWS Account Clamp Created...")
                aws_clamp.check_aws_identity()
                cprint("lightgreen", "AWS Identity Check...")
                aws_clamp.boto_session()
                cprint("lightgreen", "AWS Account Check AOK!")
            except RuntimeError:
                print("AWS Account Check Failed: Check AWS_PROFILE and/or Renew SSO Token...")
                sys.exit(1)
        except botocore.exceptions.ProfileNotFound:
            print("AWS Account Check Failed: Check AWS_PROFILE...")
            sys.exit(1)
        except botocore.exceptions.NoCredentialsError:
            print("AWS Account Check Failed: Check AWS Credentials...")
            sys.exit(1)

    def import_sageworks(self):
        # Import all the SageWorks modules
        cprint("lightpurple", "Pulling AWS Artifacts...")
        self.artifacts_text_view = importlib.import_module("sageworks.views.artifacts_text_view").ArtifactsTextView()

        # These are the classes we want to expose to the REPL
        self.commands["DataSource"] = importlib.import_module("sageworks.api.data_source").DataSource
        self.commands["FeatureSet"] = importlib.import_module("sageworks.api.feature_set").FeatureSet
        self.commands["Model"] = importlib.import_module("sageworks.api.model").Model
        self.commands["ModelType"] = importlib.import_module("sageworks.api.model").ModelType
        self.commands["Endpoint"] = importlib.import_module("sageworks.api.endpoint").Endpoint
        self.commands["Monitor"] = importlib.import_module("sageworks.api.monitor").Monitor

    @staticmethod
    def help_txt():
        help_msg = """    Commands:
        - hey: Show this help message
        - docs: Open browser to show SageWorks Documentation
        - summary: Show a summary of all the AWS Artifacts
        - incoming_data: List all the incoming S3 data
        - glue_jobs: List all the Glue Jobs in AWS
        - data_sources: List all the DataSources in AWS
        - feature_sets: List all the FeatureSets in AWS
        - models: List all the Models in AWS
        - endpoints: List all the Endpoints in AWS
        - broker_refresh: Force a refresh of the AWS broker data
        - exit: Exit SageWorks REPL"""
        return help_msg

    def hey(self):
        cprint("lightblue", self.help_txt())

    @staticmethod
    def doc_browser():
        """Open a browser and start the Dash app and open a browser."""
        url = "https://supercowpowers.github.io/sageworks/"
        webbrowser.open(url)

    def summary(self):
        cprint("yellow", "\nAWS Artifacts Summary:")
        view_data = self.artifacts_text_view.view_data()
        for name, df in view_data.items():
            # Pad the name to 15 characters
            name = (name + " " * 15)[:15]

            # Get the first three items in the first column
            examples = ", ".join(df.iloc[:, 0].tolist())
            if len(examples) > 70:
                examples = examples[:70] + "..."

            # Print the summary
            cprint(["lightpurple", "\t" + name, "lightgreen", str(df.shape[0]) + "  ", "purple_blue", examples])

    def incoming_data(self):
        return self.artifacts_text_view.incoming_data_summary()

    def glue_jobs(self):
        return self.artifacts_text_view.glue_jobs_summary()

    def data_sources(self):
        return self.artifacts_text_view.data_sources_summary()

    def feature_sets(self):
        return self.artifacts_text_view.feature_sets_summary()

    def models(self):
        return self.artifacts_text_view.models_summary()

    def endpoints(self):
        return self.artifacts_text_view.endpoints_summary()

    @staticmethod
    def log_debug():
        logging.getLogger("sageworks").setLevel(logging.DEBUG)

    @staticmethod
    def log_info():
        logging.getLogger("sageworks").setLevel(logging.INFO)

    @staticmethod
    def log_important():
        logging.getLogger("sageworks").setLevel(IMPORTANT_LEVEL_NUM)

    @staticmethod
    def log_warning():
        logging.getLogger("sageworks").setLevel(logging.WARNING)

    def broker_refresh(self):
        self.artifacts_text_view.refresh(force_refresh=True)

    @classmethod
    def status_lights(cls) -> list[(Token, str)]:
        """Check the status of AWS, Redis, and API Key and return Token colors

        Returns:
            list[(Token, str)]: A list of Token colors and status symbols
        """
        _status_lights = [(Token.Blue, "[")]

        # Check AWS Account
        # if 1 or cls.check_aws_account():
        _status_lights.append((Token.Green, "●"))

        # Check Redis
        # if 0 or cls.check_redis():
        _status_lights.append((Token.Green, "●"))

        # Check API Key
        # if 0 or cls.check_config():
        _status_lights.append((Token.Green, "●"))

        _status_lights.append((Token.Blue, "]"))

        return _status_lights

    def onboard(self):
        """Onboard a new user to SageWorks"""
        cprint("lightgreen", "Welcome to SageWorks!")
        cprint("lightblue", "Looks like this is your first time using SageWorks...")
        cprint("lightblue", "Let's get you set up...")

        # Create a Site Specific Config File
        self.cm.create_site_config()
        self.cm.platform_specific_instructions()

        # Tell the user to restart the shell
        cprint("lightblue", "After doing these instructions ^")
        cprint("lightblue", "Please rerun the SageWorks REPL to complete the onboarding process.")
        cprint("darkyellow", "Note: You'll need to start a NEW terminal to inherit the new ENV vars.")
        sys.exit(0)


# Launch Shell Entry Point
def launch_shell():
    shell = SageWorksShell()
    shell.start()


# Start the shell when running the script
if __name__ == "__main__":
    launch_shell()
