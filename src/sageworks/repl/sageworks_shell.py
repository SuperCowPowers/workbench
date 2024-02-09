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
from sageworks.utils.repl_utils import cprint, Spinner
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
        Token.Darkyellow: "#ddb777",
        Token.Orange: "#ff8700",
        Token.Red: "#dd0000",
        Token.Blue: "#4444d7",
        Token.Green: "#22cc22",
        Token.Yellow: "#ffd787",
        Token.Grey: "#aaaaaa",
    }


# Note: Hack so the Prompt Class can access these variables
aws_profile = ConfigManager().get_config("AWS_PROFILE")
sageworks_shell = None


class SageWorksPrompt(Prompts):
    """Custom SageWorks Prompt"""

    def in_prompt_tokens(self, cli=None):
        if sageworks_shell is None:
            lights = []
        else:
            lights = sageworks_shell.status_lights()
        aws_profile_prompt = [(Token.Blue, ":"), (Token.AWSProfile, f"{aws_profile}"), (Token.Blue, "> ")]
        return lights + [(Token.SageWorks, "SageWorks")] + aws_profile_prompt


class SageWorksShell:
    def __init__(self):
        # Give the SageWorks Version
        cprint("lightpurple", f"SageWorks Version: {importlib.import_module('sageworks').__version__}")

        # Check the SageWorks config
        self.cm = ConfigManager()
        if not self.cm.config_okay():
            # Invoke Onboarding Procedure
            cprint("yellow", "SageWorks Config incomplete...running onboarding procedure...")
            self.onboard()

        # Perform AWS connection test and other checks
        self.commands = dict()
        self.artifacts_text_view = None
        self.aws_status = self.check_aws_account()
        self.redis_status = self.check_redis()
        self.open_source_api_key = self.check_open_source_api_key()
        if self.aws_status:
            self.import_sageworks()

        # Set up the Prompt and the IPython shell
        config = InteractiveShellEmbed.instance().config
        config.TerminalInteractiveShell.autocall = 2
        config.TerminalInteractiveShell.prompts_class = SageWorksPrompt
        config.TerminalInteractiveShell.highlighting_style = CustomPromptStyle
        self.shell = InteractiveShellEmbed(config=config, banner1="", exit_msg="Goodbye from SageWorks!")

        # Register our custom commands
        self.commands["help"] = self.help
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
        self.commands["meta_refresh"] = self.meta_refresh
        self.commands["config"] = self.show_config
        self.commands["status"] = self.status_description

    def start(self):
        """Start the SageWorks IPython shell"""
        cprint("magenta", "Welcome to SageWorks!")
        if self.aws_status is False:
            cprint("red", "AWS Account Connection Failed...Review/Fix the SageWorks Config:")
            cprint("red", f"Path: {self.cm.site_config_path}")
            self.show_config()
        else:
            self.help()
            self.summary()

        # Start the REPL
        self.shell(local_ns=self.commands)

    def check_open_source_api_key(self) -> bool:
        """Check the current Configuration Status

        Returns:
            bool: True if Open Source API Key, False otherwise
        """
        config = self.cm.get_all_config()
        return config["API_KEY_INFO"]["license_id"] == "Open Source"

    def check_redis(self) -> str:
        """Check the Redis Cache

        Returns:
            str: The Redis status (either "OK", "FAIL", or "LOCAL")
        """
        from sageworks.utils.sageworks_cache import SageWorksCache

        # Grab the Redis Host and Port
        host = self.cm.get_config("REDIS_HOST", "localhost")
        port = self.cm.get_config("REDIS_PORT", 6379)

        # Check if Redis is running locally
        status = "OK"
        if host == "localhost":
            status = "LOCAL"

        # Open the Redis connection (class object)
        cprint("lime", f"Checking Redis connection to: {host}:{port}..")
        if SageWorksCache().check():
            cprint("lightgreen", "Redis Cache Check Success...")
        else:
            cprint("yellow", "Redis Cache Check Failed...check your SageWorks Config...")
            status = "FAIL"

        # Return the Redis status
        return status

    @staticmethod
    def check_aws_account() -> bool:
        """Check if the AWS Account is Set up Correctly

        Returns:
            bool: True if AWS Account is set up correctly, False otherwise
        """
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
                cprint("red", "AWS Account Check Failed: Check AWS_PROFILE and/or Renew SSO Token...")
                return False
        except botocore.exceptions.ProfileNotFound:
            cprint("red", "AWS Account Check Failed: Check AWS_PROFILE...")
            return False
        except botocore.exceptions.NoCredentialsError:
            cprint("red", "AWS Account Check Failed: Check AWS Credentials...")
            return False

        # Okay assume everything is good
        return True

    def show_config(self):
        """Show the current SageWorks Config"""
        cprint("yellow", "\nSageWorks Config:")
        cprint("lightblue", f"Path: {self.cm.site_config_path}")
        config = self.cm.get_all_config()
        for key, value in config.items():
            cprint(["lightpurple", "\t" + key, "lightgreen", value])

    def import_sageworks(self):
        # Import all the SageWorks modules
        spinner = self.spinner_start("Chatting with AWS:")
        try:
            self.artifacts_text_view = importlib.import_module(
                "sageworks.views.artifacts_text_view"
            ).ArtifactsTextView()
        finally:
            spinner.stop()

        # These are the classes we want to expose to the REPL
        self.commands["DataSource"] = importlib.import_module("sageworks.api.data_source").DataSource
        self.commands["FeatureSet"] = importlib.import_module("sageworks.api.feature_set").FeatureSet
        self.commands["Model"] = importlib.import_module("sageworks.api.model").Model
        self.commands["ModelType"] = importlib.import_module("sageworks.api.model").ModelType
        self.commands["Endpoint"] = importlib.import_module("sageworks.api.endpoint").Endpoint
        self.commands["Monitor"] = importlib.import_module("sageworks.api.monitor").Monitor
        self.commands["Meta"] = importlib.import_module("sageworks.api.meta").Meta
        self.commands["PluginManager"] = importlib.import_module("sageworks.utils.plugin_manager").PluginManager

        # These are 'nice to have' imports
        self.commands["pd"] = importlib.import_module("pandas")
        self.commands["pprint"] = importlib.import_module("pprint").pprint

    def help(self, *args):
        """Custom help command for the SageWorks REPL

        Args:
            *args: Arguments passed to the help command.
        """
        # If we have args forward to the built-in help function
        if args:
            help(*args)

        # Otherwise show the SageWorks help message
        else:
            cprint("lightblue", self.help_txt())

    @staticmethod
    def help_txt():
        help_msg = """    Commands:
        - help: Show this help message
        - docs: Open browser to show SageWorks Documentation
        - summary: Show a summary of all the AWS Artifacts
        - incoming_data: List all the incoming S3 data
        - glue_jobs: List all the Glue Jobs in AWS
        - data_sources: List all the DataSources in AWS
        - feature_sets: List all the FeatureSets in AWS
        - models: List all the Models in AWS
        - endpoints: List all the Endpoints in AWS
        - meta_refresh: Force a refresh of the AWS Metadata
        - config: Show the current SageWorks Config
        - status: Show the current SageWorks Status
        - exit: Exit SageWorks REPL"""
        return help_msg

    def spinner_start(self, text: str, color: str = "lightpurple") -> Spinner:
        # Import all the SageWorks modules
        spinner = Spinner(color, text)
        spinner.start()  # Start the spinner
        return spinner

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

    def meta_refresh(self):
        """Force a refresh of the AWS Metadata"""
        spinner = self.spinner_start("Refreshing AWS Metadata:")
        try:
            self.artifacts_text_view.refresh(force_refresh=True)
        finally:
            spinner.stop()

    def status_lights(self) -> list[(Token, str)]:
        """Check the status of AWS, Redis, and API Key and return Token colors

        Returns:
            list[(Token, str)]: A list of Token colors and status symbols
        """
        _status_lights = [(Token.Blue, "[")]

        # AWS Account Status
        if self.aws_status:
            _status_lights.append((Token.Green, "●"))
        else:
            _status_lights.append((Token.Red, "●"))

        # Redis Status
        if self.redis_status == "OK":
            _status_lights.append((Token.Green, "●"))
        elif self.redis_status == "LOCAL":
            _status_lights.append((Token.Blue, "●"))
        elif self.redis_status == "FAIL":
            _status_lights.append((Token.Orange, "●"))
        else:  # Unknown
            _status_lights.append((Token.Grey, "●"))

        # Check API Key
        if self.open_source_api_key:
            _status_lights.append((Token.Lightpurple, "●"))
        else:
            _status_lights.append((Token.Green, "●"))

        _status_lights.append((Token.Blue, "]"))

        return _status_lights

    def status_description(self):
        """Print a description of the status of AWS, Redis, and API Key"""

        # AWS Account
        if self.aws_status:
            cprint("lightgreen", "\t● AWS Account: OK")
        else:
            cprint("red", "\t● AWS Account: Failed to Connect")

        # Redis
        if self.redis_status == "OK":
            cprint("lightgreen", "\t● Redis: OK")
        elif self.redis_status == "LOCAL":
            cprint("lightblue", "\t● Redis: Local")
        elif self.redis_status == "FAIL":
            cprint("orange", "\t● Redis: Failed to Connect")

        # API Key
        if self.open_source_api_key:
            cprint("lightpurple", "\t● API Key: Open Source")
        else:
            cprint("lightgreen", "\t● API Key: Enterprise")

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
    global sageworks_shell
    sageworks_shell = SageWorksShell()
    sageworks_shell.start()


# Start the shell when running the script
if __name__ == "__main__":
    launch_shell()
