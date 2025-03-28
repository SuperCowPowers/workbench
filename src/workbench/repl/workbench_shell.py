import IPython
from IPython import start_ipython
from IPython.terminal.prompts import Prompts
from IPython.terminal.ipapp import load_default_config
from pygments.token import Token
import sys
import logging
import importlib
import botocore
import webbrowser
import pandas as pd
import readline  # noqa
from distutils.version import LooseVersion

try:
    import matplotlib.pyplot as plt  # noqa

    plt.ion()
    HAVE_MATPLOTLIB = True
except ImportError:
    HAVE_MATPLOTLIB = False

try:
    import plotly.io as pio  # noqa

    pio.renderers.default = "browser"
    HAVE_PLOTLY = True
except ImportError:
    HAVE_PLOTLY = False


# Workbench Imports
from workbench.utils.repl_utils import cprint, Spinner
from workbench.utils.workbench_logging import IMPORTANT_LEVEL_NUM, TRACE_LEVEL_NUM
from workbench.utils.config_manager import ConfigManager
from workbench.utils.log_utils import silence_logs, log_theme

# If we have RDKIT/Mordred let's pull in our cheminformatics utils
try:
    import rdkit  # noqa
    import mordred  # noqa
    from workbench.utils import chem_utils

    HAVE_CHEM_UTILS = True
except ImportError:
    HAVE_CHEM_UTILS = False


def onboard():
    """Onboard a new user to Workbench"""
    cprint("lightgreen", "Welcome to Workbench!")
    cprint("lightblue", "Looks like this is your first time using Workbench...")
    cprint("lightblue", "Let's get you set up...")

    # Create a Site Specific Config File
    cm = ConfigManager()
    cm.create_site_config()
    cm.platform_specific_instructions()

    # Tell the user to restart the shell
    cprint("lightblue", "After doing these instructions ^")
    cprint("lightblue", "Please rerun the Workbench REPL to complete the onboarding process.")
    cprint("darkyellow", "Note: You'll need to start a NEW terminal to inherit the new ENV vars.")
    sys.exit(0)


# Check config and onboard if necessary
if not ConfigManager().config_okay():
    onboard()

# Set the log level to important
# logging.getLogger("workbench").setLevel(IMPORTANT_LEVEL_NUM)


# We want to customize our prompt colors
prompt_styles = {
    Token.Workbench: "#af87ff",  # Light Purple color for Workbench
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
workbench_shell = None


class WorkbenchPrompt(Prompts):
    """Custom Workbench Prompt"""

    def in_prompt_tokens(self, cli=None):
        if workbench_shell is None:
            lights = []
        else:
            lights = workbench_shell.status_lights()
        aws_profile_prompt = [(Token.Blue, ":"), (Token.AWSProfile, f"{aws_profile}"), (Token.Blue, "> ")]
        return lights + [(Token.Workbench, "Workbench")] + aws_profile_prompt


class WorkbenchShell:
    def __init__(self):
        # Give the Workbench Version
        version = importlib.import_module("workbench").__version__
        cprint("lightpurple", f"Workbench Version: {version}")

        # Check the Workbench config
        self.cm = ConfigManager()
        if not self.cm.config_okay():
            # Invoke Onboarding Procedure
            onboard()

        # Our Metadata Object pull information from the Cloud Platform
        self.meta = None
        self.meta_status = "DIRECT"

        # Perform AWS connection test and other checks
        self.commands = dict()
        self.aws_status = self.check_aws_account()
        self.open_source_api_key = self.check_open_source_api_key()
        if self.aws_status:
            with silence_logs():
                self.import_workbench()

        # Try cached meta (if that fails it will be set to direct meta)
        self.try_cached_meta()

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
        self.commands["pipelines"] = self.pipelines
        self.commands["log_debug"] = self.log_debug
        self.commands["log_trace"] = self.log_trace
        self.commands["log_info"] = self.log_info
        self.commands["log_important"] = self.log_important
        self.commands["log_warning"] = self.log_warning
        self.commands["config"] = self.show_config
        self.commands["status"] = self.status_description
        self.commands["plot"] = self.plot_manager
        self.commands["log"] = logging.getLogger("workbench")
        self.commands["get_meta"] = self.get_meta
        self.commands["params"] = importlib.import_module("workbench.api.parameter_store").ParameterStore()
        self.commands["df_store"] = importlib.import_module("workbench.api.df_store").DFStore()
        self.commands["graph_store"] = importlib.import_module("workbench.api.graph_store").GraphStore()
        self.commands["version"] = lambda: print(version)
        self.commands["cached_meta"] = self.switch_to_cached_meta
        self.commands["direct_meta"] = self.switch_to_direct_meta
        self.commands["log_theme"] = log_theme
        self.commands["reconnect"] = self.check_aws_account

        # Add cheminformatics utils if available
        if HAVE_CHEM_UTILS:
            self.commands["show"] = chem_utils.show

    def start(self):
        """Start the Workbench IPython shell"""
        cprint("magenta", "\nWelcome to Workbench!")
        if self.aws_status is False:
            cprint("red", "AWS Account Connection Failed...Review/Fix the Workbench Config:")
            cprint("red", f"Path: {self.cm.site_config_path}")
            self.show_config()
        else:
            self.help()
            self.summary()

        # Load the default IPython configuration
        config = load_default_config()
        config.TerminalInteractiveShell.autocall = 2
        config.TerminalInteractiveShell.prompts_class = WorkbenchPrompt
        config.TerminalInteractiveShell.highlighting_style_overrides = prompt_styles
        config.TerminalInteractiveShell.banner1 = ""

        # Merge custom commands and globals into the namespace
        locs = self.commands.copy()  # Copy the custom commands
        locs.update(globals())  # Merge with global namespace

        # Start IPython with the config and commands in the namespace
        try:
            if LooseVersion(IPython.__version__) >= LooseVersion("9.0.0"):
                ipython_argv = ["--no-tip", "--theme", "linux"]
            else:
                ipython_argv = []
            start_ipython(ipython_argv, user_ns=locs, config=config)
        finally:
            spinner = self.spinner_start("Goodbye to AWS:")
            with silence_logs():
                self.meta.close()
            spinner.stop()
            cprint("lightgreen", "Goodbye from Workbench!\n")

    def check_open_source_api_key(self) -> bool:
        """Check the current Configuration Status

        Returns:
            bool: True if Open Source API Key, False otherwise
        """
        config = self.cm.get_all_config()
        return config["API_KEY_INFO"]["license_id"] == "Open Source"

    @staticmethod
    def check_aws_account() -> bool:
        """Check if the AWS Account is Set up Correctly

        Returns:
            bool: True if AWS Account is set up correctly, False otherwise
        """
        cprint("lightgreen", "Checking AWS Account Connection...")
        try:
            try:
                aws_clamp = importlib.import_module(
                    "workbench.core.cloud_platform.aws.aws_account_clamp"
                ).AWSAccountClamp()
                aws_clamp.check_aws_identity()
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
        """Show the current Workbench Config"""
        cprint("yellow", "\nWorkbench Config:")
        cprint("lightblue", f"Path: {self.cm.site_config_path}")
        config = self.cm.get_all_config()
        for key, value in config.items():
            cprint(["lightpurple", "\t" + key, "lightgreen", value])

    def import_workbench(self):
        # Import all the Workbench modules
        spinner = self.spinner_start("Importing Workbench:")
        try:
            # These are the classes we want to expose to the REPL
            self.commands["DataSource"] = importlib.import_module("workbench.api.data_source").DataSource
            self.commands["FeatureSet"] = importlib.import_module("workbench.api.feature_set").FeatureSet
            self.commands["Model"] = importlib.import_module("workbench.api.model").Model
            self.commands["CachedModel"] = importlib.import_module("workbench.cached.cached_model").CachedModel
            self.commands["ModelType"] = importlib.import_module("workbench.api.model").ModelType
            self.commands["Endpoint"] = importlib.import_module("workbench.api.endpoint").Endpoint
            self.commands["Monitor"] = importlib.import_module("workbench.api.monitor").Monitor
            self.commands["ParameterStore"] = importlib.import_module("workbench.api.parameter_store").ParameterStore
            self.commands["DFStore"] = importlib.import_module("workbench.api.df_store").DFStore
            self.commands["GraphStore"] = importlib.import_module("workbench.api.graph_store").GraphStore
            self.commands["PandasToFeatures"] = importlib.import_module(
                "workbench.core.transforms.pandas_transforms"
            ).PandasToFeatures
            self.commands["Meta"] = importlib.import_module("workbench.api").Meta
            self.commands["CachedMeta"] = importlib.import_module("workbench.cached.cached_meta").CachedMeta
            self.commands["View"] = importlib.import_module("workbench.core.views.view").View
            self.commands["DisplayView"] = importlib.import_module("workbench.core.views.display_view").DisplayView
            self.commands["TrainingView"] = importlib.import_module("workbench.core.views.training_view").TrainingView
            self.commands["ComputationView"] = importlib.import_module(
                "workbench.core.views.computation_view"
            ).ComputationView
            self.commands["InferenceView"] = importlib.import_module(
                "workbench.core.views.inference_view"
            ).InferenceView
            self.commands["PandasToView"] = importlib.import_module("workbench.core.views.pandas_to_view").PandasToView
            self.commands["Pipeline"] = importlib.import_module("workbench.api.pipeline").Pipeline

            # Algorithms
            self.commands["FSP"] = importlib.import_module(
                "workbench.algorithms.dataframe.feature_space_proximity"
            ).FeatureSpaceProximity

            # These are 'nice to have' imports
            self.commands["pd"] = importlib.import_module("pandas")
            self.commands["wr"] = importlib.import_module("awswrangler")
            self.commands["pprint"] = importlib.import_module("pprint").pprint
        finally:
            spinner.stop()

    def help(self, *args):
        """Custom help command for the Workbench REPL

        Args:
            *args: Arguments passed to the help command.
        """
        # If we have args forward to the built-in help function
        if args:
            help(*args)

        # Otherwise show the Workbench help message
        else:
            cprint("lightblue", self.help_txt())

    @staticmethod
    def help_txt():
        help_msg = """    Commands:
        - help: Show this help message
        - docs: Open browser to show Workbench Documentation
        - data_sources: List all the DataSources in AWS
        - feature_sets: List all the FeatureSets in AWS
        - models: List all the Models in AWS
        - endpoints: List all the Endpoints in AWS
        - config: Show the current Workbench Config
        - status: Show the current Workbench Status
        - log_(debug/info/important/warning): Set the Workbench log level
        - exit: Exit Workbench REPL"""
        return help_msg

    def spinner_start(self, text: str, color: str = "lightpurple") -> Spinner:
        # Import all the Workbench modules
        spinner = Spinner(color, text)
        spinner.start()  # Start the spinner
        return spinner

    @staticmethod
    def doc_browser():
        """Open a browser and start the Dash app and open a browser."""
        url = "https://supercowpowers.github.io/workbench/"
        webbrowser.open(url)

    def summary(self):
        """Show a summary of all the AWS Artifacts"""

        # Grab information about all the AWS Artifacts
        spinner = self.spinner_start("Chatting with AWS:")
        try:
            # We're filling in Summary Data for all the AWS Services
            summary_data = {
                "INCOMING_DATA": self.meta.incoming_data(),
                "ETL_JOBS": self.meta.etl_jobs(),
                "DATA_SOURCES": self.meta.data_sources(),
                "FEATURE_SETS": self.meta.feature_sets(),
                "MODELS": self.meta.models(),
                "ENDPOINTS": self.meta.endpoints(),
            }
        finally:
            spinner.stop()

        # Print out the AWS Artifacts Summary
        cprint("yellow", "\nAWS Artifacts Summary:")
        for name, df in summary_data.items():
            # Pad the name to 15 characters
            name = (name + " " * 15)[:15]

            # Sanity check the dataframe
            if df.empty:
                examples = ""

            # Get the first three items in the first column
            else:
                examples = ", ".join(df.iloc[:, 0].tolist())
                if len(examples) > 70:
                    examples = examples[:70] + "..."

            # Print the summary
            cprint(["lightpurple", "\t" + name, "lightgreen", str(df.shape[0]) + "  ", "purple_blue", examples])

    def incoming_data(self):
        return self.meta.incoming_data()

    def glue_jobs(self):
        return self.meta.etl_jobs()

    def data_sources(self):
        return self.meta.data_sources()

    def feature_sets(self, details: bool = False):
        return self.meta.feature_sets(details=details)

    def models(self, details: bool = False):
        return self.meta.models(details=details)

    def endpoints(self):
        return self.meta.endpoints()

    def pipelines(self):
        return self.meta.pipelines()

    @staticmethod
    def log_debug():
        logging.getLogger("workbench").setLevel(logging.DEBUG)

    @staticmethod
    def log_trace():
        logging.getLogger("workbench").setLevel(TRACE_LEVEL_NUM)

    @staticmethod
    def log_info():
        logging.getLogger("workbench").setLevel(logging.INFO)

    @staticmethod
    def log_important():
        logging.getLogger("workbench").setLevel(IMPORTANT_LEVEL_NUM)

    @staticmethod
    def log_warning():
        logging.getLogger("workbench").setLevel(logging.WARNING)

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

        # Cached Meta Status
        if self.meta_status == "CACHED":
            _status_lights.append((Token.Green, "●"))
        elif self.meta_status == "DIRECT":
            _status_lights.append((Token.Blue, "●"))
        elif self.meta_status == "FAIL":
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
        if self.meta_status == "CACHED":
            cprint("lightgreen", "\t● Meta: Cached")
        elif self.meta_status == "DIRECT":
            cprint("lightblue", "\t● Meta: Direct")
        elif self.meta_status == "FAIL":
            cprint("orange", "\t● Meta: Failed to Connect")

        # API Key
        if self.open_source_api_key:
            cprint("lightpurple", "\t● API Key: Open Source")
        else:
            cprint("lightgreen", "\t● API Key: Enterprise")

    # Helpers method to switch from direct Meta to Cached Meta
    def try_cached_meta(self):
        from workbench.api import Meta
        from workbench.cached.cached_meta import CachedMeta

        with silence_logs():
            self.meta = CachedMeta()
        if self.meta.check():
            self.meta_status = "CACHED"
            cprint("lightblue", "Using Cached Meta...")
        else:
            self.meta_status = "DIRECT"
            cprint("darkyellow", "Using Direct Meta [slower]...")
            with silence_logs():
                self.meta.close()
                self.meta = Meta()

    def switch_to_cached_meta(self):
        from workbench.api import Meta
        from workbench.cached.cached_meta import CachedMeta

        self.meta = CachedMeta()
        if self.meta.check():
            self.meta_status = "CACHED"
            cprint("lightblue", "Switched to Cached Meta...")
        else:
            self.meta.close()
            self.meta_status = "FAIL"
            cprint("orange", "Failed to Switch to Cached Meta...")
            cprint("darkyellow", "Using Direct Meta [slower]...")
            self.meta = Meta()

    def switch_to_direct_meta(self):
        from workbench.api import Meta

        # Close the current Meta object
        if self.meta:
            self.meta.close()
        # Create a new direct Meta object
        self.meta = Meta()
        self.meta_status = "DIRECT"
        cprint("darkyellow", "Switched to Direct Meta...")

    def get_meta(self):
        return self.meta

    def plot_manager(self, data, plot_type: str = "table", **kwargs):
        """Plot Manager for Workbench"""
        from workbench.web_interface.components.plugins import ag_table, graph_plot, scatter_plot

        if plot_type == "table":
            # Check what type of data we have
            if isinstance(data, pd.DataFrame):
                self.plot_plugin(ag_table.AGTable, data, **kwargs)
            else:
                # Does this object have a pull_dataframe() method?
                if hasattr(data, "pull_dataframe"):
                    data = data.pull_dataframe(limit=100)
                    self.plot_plugin(ag_table.AGTable, data, **kwargs)
                else:
                    cprint("yellow", f"Unknown Data Type for Table Plot '{type(data)}'")
        elif plot_type == "graph":
            self.plot_plugin(graph_plot.GraphPlot, data, **kwargs)
        elif plot_type == "scatter":
            self.plot_plugin(scatter_plot.ScatterPlot, data, **kwargs)
        else:
            cprint("yellow", f"Unknown Plot Type '{plot_type}'")

    @staticmethod
    def plot_plugin(plugin_class, data=None, **kwargs):
        """Plot data using a plugin.

        Args:
            plugin_class (PluginInterface): The plugin class to use.
            input_data (Optional): Optional input data (e.g., DataSource, FeatureSet, etc.)
            **kwargs: Additional keyword arguments for plugin properties.
                theme (str): The theme to use for the Dash app (default: "dark")
        """
        from workbench.web_interface.components.plugin_unit_test import PluginUnitTest

        # Get kwargs
        theme = kwargs.get("theme", "dark")

        plugin_test = PluginUnitTest(plugin_class, theme=theme, input_data=data, **kwargs)

        # Run the server and open in the browser
        plugin_test.run()
        url = f"http://127.0.0.1:{plugin_test.port}"
        webbrowser.open(url)


# Launch Shell Entry Point
def launch_shell():
    global workbench_shell
    workbench_shell = WorkbenchShell()
    workbench_shell.start()


# Start the shell when running the script
if __name__ == "__main__":
    launch_shell()
