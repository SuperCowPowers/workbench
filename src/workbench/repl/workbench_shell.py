# flake8: noqa: E402
import os
import sys
import logging
import importlib
import webbrowser

importlib.import_module("readline")  # side effect: enables line editing/history

# Disable OpenMP parallelism to avoid segfaults with PyTorch in iPython
# This is a known issue on macOS where libomp crashes during thread synchronization
# Must be set before importing numpy/pandas/torch or any library that uses OpenMP
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import IPython
from IPython import start_ipython
from IPython.terminal.prompts import Prompts
from IPython.terminal.ipapp import load_default_config
from distutils.version import LooseVersion
from pygments.token import Token
import botocore

# Enable matplotlib's interactive mode so user plots show non-blocking.
import matplotlib.pyplot as plt  # noqa: F401

plt.ion()

# Route plotly figures to the browser when the user plots in the REPL.
# plotly is an optional dep ([ui] extra) — REPL works fine without it.
try:
    import plotly.io as pio

    pio.renderers.default = "browser"
except ImportError:
    # plotly is optional ([ui] extra)
    pass


# Workbench Imports
from workbench.local.local_meta import LocalMeta
from workbench.utils.repl_utils import cprint, Spinner
from workbench.utils.repl_themes import prompt_styles, current_theme, token_color
from workbench.utils.cow_puns import random_cow_pun
from workbench.utils.contest_utils import contest_summary
from workbench.utils.workbench_logging import IMPORTANT_LEVEL_NUM, TRACE_LEVEL_NUM
from workbench.utils.config_manager import ConfigManager, FatalConfigError

# Where local-mode users go to set up their own AWS account
SCP_URL = "https://www.supercowpowers.com/"
from workbench.utils.log_utils import silence_logs
from workbench.utils.aws_utils import sso_login_hint
from workbench.utils.web_utils import EGRESS_MODE
from workbench.utils.chem_utils import vis

# Egress dots
EGRESS_LIGHTS = {"off": Token.Lightgreen, "guarded": Token.Blue, "full": Token.Darkyellow}


def aws_setup():
    """Set up this user's AWS account for Workbench"""
    cprint("lightgreen", "Welcome to Workbench!")
    cprint("lightblue", "Looks like this is your first time using Workbench...")
    cprint("lightblue", "Let's get you set up...")

    # Create a Site Specific Config File
    cm = ConfigManager()
    cm.create_site_config()
    cm.platform_specific_instructions()

    # Tell the user to restart the shell
    cprint("lightblue", "After doing these instructions ^")
    cprint("lightblue", "Please rerun the Workbench REPL to finish connecting your account.")
    cprint("darkyellow", "Note: You'll need to start a NEW terminal to inherit the new ENV vars.")
    sys.exit(0)


# Set the log level to important
log = logging.getLogger("workbench")
log.setLevel(IMPORTANT_LEVEL_NUM)
log.addFilter(
    lambda record: not (
        record.getMessage().startswith("Async: Metadata") or record.getMessage().startswith("Updated Metadata")
    )
)


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
        aws_profile_prompt = [(Token.Blue, ":"), (Token.Darkyellow, f"{aws_profile}")]
        prompt = lights + [(Token.Workbench, "Workbench")] + aws_profile_prompt
        if workbench_shell is not None and workbench_shell.bedrock_status:
            prompt += [(Token.Blue, ":"), (Token.Lightgreen, "Bosco")]
        return prompt + [(Token.Blue, "> ")]


class WorkbenchShell:
    def __init__(self):
        # Give the Workbench Version
        version = importlib.import_module("workbench").__version__
        cprint("lightpurple", f"Workbench Version: {version}")

        # Check the Workbench config. No config at all means the user never set AWS up,
        # which is a supported mode -- local artifacts and PublicData need none. A config
        # that exists but is incomplete means they meant to connect, so run setup.
        self.cm = ConfigManager()
        self.local_only = False
        if not self.cm.config_okay():
            if self.cm.using_default_config:
                self.local_only = True
            else:
                aws_setup()

        if not self.local_only:
            # Show which role this session is running as
            role = self.cm.get_config("WORKBENCH_ROLE")
            cprint("lightpurple", f"Workbench Role: {role}")

        # Our Metadata Object pull information from the Cloud Platform
        self.meta = None
        self.meta_status = "DIRECT"
        self.bedrock_status = False  # set once AWS is confirmed (Bosco needs Bedrock)

        # Bosco is opt-in per account: only wired up when ENABLE_BOSCO is truthy in the config.
        self.bosco_enabled = str(self.cm.get_config("ENABLE_BOSCO", False)).strip().lower() in ("true", "1", "yes")

        # Perform AWS connection test and other checks
        self.commands = dict()
        self.aws_status = False if self.local_only else self.check_aws_account()
        if self.aws_status:
            with silence_logs():
                self.import_workbench()

            # Try cached meta (if that fails it will be set to direct meta)
            self.try_cached_meta()

        # Register our custom commands
        self.commands["help"] = self.help
        self.commands["docs"] = self.doc_browser
        self.commands["summary"] = self.summary
        self.commands["local_summary"] = self.local_summary

        # Local artifacts and the model enums need no AWS, so they're registered
        # whether or not the account check passed -- a broken config is when the
        # local classes are most useful
        local = importlib.import_module("workbench.local")
        local_names = ["LocalDataSource", "LocalFeatureSet", "LocalModel", "LocalEndpoint", "LocalMeta"]
        for class_name in local_names + ["ModelType", "ModelFramework"]:
            self.commands[class_name] = getattr(local, class_name)
        self.commands["contests"] = self.contests
        self.commands["incoming_data"] = self.incoming_data
        self.commands["glue_jobs"] = self.glue_jobs
        self.commands["batch_jobs"] = importlib.import_module("workbench.utils.batch_utils").batch_jobs
        self.commands["data_sources"] = self.data_sources
        self.commands["feature_sets"] = self.feature_sets
        self.commands["models"] = self.models
        self.commands["endpoints"] = self.endpoints
        self.commands["log_debug"] = self.log_debug
        self.commands["log_trace"] = self.log_trace
        self.commands["log_info"] = self.log_info
        self.commands["log_important"] = self.log_important
        self.commands["log_warning"] = self.log_warning
        self.commands["config"] = self.show_config
        self.commands["aws_setup"] = aws_setup
        self.commands["status"] = self.status_description
        self.commands["log"] = logging.getLogger("workbench")
        self.commands["get_meta"] = self.get_meta
        # These build AWS clients as they're registered, so they exist only when AWS does
        if self.aws_status:
            self.commands["params"] = importlib.import_module("workbench.api.parameter_store").ParameterStore()
            self.commands["secrets"] = importlib.import_module(
                "workbench.core.cloud_platform.aws.aws_secrets_manager"
            ).AWSSecretsManager()
            self.commands["df_store"] = importlib.import_module("workbench.api.df_store").DFStore()
            self.commands["inf_store"] = importlib.import_module("workbench.api.inference_store").InferenceStore()
            self.commands["graph_store"] = importlib.import_module("workbench.api.graph_store").GraphStore()
        self.commands["version"] = lambda: print(version)
        self.commands["cached_meta"] = self.switch_to_cached_meta
        self.commands["direct_meta"] = self.switch_to_direct_meta
        self.commands["theme"] = importlib.import_module("workbench.utils.repl_themes").set_theme
        self.commands["reconnect"] = self.check_aws_account
        self.commands["pub_data"] = importlib.import_module("workbench.public_data").PublicData()
        self.commands["web_get"] = importlib.import_module("workbench.utils.web_utils").web_get
        # Bosco is opt-in (ENABLE_BOSCO); when off, the agent, prompt tag, and router stay dark
        if self.bosco_enabled:
            self.commands["bosco"] = importlib.import_module("workbench.agent.bosco").bosco
            bosco_utils = importlib.import_module("workbench.utils.bosco_utils")
            self.commands["show_session"] = bosco_utils.show_session
            self.commands["recent_sessions"] = bosco_utils.recent_sessions

            # Bosco needs Bedrock; light the prompt tag only when it's actually reachable
            if self.aws_status:
                with silence_logs():
                    self.bedrock_status = importlib.import_module("workbench.utils.bedrock_utils").bedrock_available()

        self.commands["show"] = vis.show

    def start(self):
        """Start the Workbench IPython shell"""
        cprint("magenta", "\nWelcome to Workbench!")
        if self.local_only:
            self.cow_pun()
            self.local_summary()
            cprint("lightpurple", "Local Mode: no AWS account, everything runs on this machine.")
            cprint("lightpurple", "  pub_data.list()                public datasets, no credentials needed")
            cprint("lightpurple", "  LocalDataSource(df, name=...)  -> to_features() -> to_model()")
            cprint("lightpurple", "\nAlready have an AWS account? Run aws_setup() to connect it.")
            cprint("lightpurple", f"Need one set up? {SCP_URL}\n")
        elif not self.aws_status:
            cprint("red", "AWS Account Connection Failed...Review/Fix the Workbench Config:")
            cprint("red", f"Path: {self.cm.site_config_path}")
            self.show_config()
        else:
            self.cow_pun()
            self.local_summary()
            self.summary()
            self.contests()
            if self.bosco_enabled:
                cprint("lightpurple", "\n🐶  New to Workbench? Ask me to walk you through building your first model.")

        # Load the default IPython configuration
        config = load_default_config()
        config.TerminalInteractiveShell.autocall = 2
        config.TerminalInteractiveShell.prompts_class = WorkbenchPrompt
        config.TerminalInteractiveShell.highlighting_style_overrides = prompt_styles
        config.TerminalInteractiveShell.banner1 = ""

        # Wire up the shell once it exists: job lights on the right prompt, and
        # the `bosco <text>` line router when Bosco is enabled.
        exec_lines = ["from workbench.utils.job_tracker import install_job_lights; install_job_lights()"]
        if self.bosco_enabled:
            exec_lines.append("from workbench.agent.bosco import register; register()")
        config.InteractiveShellApp.exec_lines = exec_lines

        # Merge custom commands and globals into the namespace
        locs = self.commands.copy()  # Copy the custom commands
        locs.update(globals())  # Merge with global namespace

        # Start IPython with the config and commands in the namespace
        try:
            if LooseVersion(IPython.__version__) >= LooseVersion("9.0.0"):
                # IPython's own theme colors the text being typed, so it has to
                # follow REPL_THEME or the input is unreadable on that background.
                ipython_theme = "lightbg" if current_theme() == "light" else "linux"
                ipython_argv = ["--no-tip", "--theme", ipython_theme]
            else:
                ipython_argv = []
            start_ipython(ipython_argv, user_ns=locs, config=config)
        finally:
            spinner = self.spinner_start("Goodbye to AWS:")
            with silence_logs():
                self.meta.close()
            spinner.stop()
            cprint("lightgreen", "Goodbye from Workbench!\n")

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
                cprint("red", f"AWS Account Check Failed: renew with '{sso_login_hint(aws_profile)}'")
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
        spinner = self.spinner_start("Spinning up Workbench:")
        try:
            # These are the classes we want to expose to the REPL
            self.commands["DataSource"] = importlib.import_module("workbench.api.data_source").DataSource
            self.commands["FeatureSet"] = importlib.import_module("workbench.api.feature_set").FeatureSet
            self.commands["Model"] = importlib.import_module("workbench.api.model").Model
            self.commands["CachedModel"] = importlib.import_module("workbench.cached.cached_model").CachedModel
            self.commands["Endpoint"] = importlib.import_module("workbench.api.endpoint").Endpoint
            self.commands["MetaEndpoint"] = importlib.import_module("workbench.api.meta_endpoint").MetaEndpoint
            self.commands["Monitor"] = importlib.import_module("workbench.api.monitor").Monitor
            self.commands["ParameterStore"] = importlib.import_module("workbench.api.parameter_store").ParameterStore
            self.commands["DFStore"] = importlib.import_module("workbench.api.df_store").DFStore
            self.commands["GraphStore"] = importlib.import_module("workbench.api.graph_store").GraphStore
            self.commands["PandasToFeatures"] = importlib.import_module(
                "workbench.core.transforms.pandas_transforms"
            ).PandasToFeatures
            self.commands["Meta"] = importlib.import_module("workbench.api").Meta
            self.commands["CachedMeta"] = importlib.import_module("workbench.cached.cached_meta").CachedMeta
            self.commands["InferenceCache"] = importlib.import_module("workbench.api.inference_cache").InferenceCache
            self.commands["View"] = importlib.import_module("workbench.core.views.view").View
            self.commands["DisplayView"] = importlib.import_module("workbench.core.views.display_view").DisplayView
            self.commands["ComputationView"] = importlib.import_module(
                "workbench.core.views.computation_view"
            ).ComputationView
            self.commands["InferenceView"] = importlib.import_module(
                "workbench.core.views.inference_view"
            ).InferenceView
            self.commands["PandasToView"] = importlib.import_module("workbench.core.views.pandas_to_view").PandasToView

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

    def help_txt(self):
        help_msg = """    Commands:
        - help: Show this help message
        - docs: Open browser to show Workbench Documentation
        - data_sources: List all the DataSources in AWS
        - feature_sets: List all the FeatureSets in AWS
        - models: List all the Models in AWS
        - endpoints: List all the Endpoints in AWS
        - local_summary: List the Local artifacts on this machine
        - config: Show the current Workbench Config
        - status: Show the current Workbench Status
        - log_(debug/info/important/warning): Set the Workbench log level
        - exit: Exit Workbench REPL"""

        # Bosco is only usable when Bedrock is reachable
        if self.bedrock_status:
            help_msg += """

    Bosco (ML Agent):
        - Just ask: anything that isn't valid Python routes to Bosco
              what pxr models do we have?
        - bosco <text>: Force a question when the text IS valid Python
        - ?Model / ??Model: Object help (a trailing ? is a question for Bosco)
        - Multi-line: Shift+Enter for a new line (see docs to map it); Enter sends
        - "show code" / "hide code": Toggle echoing the code Bosco runs
        - Ctrl-C: Interrupt Bosco (the conversation stays usable)
        - "how do I use you": Bosco explains the rest"""
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

    def local_summary(self):
        """Show a summary of the Local Artifacts, if there are any"""
        local_meta = LocalMeta()
        summary_data = {
            "DATA_SOURCES": local_meta.data_sources(),
            "FEATURE_SETS": local_meta.feature_sets(),
            "MODELS": local_meta.models(),
            "ENDPOINTS": local_meta.endpoints(),
        }
        if all(df.empty for df in summary_data.values()):
            return
        print()
        self._print_summary("Local Artifacts Summary:", summary_data)

    def summary(self):
        """Show a summary of all the AWS Artifacts"""

        # Grab information about all the AWS Artifacts
        print()
        spinner = self.spinner_start("Chatting with AWS:")
        try:
            # We're filling in Summary Data for all the AWS Services
            summary_data = {
                "ETL_JOBS": self.meta.etl_jobs(),
                "DATA_SOURCES": self.meta.data_sources(),
                "FEATURE_SETS": self.meta.feature_sets(),
                "MODELS": self.meta.models(),
                "ENDPOINTS": self.meta.endpoints(),
            }
        finally:
            spinner.stop()

        self._print_summary("AWS Artifacts Summary:", summary_data)

    @staticmethod
    def _print_summary(title: str, summary_data: dict):
        """Print an artifact summary: one row per type with a count and examples

        Args:
            title (str): The heading for this summary
            summary_data (dict): Artifact type -> DataFrame, name in the first column
        """
        cprint("yellow", title)
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
            cprint(["lightpurple", "\t" + name, "lightgreen", str(df.shape[0]) + "  ", "darkblue", examples])

    def cow_pun(self):
        """Print a random cow pun -- the REPL greeting's opening moo."""
        question, punchline = random_cow_pun()
        cprint("lightpurple", f"\n🐄  {question}")
        cprint("darkblue", f"       {punchline}")

    def contests(self):
        """Show the champion/challenger contests, most recent first."""
        spinner = self.spinner_start("Checking contests:")
        try:
            rows = contest_summary()
        finally:
            spinner.stop()

        if not rows:
            return  # nothing to show; stay quiet at startup

        # Recent first (by day), contested above settled within a day, newest last.
        # Customers can have 30+ contests, so cap the greeting at the top 10.
        dated = [r for r in rows if r["timestamp"] is not None]
        undated = [r for r in rows if r["timestamp"] is None]
        dated.sort(key=lambda r: (r["timestamp"].date(), r["contested"], r["timestamp"]), reverse=True)
        rows = (dated + undated)[:10]

        cprint("yellow", "\nContests:")
        for row in rows:
            name = (row["contest"] + " " * 24)[:24]
            flag = "contested" if row["contested"] else "stable"
            flag_color = "lightgreen" if row["contested"] else "darkblue"
            when = row["timestamp"].strftime("%Y-%m-%d") if row["timestamp"] is not None else ""
            segments = [
                "lightpurple",
                "\t" + name,
                flag_color,
                (flag + " " * 10)[:10],
                "tan",
                f" ({row['challengers']} challengers)  {when}",
            ]
            if row.get("recent_change"):
                segments += ["lightblue", "  ★ recent change"]
            cprint(segments)

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

    def status_rows(self) -> list[(Token, str)]:
        """AWS, Redis, and egress state as (Token color, label) pairs.

        The one place that decides what color each state is; the prompt dots and
        the `status` listing both render from here.

        Returns:
            list[(Token, str)]: A Token color and label per status row
        """
        if self.local_only:
            aws = (Token.Darkyellow, "AWS Account: Local Mode")
        elif self.aws_status:
            aws = (Token.Lightgreen, "AWS Account: OK")
        else:
            aws = (Token.Red, "AWS Account: Failed to Connect")
        cached = self.meta_status == "CACHED"
        redis = (Token.Lightgreen, "Redis: Connected") if cached else (Token.Darkyellow, "Redis: Not Connected")
        return [aws, redis, (EGRESS_LIGHTS[EGRESS_MODE], f"Egress: {EGRESS_MODE}")]

    def status_lights(self) -> list[(Token, str)]:
        """The bracketed status dots shown in the prompt.

        Returns:
            list[(Token, str)]: A list of Token colors and status symbols
        """
        dots = [(token, "●") for token, _ in self.status_rows()]
        return [(Token.Blue, "[")] + dots + [(Token.Blue, "]")]

    def status_description(self):
        """Print a description of the status of AWS, Redis, and egress"""
        for token, label in self.status_rows():
            cprint(token_color(token), f"\t● {label}")

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


# Launch Shell Entry Point
def launch_shell():
    global workbench_shell
    try:
        workbench_shell = WorkbenchShell()
    except FatalConfigError:
        aws_setup()
        return
    workbench_shell.start()


# Start the shell when running the script
if __name__ == "__main__":
    launch_shell()
