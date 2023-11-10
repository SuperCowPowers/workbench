from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style
from prompt_toolkit.completion import Completer, Completion
from pprint import pprint
import pandas as pd

# SageWorks Imports
from sageworks.views.artifacts_text_view import ArtifactsTextView
from sageworks.utils.sageworks_logging import IMPORTANT_LEVEL_NUM
import logging

# SageWorks is currently quite verbose so lets set the logs to warning
logging.getLogger("sageworks").setLevel(IMPORTANT_LEVEL_NUM)

# Create a global instance of the ArtifactsTextView
artifacts_text_view = ArtifactsTextView()

# History instance to store the history of commands
history = InMemoryHistory()

# Setup Pandas output options
pd.set_option("display.max_colwidth", 45)
pd.set_option("display.width", 1000)


class SageWorksCompleter(Completer):
    def __init__(self, globals):
        self.globals = globals

    def get_completions(self, document, complete_event):
        word_before_cursor = document.get_word_before_cursor()

        for name in self.globals:
            if name.startswith(word_before_cursor):
                yield Completion(name, start_position=-len(word_before_cursor))


# Command handler
class CommandHandler:
    def __init__(self):
        # This dictionary will hold all the variables and functions defined during the REPL session.
        self.session_globals = globals().copy()  # Copy the current global namespace

        # Add the necessary imports to the session globals
        self.preload_imports()

        # Add custom helper functions to the session globals
        self.session_globals["get_data_sources"] = self.get_data_sources
        self.session_globals["get_feature_sets"] = self.get_feature_sets
        self.session_globals["get_models"] = self.get_models
        self.session_globals["get_endpoints"] = self.get_endpoints

    def preload_imports(self):
        from sageworks.artifacts.data_sources.data_source import DataSource
        from sageworks.artifacts.feature_sets.feature_set import FeatureSet
        from sageworks.artifacts.models.model import Model
        from sageworks.artifacts.endpoints.endpoint import Endpoint

        self.session_globals["DataSource"] = DataSource
        self.session_globals["FeatureSet"] = FeatureSet
        self.session_globals["Model"] = Model
        self.session_globals["Endpoint"] = Endpoint

    @staticmethod
    def get_data_sources():
        """Get a dataframe of all data sources"""
        return artifacts_text_view.data_sources_summary()

    @staticmethod
    def get_feature_sets():
        """Get a dataframe of all feature sets"""
        return artifacts_text_view.feature_sets_summary()

    @staticmethod
    def get_models():
        """Get a dataframe of all models"""
        return artifacts_text_view.models_summary()

    @staticmethod
    def get_endpoints():
        """Get a dataframe of all endpoints"""
        return artifacts_text_view.endpoints_summary()

    def exit(self):
        print("Exiting SageWorks REPL...")
        return True  # Returning True will exit the REPL loop

    def help(self):
        print("Commands:")
        print("  - list <artifact_type>: List all the data_sources, feature_sets, etc")
        print("  - get_data_sources(): Get a dataframe of the data_sources")
        print("  - get_feature_sets(): Get a dataframe of the feature_sets")
        print("  - get_models(): Get a dataframe of the models")
        print("  - get_endpoints(): Get a dataframe of the endpoints")
        print("  - exit: Exit SageWorks REPL")

    def list(self, arg=None):
        if arg is None:
            print("Please specify what you'd like to list. Options are:")
            print("  - data_sources: List all data sources")
            print("  - feature_sets: List all feature sets")
            print("  - models: List all models")
            print("  - endpoints: List all endpoints")
            return
        elif arg == "data_sources":
            print(artifacts_text_view.data_sources_summary())
        elif arg == "feature_sets":
            print(artifacts_text_view.feature_sets_summary())
        elif arg == "models":
            print(artifacts_text_view.models_summary())
        if arg == "endpoints":
            print(artifacts_text_view.endpoints_summary())

    def get_data_sources(self):
        """Get a dataframe of all data sources"""
        return artifacts_text_view.data_sources_summary()

    def handle_command(self, raw_text):
        # Check for custom commands first
        tokens = raw_text.strip().split()
        if not tokens:
            return False  # No command entered

        command = tokens[0].lower()
        args = tokens[1:]

        if hasattr(self, command):
            method = getattr(self, command)
            result = method(*args)
            return result
        else:
            # If not a custom command, try to execute it as Python code
            try:
                # Attempt to evaluate it as an expression
                result = eval(raw_text, self.session_globals, self.session_globals)
                pprint(result)  # This will print the result of the expression
            except SyntaxError:
                # If eval fails, it could be a statement (like an assignment).
                try:
                    exec(raw_text, self.session_globals, self.session_globals)
                except Exception as e:
                    print(f"Python execution error: {e}")
            except Exception as e:
                # Catch all other exceptions that might be raised by eval
                print(f"Python execution error: {e}")
            return False


# REPL loop
def repl():
    handler = CommandHandler()
    completer = SageWorksCompleter(handler.session_globals)  # Use the updated globals for the completer
    session = PromptSession(completer=completer, history=history)

    while True:
        try:
            # The 'prompt' parameter defines the text to display for the prompt
            text = session.prompt("🍺  SageWorks> ", style=Style.from_dict({"prompt": "hotpink"}))
            if handler.handle_command(text):
                break  # Exit the REPL loop if the command handler returns True
        except KeyboardInterrupt:
            # Handle Ctrl-C
            continue
        except EOFError:
            # Handle Ctrl-D
            break

    print("Goodbye!")


if __name__ == "__main__":
    repl()
