from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style
from prompt_toolkit.completion import Completer, Completion
from pprint import pprint

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
        # This dictionary will hold all the variables defined during the REPL session.
        self.session_globals = {}
        initial_imports = """
from sageworks.artifacts.data_sources.data_source import DataSource
from sageworks.artifacts.feature_sets.feature_set import FeatureSet
from sageworks.artifacts.models.model import Model
from sageworks.artifacts.endpoints.endpoint import Endpoint
        """
        exec(initial_imports, self.session_globals)

    def exit(self):
        print("Exiting SageWorks REPL...")
        return True  # Returning True will exit the REPL loop

    def help(self):
        print("Commands:")
        print("  - list <artifact_type>: List all the data_sources, feature_sets, etc")
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
                # Try evaluating it as an expression first
                result = eval(raw_text, self.session_globals, self.session_globals)
                pprint(result)  # This will print the result of the expression if eval succeeds
            except SyntaxError:
                # If it's not an expression (or if eval fails), it might be a statement.
                # For example, assignment (`foo = 5`) is a statement.
                try:
                    exec(raw_text, self.session_globals, self.session_globals)
                    if raw_text.startswith("print"):
                        pass  # Avoid printing None if it was a print statement
                    elif "=" in raw_text:
                        # If there was an assignment, print the assigned variable
                        left_hand_side = raw_text.split("=")[0].strip()
                        # Evaluate the left hand side to print its new value
                        print(eval(left_hand_side, self.session_globals, self.session_globals))
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
