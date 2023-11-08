from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style

# SageWorks Imports
from sageworks.views.artifacts_text_view import ArtifactsTextView

# Create a global instance of the ArtifactsTextView
artifacts_text_view = ArtifactsTextView()

# History instance to store the history of commands
history = InMemoryHistory()


# Command handler
class CommandHandler:
    def exit(self):
        print('Exiting SageWorks REPL...')
        return True  # Returning True will exit the REPL loop

    def help(self):
        print('Commands:')
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
        elif arg == 'data_sources':
            print(artifacts_text_view.data_sources_summary())
        elif arg == 'feature_sets':
            print(artifacts_text_view.feature_sets_summary())
        elif arg == 'models':
            print(artifacts_text_view.models_summary())
        if arg == 'endpoints':
            print(artifacts_text_view.endpoints_summary())

    def handle_command(self, raw_text):
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
            print(f'Unknown command: {command}')
            return False


# REPL loop
def repl():
    session = PromptSession(history=history)
    handler = CommandHandler()

    while True:
        try:
            # The 'prompt' parameter defines the text to display for the prompt
            text = session.prompt('🍺  SageWorks>', style=Style.from_dict({'prompt': 'hotpink'}))
            if handler.handle_command(text):
                break  # Exit the REPL loop if the command handler returns True
        except KeyboardInterrupt:
            # Handle Ctrl-C
            continue
        except EOFError:
            # Handle Ctrl-D
            break

    print('Goodbye!')


if __name__ == '__main__':
    repl()


