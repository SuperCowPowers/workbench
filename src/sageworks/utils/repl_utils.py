"""Repl utilities for Sageworks"""

import threading
import itertools
import time
import sys

# Colors
colors = {
    "lightblue": "\x1b[38;5;69m",
    "lightpurple": "\x1b[38;5;141m",
    "lightgreen": "\x1b[38;5;113m",
    "lime": "\x1b[38;5;154m",
    "darkyellow": "\x1b[38;5;220m",
    "orange": "\x1b[38;5;208m",
    "red": "\x1b[38;5;198m",
    "pink": "\x1b[38;5;213m",
    "magenta": "\x1b[38;5;206m",
    "tan": "\x1b[38;5;179m",
    "lighttan": "\x1b[38;5;180m",
    "yellow": "\x1b[38;5;226m",
    "green": "\x1b[38;5;34m",
    "blue": "\x1b[38;5;21m",
    "purple": "\x1b[38;5;91m",
    "purple_blue": "\x1b[38;5;63m",
    "lightgrey": "\x1b[38;5;250m",
    "grey": "\x1b[38;5;244m",
    "darkgrey": "\x1b[38;5;240m",
    "reset": "\x1b[0m",
}


def cprint(*args):
    """
    Print text in color. Supports either a single color and text or a list of color-text pairs.

    Args:
         A single color and text or a list of color-text pairs.
         For example: cprint('red', 'Hello') or cprint(['red', 'Hello', 'green', 'World'])
    """
    # If the first argument is a list, use it as the list of color-text pairs
    if isinstance(args[0], list):
        args = args[0]

    # Iterate over the arguments in pairs
    for i in range(0, len(args), 2):
        color = args[i]
        text = args[i + 1]
        print(f"{colors[color]}{text}{colors['reset']}", end=" ")

    print()  # Newline at the end


def status_lights(status_colors: list[str]):
    """
    Status lights (circles) in color

    Args:
         status_colors: A list of status colors (e.g. ['red', 'green', 'yellow'])
    """
    circle = "●"
    lights_str = ""
    for color in status_colors:
        lights_str += f"{colors[color]}{circle}{colors['reset']}"
    return lights_str


# Spinner implementation
class Spinner:
    def __init__(self, color, message="Loading..."):
        self.done = False
        self.message = message
        self.color = color

    @staticmethod
    def _hide_cursor():
        """Hide the terminal cursor."""
        sys.stdout.write("\033[?25l")
        sys.stdout.flush()

    @staticmethod
    def _show_cursor():
        """Show the terminal cursor."""
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()

    def spin(self):
        frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"  # Spinner frames
        # frames = "⠧⠩⠪⠫⠭⠮⠯⠳⠵⠷⠹⠺⠻⠼⠽⠾"  # Spinner frames
        done_frame = "⠿⠿⠿⠿"  # Done frame

        spinners = [itertools.cycle(frames) for _ in range(4)]  # Create four separate cycle iterators

        # Initialize each spinner to a random position in the cycle
        for i, spinner in enumerate(spinners):
            for _ in range(i * 2):
                next(spinner)

        self._hide_cursor()
        while not self.done:
            spinner_display = "".join([next(spinner) for spinner in spinners])
            sys.stdout.write(f"\r{colors[self.color]}{self.message} {colors['darkyellow']}{spinner_display}")
            sys.stdout.flush()
            time.sleep(0.1)

        # Complete the spinner
        complete = f"{colors['lightgreen']}{done_frame}"
        sys.stdout.write(f"\r{colors[self.color]}{self.message} {complete}")
        time.sleep(0.5)
        sys.stdout.write("\n")
        self._show_cursor()

    def start(self):
        threading.Thread(target=self.spin).start()

    def stop(self):
        self.done = True


if __name__ == "__main__":
    # Print all the colors
    for color in colors.keys():
        cprint(color, f"Hello world! ({color})")

    # Print a list of color-text pairs
    cprint(["red", "Hello", "green", "World"])

    # Print status lights
    print(status_lights(["red", "green", "yellow"]))

    # Spinner Testing
    def long_running_operation():
        # Simulate a long operation, replace this with your actual operation
        time.sleep(2)

    if __name__ == "__main__":
        spinner = Spinner("lightpurple", "Chatting with AWS:")
        spinner.start()  # Start the spinner
        try:
            long_running_operation()  # Your long-running operation here
        finally:
            spinner.stop()
