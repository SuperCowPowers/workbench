"""Repl utilities for Workbench"""

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
    if isinstance(args[0], list):
        args = args[0]
    # Iterate over the arguments in pairs of color and text
    for i in range(0, len(args), 2):
        print(f"{colors[args[i]]}{args[i + 1]}{colors['reset']}", end=" ")
    print()  # Print a newline at the end


def status_lights(status_colors: list[str]):
    """
    Create status lights (circles) in color.

    Args:
        status_colors: A list of status colors (e.g. ['red', 'green', 'yellow']).

    Returns:
        A string representation of colored status lights.
    """
    circle = "●"  # Unicode character for a filled circle
    return "".join(f"{colors[color]}{circle}{colors['reset']}" for color in status_colors)


# Spinner Class
class Spinner:
    def __init__(self, color, message="Loading..."):
        """
        Initialize the Spinner object.

        Args:
            color: The color of the spinner.
            message: The message to display beside the spinner.
        """
        self.done = False
        self.message = message
        self.color = color
        self.thread = threading.Thread(target=self.spin)

    def _write(self, text):
        """Write text to stdout and flush."""
        sys.stdout.write(text)
        sys.stdout.flush()

    def _hide_cursor(self):
        """Hide the terminal cursor."""
        self._write("\033[?25l")

    def _show_cursor(self):
        """Show the terminal cursor."""
        self._write("\n\033[?25h")

    def spin(self):
        """Display a multi-spinner animation until stopped."""
        frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"  # Frames used for each spinner
        done_frame = "⠿⠿⠿⠿"  # Frames to show on completion

        # Create four separate cycle iterators for the spinners
        spinners = [itertools.cycle(frames) for _ in range(4)]

        # Initialize each spinner with staggered positions for visual effect
        for i, spinner in enumerate(spinners):
            for _ in range(i * 2):  # Stagger each spinner by 2 frames
                next(spinner)

        self._hide_cursor()
        while not self.done:
            # Construct the spinner display from each staggered spinner
            spinner_display = "".join(next(spinner) for spinner in spinners)
            self._write(f"\r{colors[self.color]}{self.message} {colors['darkyellow']}{spinner_display}")
            time.sleep(0.1)  # Control the speed of spinner animation

        # Display the "done" frame when completed
        self._write(f"\r{colors[self.color]}{self.message} {colors['lightgreen']}{done_frame}")
        self._show_cursor()

    def start(self):
        """Start the spinner in a separate thread."""
        self.thread.start()

    def stop(self):
        """Stop the spinner and wait for the thread to finish."""
        self.done = True
        self.thread.join()  # Ensure the spinner thread is fully stopped


if __name__ == "__main__":
    # Print all the colors for a quick visual test
    for color, _ in colors.items():
        cprint(color, f"Hello world! ({color})")

    # Print a list of color-text pairs
    cprint(["red", "Hello", "green", "World"])

    # Print a string of status lights
    print(status_lights(["red", "green", "yellow"]))

    # Spinner Testing
    def long_running_operation():
        """Simulate a long-running operation."""
        time.sleep(3)

    # Test the spinner with a sample message
    spinner = Spinner("lightpurple", "Chatting with AWS:")
    spinner.start()
    try:
        long_running_operation()  # Simulate a long-running task
    finally:
        spinner.stop()
        print("Done!")
