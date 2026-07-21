"""Repl utilities for Workbench"""

import threading
import itertools
import time
import sys

# The color palette, cprint, and markdown rendering live in color_utils (the one
# place colors are defined). Re-exported here for the many `repl_utils` importers.
from workbench.utils.color_utils import colors, cprint, render_markdown  # noqa: F401


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
        self.clear = False
        self.message = message
        self.color = color
        self.thread = threading.Thread(target=self.spin)
        self.stream = sys.stdout

    def _write(self, text):
        """Write to the pinned terminal stream and flush.

        Pinned at start() rather than read live, so a `redirect_stdout` running
        on another thread (e.g. Bosco capturing a tool's output) can't divert the
        animation into its capture buffer.
        """
        self.stream.write(text)
        self.stream.flush()

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

        if self.clear:
            # Erase the spinner line entirely, leaving the cursor where it started
            self._write(f"\r\033[K{colors['reset']}\033[?25h")
        else:
            # Display the "done" frame when completed
            self._write(f"\r{colors[self.color]}{self.message} {colors['lightgreen']}{done_frame}")
            self._show_cursor()

    def start(self):
        """Start the spinner in a separate thread."""
        self.stream = sys.stdout  # pin the terminal stream before any redirect
        self.thread.start()

    def stop(self, clear: bool = False):
        """Stop the spinner and wait for the thread to finish.

        Args:
            clear: Erase the spinner line instead of leaving a "done" frame.
                   Use for repeated/transient waits that shouldn't pile up.
        """
        self.clear = clear
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
