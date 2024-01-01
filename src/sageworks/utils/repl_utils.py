"""Repl utilities for Sageworks"""
colors = {
    "lightblue": "\x1b[38;5;111m",
    "lightpurple": "\x1b[38;5;141m",
    "lightgreen": "\x1b[38;5;113m",
    "lime": "\x1b[38;5;154m",
    "darkyellow": "\x1b[38;5;172m",
    "orange": "\x1b[38;5;202m",
    "pink": "\x1b[38;5;213m",
    "magenta": "\x1b[38;5;206m",
    "red": "\x1b[38;5;196m",
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


def cprint(text, color):
    print(f"{colors[color]}{text}{colors['reset']}")


if __name__ == "__main__":
    for color in colors.keys():
        cprint(f"Hello world! ({color})", color)
