"""Cow puns for the REPL greeting. Puns live in agent/guides/cow_puns.md."""

import random
from pathlib import Path

PUNS_FILE = Path(__file__).parent.parent / "agent" / "guides" / "cow_puns.md"


def cow_puns() -> list[tuple[str, str]]:
    """Every `question :: punchline` pun in the guide, as (question, punchline)."""
    puns = []
    for line in PUNS_FILE.read_text().splitlines():
        if " :: " in line and not line.startswith(("#", ">")):
            question, _, punchline = line.partition(" :: ")
            puns.append((question.strip(), punchline.strip()))
    return puns


def random_cow_pun() -> tuple[str, str]:
    """One pun at random, as (question, punchline)."""
    return random.choice(cow_puns())
