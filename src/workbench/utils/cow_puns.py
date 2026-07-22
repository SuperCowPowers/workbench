"""Cow puns for the REPL greeting. Groan-worthy by design."""

import random

# (question, punchline) -- keep them groan-worthy.
COW_PUNS = [
    ("What do you call a cow with no legs?", "Ground beef."),
    ("What's a cow's best subject in school?", "Cow-culus."),
    ("What do you call that feeling like you've done this before?", "Deja-moo."),
    ("What do you call a nonsense meeting?", "Moo-larkey."),
    ("What do you call an obnoxious cow?", "Beef jerky."),
    ("What did the mother cow say to the baby cow?", "It's pasture bedtime."),
    ("Why couldn't the cow learn?", "Everything went in one ear and out the udder."),
    ("What do you call a cow that just had a calf?", "De-calf-inated."),
    ("What do you call a cow on a trampoline?", "A milkshake."),
    ("What do you get from a pampered cow?", "Spoiled milk."),
    ("Why did the cow cross the road?", "To get to the udder side."),
    ("What do you call a sleeping cow?", "A bulldozer."),
    ("What do you call a magical cow?", "Moo-dini."),
    ("What music do cows listen to?", "Moo-sic."),
    ("What do you call a cow that can't moo?", "A milk dud."),
    ("Where do cows go on Friday night?", "To the moo-vies."),
    ("What do you call a cow eating grass in your yard?", "A lawn moo-er."),
    ("Why do cows wear bells?", "Because their horns don't work."),
    ("What do you call a herd of cows telling stories?", "A load of bull."),
    ("What do you get when you cross a cow and a rooster?", "Roost beef."),
    ("What do you call a group of cows with a sense of humor?", "Laughing stock."),
    ("What do you call a cow with two legs?", "Lean beef."),
    ("What do you call a cow that tells jokes?", "A cow-median."),
    ("What do you get when you cross a smurf with a cow?", "Blue cheese."),
    ("Why did the Secret Service surround the president with cows?", "They were beefing up security."),
]


def random_cow_pun() -> tuple[str, str]:
    """One pun at random, as (question, punchline)."""
    return random.choice(COW_PUNS)
