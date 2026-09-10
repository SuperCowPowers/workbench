"""Structural invariants for the bucketed guide directory."""

import pytest

from workbench.agent.tools import ALWAYS_LOADED, GUIDES_DIR, _description, guide_names, guide_paths, read_guide


def test_guide_stems_are_unique_across_buckets():
    """`read_guide` addresses a guide by bare name, so two buckets can't share a stem."""
    by_stem = {}
    for path in GUIDES_DIR.rglob("*.md"):
        by_stem.setdefault(path.stem, []).append(path.relative_to(GUIDES_DIR))
    collisions = {stem: paths for stem, paths in by_stem.items() if len(paths) > 1}
    assert not collisions, f"Guide names must be unique across buckets: {collisions}"


def test_every_guide_has_a_description():
    """The `> one-liner` is how Bosco decides to open a guide; a missing one hides it."""
    missing = [p.stem for p in guide_paths() if not _description(p)]
    assert not missing, f"Guides with no `> description` under their H1: {missing}"


@pytest.mark.parametrize("name", guide_names())
def test_every_indexed_guide_is_readable_by_bare_name(name):
    """Whatever the index offers, `read_guide` must resolve without a bucket prefix."""
    assert not read_guide(name).startswith("No guide named")


def test_always_loaded_guides_stay_out_of_the_menu():
    """An always-injected guide in the lazy-read menu is a second copy in context."""
    assert ALWAYS_LOADED.isdisjoint(guide_names())
