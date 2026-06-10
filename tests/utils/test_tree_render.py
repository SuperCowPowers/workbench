"""Unit tests for the generic Unicode forest renderer."""

from workbench.utils.tree_render import render_forest


def test_single_chain_to_leaf_collapses():
    """A run of single-output nodes ending in a leaf renders on one line."""
    children = {"a": ["b"], "b": ["c"], "c": []}
    labels = {"a": "a", "b": "b", "c": "c"}
    lines = render_forest(["a"], children, labels, indent="")
    assert lines == ["a ─╼ b ─╼ c"]


def test_chain_to_fanout_stays_vertical():
    """A single-output run that ends at a fan-out is rendered vertically."""
    children = {"a": ["b"], "b": ["c", "d"], "c": [], "d": []}
    labels = {k: k for k in children}
    lines = render_forest(["a"], children, labels, indent="")
    assert lines == [
        "a",
        "└─╼ b",
        "    ├─╼ c",
        "    └─╼ d",
    ]


def test_leaf_children_collapse_under_fanout():
    """Each fan-out child that ends in a leaf collapses inline."""
    children = {"fs": ["m1", "m2"], "m1": ["out1"], "m2": ["out2"], "out1": [], "out2": []}
    labels = {k: k for k in children}
    lines = render_forest(["fs"], children, labels, indent="")
    assert lines == [
        "fs",
        "├─╼ m1 ─╼ out1",
        "└─╼ m2 ─╼ out2",
    ]


def test_shared_node_marked_shown_above():
    """A node reachable from two parents is expanded once, then referenced.

    The first root absorbs ``shared`` into its leaf-terminated chain; the second
    root, finding ``shared`` already shown, marks it instead of re-expanding.
    """
    children = {"r1": ["shared"], "r2": ["shared"], "shared": ["leaf"], "leaf": []}
    labels = {k: k for k in children}
    lines = render_forest(["r1", "r2"], children, labels, indent="")
    assert lines == [
        "r1 ─╼ shared ─╼ leaf",
        "",  # blank line between roots
        "r2",
        "└─╼ shared  (shown above)",
    ]


def test_multiple_roots_separated_by_blank_line():
    """Independent roots render as separate flush-left trees."""
    children = {"a": [], "b": []}
    labels = {"a": "a", "b": "b"}
    lines = render_forest(["a", "b"], children, labels, indent="")
    assert lines == ["a", "", "b"]


def test_indent_prefixes_every_line():
    children = {"a": ["b"], "b": []}
    labels = {"a": "a", "b": "b"}
    lines = render_forest(["a"], children, labels, indent="   ")
    assert lines == ["   a ─╼ b"]
