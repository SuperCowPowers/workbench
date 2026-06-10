"""Render a forest of DAGs as Unicode trees for terminal display.

Domain-agnostic and stdlib-only: you hand it ``roots`` (in render order), a
``children`` map (key -> ordered child keys), and a ``labels`` map (key ->
display string), and it returns the lines. Knows nothing about what the nodes
are -- callers build the maps (and any coloring) themselves.

Layout rules:
  - Each root renders as its own flush-left tree (no forest cap), roots
    separated by a blank line.
  - A run of single-output nodes collapses onto one line ("a ─╼ b ─╼ c") when
    it ends in a leaf; a run that ends at a fan-out stays vertical so the
    branch point reads clearly.
  - A node with children encountered on more than one branch is rendered in
    full once; later appearances are marked "(shown above)".
"""


def render_forest(roots, children, labels, indent="   "):
    """Render ``roots`` and their descendants as Unicode trees.

    Args:
        roots (list): Root node keys, in the order to render them.
        children (dict): key -> ordered list of child keys (missing/empty = leaf).
        labels (dict): key -> display string for that node.
        indent (str): Left margin prepended to every line.

    Returns:
        list[str]: The rendered lines.
    """
    lines: list[str] = []
    seen = set()

    def walk(key, prefix, is_root, is_last):
        connector = "" if is_root else ("└─╼ " if is_last else "├─╼ ")
        if key in seen and children.get(key):
            lines.append(f"{indent}{prefix}{connector}{labels[key]}  (shown above)")
            return
        # Look ahead along single-output edges; collapse onto one line only when
        # the run ends in a leaf, otherwise render this node and recurse.
        chain = [key]
        cur = key
        while len(children.get(cur, ())) == 1 and children[cur][0] not in seen:
            cur = children[cur][0]
            chain.append(cur)

        if not children.get(cur):  # ends in a leaf -> collapse the run inline
            seen.update(chain)
            lines.append(f"{indent}{prefix}{connector}" + " ─╼ ".join(labels[k] for k in chain))
            return

        # Ends at a fan-out: render just this node and recurse (stays vertical).
        seen.add(key)
        lines.append(f"{indent}{prefix}{connector}{labels[key]}")
        child_prefix = prefix + ("" if is_root else ("    " if is_last else "│   "))
        kids = children.get(key, ())
        for i, child in enumerate(kids):
            walk(child, child_prefix, False, i == len(kids) - 1)

    for root in roots:
        if lines:
            lines.append("")  # blank line between independent trees
        walk(root, "", True, True)
    return lines
