# Searching the Workbench Source

When you need the real signature, the actual default, or how something is
implemented — read the source instead of guessing. It ships with the package, so
it is always available and always matches the version the user is running.

Most users `pip install workbench`, so the code lives in site-packages, not a
cloned repo. Always locate it dynamically; never hardcode a path.

## Locate the package

```python
import workbench, pathlib
ROOT = pathlib.Path(workbench.__file__).parent
```

## Search

```python
import re

def code_search(pattern, root=ROOT, flags=0):
    rx = re.compile(pattern, flags)
    for path in sorted(root.rglob("*.py")):
        for i, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
            if rx.search(line):
                print(f"{path.relative_to(root)}:{i}: {line.strip()}")

code_search(r"def cross_fold_inference")
code_search(r"serverless")
```

Useful patterns:

- Definition of a method or class: `rf"(def|class)\s+{name}\b"`
- Everywhere something is called: `rf"\.{name}\("`
- A keyword argument's default: `rf"{name}\s*[:=]"`

## Read the surrounding code

Once you have a hit, print the block — a signature alone often hides the
defaults and the docstring that explain it.

```python
path = ROOT / "api/model.py"
lines = path.read_text().splitlines()
print("\n".join(lines[54:90]))     # 1-indexed line 55 onward
```

## Answer from what you read

Quote the real signature and defaults you found, and cite the location as
`api/model.py:55`. If the source contradicts a guide or something the user
believes, say so — the source wins.

## Pointing the user at GitHub

The REPL session has no browser, so search locally as above. When the user wants
something to open or share, hand them a link built from the path you found:

```
https://github.com/SuperCowPowers/workbench/blob/main/src/workbench/<relative_path>
```

For example `api/model.py` → `.../blob/main/src/workbench/api/model.py`.
