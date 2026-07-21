# Introspecting Live Objects

> when you have an object or class in hand, ask Python what it really is — signature, docstring, methods, source

You already have a `Model`, an `Endpoint`, a store, or a util function. Don't
guess what it can do — introspect it. This is faster than searching the tree
(you skip the name hunt) and it is **exact**: the answer comes from the running
code, at the version the user has installed. Reach for this before calling a
method you're unsure exists — `dir()` and `inspect.signature` are how you avoid
inventing an API.

The sibling guide `code_search` is for the other direction: hunting by name
across the whole source tree when you *don't* have the object yet.

## What can this thing do?

```python
[m for m in dir(obj) if not m.startswith("_")]     # public attributes + methods
```

`dir()` is the source of truth for what exists. If a method you were about to
call isn't in this list, it doesn't exist — find the real one instead of calling
it and reading a traceback.

## Exact signature and defaults

```python
import inspect

inspect.signature(obj.get_inference_metrics)     # (capture_name: str = 'default')
```

This resolves the real parameter names and their defaults — no guessing whether
an argument is `capture` or `capture_name`, or what it defaults to.

## Docstring

```python
print(inspect.getdoc(obj))                  # the class docstring
print(inspect.getdoc(obj.to_endpoint))      # one method's docstring
```

`getdoc` cleans indentation and follows inheritance; prefer it over `.__doc__`.

## The actual source

```python
print(inspect.getsource(obj.cross_fold_inference))   # the method body
print(inspect.getsource(type(obj)))                  # the whole class
```

When the docstring is thin or you need to know *how* something behaves (what it
does on an empty frame, which default path it takes), read the body. The source
always wins over a guide.

## Where does it live?

```python
cls = type(obj)
cls.__module__                     # 'workbench.api.model'
inspect.getsourcefile(cls)         # absolute path in site-packages
inspect.getsourcelines(cls)[1]     # 1-indexed line where the class starts
```

Turn that into a shareable GitHub link with the relative path (see
`code_search` for the URL shape).

## Utility functions, not just classes

Same tools work on any module-level function — this is how you inspect
`chem_utils`, `color_utils`, or any helper. `chem_utils` is a namespace package,
so import the submodule the function lives in (e.g. `vis`), not the package:

```python
from workbench.utils.chem_utils import vis
inspect.signature(vis.molecule_grid)
print(inspect.getsource(vis.molecule_grid))
[f for f in dir(vis) if not f.startswith("_")]   # what the submodule offers
```

## One helper to do it all

```python
def explain(obj):
    """Signature (if callable), docstring, and where it's defined."""
    import inspect
    if callable(obj):
        try:
            print(f"{getattr(obj, '__name__', obj)}{inspect.signature(obj)}\n")
        except (TypeError, ValueError):
            pass
    print(inspect.getdoc(obj) or "(no docstring)")
    try:
        print(f"\n-- {inspect.getsourcefile(obj)}:{inspect.getsourcelines(obj)[1]}")
    except (TypeError, OSError):
        pass

explain(Endpoint.cross_fold_inference)
explain(vis.molecule_grid)
```

## Answer from what you found

Quote the real signature and defaults, and cite the location as
`api/model.py:55`. If introspection contradicts a guide or what the user
expects, say so plainly — the live code is the truth.

When the user asked to **see** the code ("show me the Model class"), don't just
`print()` it — `run_python` output comes back to you, not to their screen. Put
the source in your reply inside a ```python fenced block so it renders for them
(see `general` → showing code). If a whole class is too long to return in one
read, show its signature and the methods they care about, then cite the path.
