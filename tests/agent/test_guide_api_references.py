"""Lint: every Workbench API reference in Bosco's guides must resolve to a real symbol.

The guides are Bosco's priors. A guide that names a method which doesn't exist
teaches Bosco to call it — the exact failure that once put
`model.performance_metrics()` into a guide. This walks the guide markdown, pulls
out `<receiver>.<method>(` calls whose receiver is a known Workbench class or a
conventional instance variable, and asserts each method exists on that class. It
also checks that every symbol the guides import from `workbench.*` actually imports.

The check is deliberately high-precision: it only validates receivers it can
resolve (the conventions below), so it never flags `df.head()` or
`pd.to_datetime()`. As the guides adopt new conventions, extend CLASS_REFS /
CONVENTIONAL_VARS.
"""

import importlib
import re

import pytest

from workbench.agent.tools import GUIDES_DIR
from workbench.algorithms.dataframe.fingerprint_proximity import FingerprintProximity
from workbench.api import (
    DataSource,
    DFStore,
    Endpoint,
    FeatureSet,
    InferenceCache,
    InferenceStore,
    Meta,
    MetaEndpoint,
    Model,
    ModelFramework,
    ModelType,
    Monitor,
    ParameterStore,
    PublicData,
    Reports,
)
from workbench.cached.cached_meta import CachedMeta

# Class name as written in the guides -> the actual class.
CLASS_REFS = {
    "DataSource": DataSource,
    "FeatureSet": FeatureSet,
    "Model": Model,
    "ModelType": ModelType,
    "ModelFramework": ModelFramework,
    "Endpoint": Endpoint,
    "MetaEndpoint": MetaEndpoint,
    "InferenceCache": InferenceCache,
    "Meta": Meta,
    "ParameterStore": ParameterStore,
    "DFStore": DFStore,
    "Reports": Reports,
    "InferenceStore": InferenceStore,
    "PublicData": PublicData,
    "Monitor": Monitor,
    "CachedMeta": CachedMeta,
}

# Instance-variable names the guides use consistently -> their class. These are the
# naming conventions the guides follow (fs = FeatureSet, end = Endpoint, ...); the
# lint leans on them to resolve instance-method calls it otherwise couldn't type.
CONVENTIONAL_VARS = {
    "ds": DataSource,
    "fs": FeatureSet,
    "model": Model,
    "end": Endpoint,
    "endpoint": Endpoint,
    "prox": FingerprintProximity,
}

RECEIVERS = {**CLASS_REFS, **CONVENTIONAL_VARS}

# `receiver.method(` or `Receiver().method(` — the optional `()` covers class
# instantiation like `CachedMeta().feature_sets()`. Only the first link of a chain
# is captured (a chained call's receiver is a `)`, whose return type we can't resolve).
CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\s*(?:\([^)]*\))?\s*\.\s*([A-Za-z_]\w*)\s*\(")

# Bare attribute reference `receiver.name` NOT followed by `(` — a method passed by
# reference (`explain(Endpoint.cross_fold_inference)`) or an enum member
# (`ModelType.REGRESSOR`). Checked for CLASS receivers only (see method_failures).
ATTR_RE = re.compile(r"\b([A-Za-z_]\w*)\s*(?:\([^)]*\))?\s*\.\s*([A-Za-z_]\w*)\b(?!\s*\()")

# `from workbench... import (...)` in both parenthesized-multiline and single-line forms.
PAREN_IMPORT_RE = re.compile(r"from\s+(workbench[\w.]*)\s+import\s+\((.*?)\)", re.DOTALL)
LINE_IMPORT_RE = re.compile(r"from\s+(workbench[\w.]*)\s+import\s+([^\n(]+)")

FENCE_RE = re.compile(r"```.*?\n(.*?)```", re.DOTALL)
INLINE_RE = re.compile(r"`([^`\n]+)`")


def _code_text(md_text: str) -> str:
    """Just the code: fenced blocks plus inline-code spans, prose stripped out."""
    return "\n".join(FENCE_RE.findall(md_text) + INLINE_RE.findall(md_text))


def method_failures(md_text: str) -> list:
    """API references in the code that don't resolve, as readable strings.

    Two rules, each tuned for precision:

    - **Method calls** `<recv>.<name>(` on any known receiver — a class or a
      conventional instance var. Calls are where invented methods surface
      (`model.performance_metrics()`).
    - **Bare attribute refs** `<Class>.<name>` (a method passed by reference, an
      enum member) on a known *class* only. Class members live on the class, so
      `hasattr` is reliable; instance vars are skipped here because attributes set
      in `__init__` aren't visible on the class and would false-positive.
    """
    code = _code_text(md_text)
    bad = []
    for receiver, name in CALL_RE.findall(code):
        cls = RECEIVERS.get(receiver)
        if cls is not None and not hasattr(cls, name):
            bad.append(f"{receiver}.{name}()")
    for receiver, name in ATTR_RE.findall(code):
        cls = CLASS_REFS.get(receiver)
        if cls is not None and not hasattr(cls, name):
            bad.append(f"{receiver}.{name}")
    return bad


_IDENT_RE = re.compile(r"[A-Za-z_]\w*")


def import_failures(md_text: str) -> list:
    """(module, name) pairs the guides import from workbench that don't resolve."""
    bad = []
    matches = PAREN_IMPORT_RE.findall(md_text) + LINE_IMPORT_RE.findall(md_text)
    for module_path, names_blob in matches:
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            bad.append((module_path, "<module import failed>"))
            continue
        for raw in names_blob.split(","):
            ident = _IDENT_RE.match(raw.strip().split(" as ")[0].strip())
            if not ident:
                continue  # trailing backtick, comment, or other prose noise
            name = ident.group()
            if hasattr(module, name):
                continue
            try:
                importlib.import_module(f"{module_path}.{name}")  # `from pkg import submodule`
            except ImportError:
                bad.append((module_path, name))
    return bad


def _guides():
    return sorted(GUIDES_DIR.glob("*.md"))


@pytest.mark.parametrize("guide", _guides(), ids=lambda p: p.stem)
def test_guide_method_calls_resolve(guide):
    """Every method call and class-attribute reference in a guide resolves."""
    failures = method_failures(guide.read_text())
    assert not failures, "Unknown API references in {}: {}".format(guide.name, ", ".join(failures))


@pytest.mark.parametrize("guide", _guides(), ids=lambda p: p.stem)
def test_guide_imports_resolve(guide):
    """Every `from workbench... import X` in a guide resolves to a real symbol."""
    failures = import_failures(guide.read_text())
    assert not failures, "Unresolved imports in {}: {}".format(
        guide.name, ", ".join(f"{mod}.{name}" for mod, name in failures)
    )


def test_linter_flags_a_fake_method():
    """Guard the lint itself: bogus references caught, real ones not."""
    # Invented method call on an instance var.
    assert method_failures("`model.performance_metrics()`") == ["model.performance_metrics()"]
    assert method_failures("`model.details()`") == []
    # Misattributed method passed by reference — a class attr not followed by `(`.
    assert method_failures("`explain(Model.cross_fold_inference)`") == ["Model.cross_fold_inference"]
    assert method_failures("`explain(Endpoint.cross_fold_inference)`") == []
    # Enum member on a class resolves; a bogus one is caught.
    assert method_failures("`ModelType.REGRESSOR`") == []
    assert method_failures("`ModelType.NOPE`") == ["ModelType.NOPE"]
    # Instance-var bare attribute is deliberately skipped (would false-positive).
    assert method_failures("`prox.id_column`") == []
    # Imports.
    assert import_failures("from workbench.api import NoSuchClass") == [("workbench.api", "NoSuchClass")]
    assert import_failures("from workbench.api import Model") == []
