"""workbench.training — the import surface for training-container code.

The training-time counterpart to :mod:`workbench.endpoints`. A generated
model script runs in **two** containers — the training image and the (leaner)
inference endpoint — so its imports are split by where they're needed:

* **Inference-time** code (``model_fn``/``predict_fn`` and anything they reach)
  imports from :mod:`workbench.endpoints`, which is bound by a lean CI dependency
  contract (``endpoint-import-smoke``): no heavy deps may be reachable from that
  surface.
* **Training-only** code imports from here. The training image is heavy, so these
  modules MAY require training-only deps (``torch``, ``chemprop``, ``ray``,
  ``optuna``). The ``training-import-smoke`` CI check verifies they import against
  the training-image manifest.

**Critical invariant.** The endpoint imports the whole model script at load time
and calls the inference handlers, but it never runs the script's ``__main__``.
Training-only deps are absent from the endpoint image, so a template may import
``workbench.training.*`` **only inside its ``__main__`` block** (a deferred/local
import) — never at top level. A top-level ``workbench.training.*`` import would
break endpoint loading. A lint enforces this.

Rule for placing a module: it belongs here only if it is **not on any endpoint
import surface** (no top-level import in any deployed template). Dual-use modules
(e.g. :mod:`workbench.endpoints.uq_harness`, imported at endpoint load for the UQ
apply path) stay in :mod:`workbench.endpoints`.

Modules:

* :mod:`training_harness` — the ModelTrainer wrapper copied into the training
  source dir and run as ``python training_harness.py <entry_point>`` (a copied
  script, not an imported symbol).
"""

# Pin awswrangler to its single-node Python/pandas engine (see workbench/__init__.py).
# This surface owns the pin for the training container — where `ray` lives (HPO's
# `ray[tune]`) and would flip awswrangler to its distributed engine — since model
# scripts import only `workbench.endpoints.*`/`workbench.training.*`, never a global
# `workbench`.
try:
    import awswrangler as wr

    wr.engine.set("python")
    wr.memory_format.set("pandas")
except ImportError:
    # awswrangler is optional; engine tuning only applies when it's installed
    pass
