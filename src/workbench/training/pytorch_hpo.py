"""PyTorch tabular hyperparameter-search objective + default search space.

The PyTorch adapter for :mod:`workbench.training.hpo_runner`: a default search space
(this module) and, per trial, a full ``n_folds`` ensemble scored as pooled out-of-fold
MAE (the search *objective*). Scoring a trial in the regime the winner is
published in is the point — a config selected as a lone model does not carry over to an
ensemble. The runner owns everything around that: fold construction, the baseline trial, the artifacts.

Trials train through :func:`workbench.training.pytorch_core.train_pytorch_fold` — the
same recipe the template publishes with — so a searched config maps to the identical
architecture, seeding, and optimizer.

Training-only; imported **only inside the PyTorch template's ``__main__``** (deferred).
"""

from __future__ import annotations

from workbench.training.hpo_harness import Choice, FloatRange, SearchSpace
from workbench.training.hpo_runner import HpoAdapter, pooled_mae, run_hpo

# Default per-knob search space, split into the two levers that matter for a tabular MLP.
# Both groups are searched by default. Everything else — n_folds, split_strategy, loss,
# uq_version, seed — stays fixed at its configured value.
#
# Deliberately NOT searched:
#   * max_epochs / early_stopping_patience — early stopping already owns the epoch budget,
#     so a searched epoch count would only fight it.
#   * restore_best_weights — it is False on purpose. Letting each fold stop a few epochs
#     past its best val loss is what produces distinct weights across folds, which is the
#     dominant source of ensemble std for UQ. A cv_mae search cannot see that cost and
#     would flip it, quietly degrading the uncertainty estimates.
_SEARCH_GROUPS = {
    # Capacity. `layers` is the template's dash-separated widths, searched as whole shapes
    # rather than decomposed into depth/width — the shapes carry the taper, which is the
    # part that matters, and the template needs no new knob format.
    "basic": {
        "layers": Choice(
            [
                "128-64",
                "256-128",
                "512-256",
                "512-256-128",
                "1024-512-256",
                "512-512-512",
                "1024-512-256-128",
            ],
            default="512-256-128",
        ),
        "dropout": FloatRange(0.0, 0.4, step=0.05, default=0.05),
    },
    # Optimizer. learning_rate and batch_size interact (they scale together), and
    # weight_decay is the main explicit regularizer on a tabular MLP.
    "optimizer": {
        "learning_rate": FloatRange(1e-4, 1e-2, log=True, default=1e-3),
        "weight_decay": FloatRange(1e-6, 1e-2, log=True, default=1e-4),
        "batch_size": Choice([32, 64, 128, 256, 512], default=64),
    },
}


def pytorch_search_space(groups=("basic", "optimizer")) -> dict:
    """Build the default PyTorch search space for the named knob ``groups``.

    Args:
        groups: iterable of group names — ``"basic"`` (architecture capacity) and/or
            ``"optimizer"`` (optimizer knobs + batch size). Both by default.

    Returns:
        dict: ``{knob: Spec}`` for :func:`workbench.training.hpo_harness.run_search`.
    """
    space = {}
    for group in groups:
        if group not in _SEARCH_GROUPS:
            raise ValueError(f"Unknown search group {group!r}. Known: {sorted(_SEARCH_GROUPS)}")
        space.update(_SEARCH_GROUPS[group])
    return space


def resolve_search_space(spec) -> dict:
    """Resolve an ``hpo['search_space']`` value into a ``{knob: Spec}`` space.

    Accepts a shorthand string (``"basic"``, ``"basic+optimizer"``), an iterable of group
    names, or a ready ``{knob: Spec}`` dict (passed through for full custom control).
    Defaults to all groups (``basic+optimizer``).
    """
    if spec is None:
        return pytorch_search_space()
    if isinstance(spec, str):
        return pytorch_search_space(spec.split("+"))
    if isinstance(spec, dict):
        # A dict of dicts is the JSON wire form; a dict of Specs is already a space
        if spec and all(isinstance(v, dict) for v in spec.values()):
            return SearchSpace.from_dict(spec)
        return spec
    return pytorch_search_space(tuple(spec))


class PyTorchAdapter(HpoAdapter):
    """Trains and scores one PyTorch tabular candidate for :func:`run_hpo`.

    Carries the template's :class:`~workbench.training.pytorch_core.PyTorchFoldSpec` —
    the fitted preprocessing (scaler, category mappings) and model shape — so a trial
    sees exactly the inputs the published model will. Regression only — the objective
    is MAE.
    """

    def __init__(self, *, spec):
        self.spec = spec

    # A tabular trial is minutes, so the budget buys breadth cheaply.
    default_n_trials = 100

    def resources_per_trial(self, hpo_block, backend):
        """Two trials per GPU roughly saturates one without spilling. Ray only."""
        if backend == "optuna":
            return None
        return {"gpu": hpo_block.get("gpus_per_trial", 0.5)}

    def make_trial_fn(self, *, train_df, folds, hyperparameters, concurrency):
        """Build the ensemble PyTorch ``trial_fn`` (closes over the folds).

        Each trial trains one MLP per fold through
        :func:`workbench.training.pytorch_core.train_pytorch_fold` — the same recipe the
        template publishes with, so a trial's members are seeded, built, and early-stopped
        identically. Each member predicts its own out-of-fold rows and the objective is one
        MAE over those pooled predictions, reported after every fold so the harness can stop
        a trial that is already off the pace.
        """
        from dataclasses import replace

        from workbench.endpoints.pytorch_utils import predict, prepare_data
        from workbench.training.pytorch_core import train_pytorch_fold

        spec = self.spec

        def prep(df):
            tensors = prepare_data(
                df,
                spec.continuous_cols,
                spec.categorical_cols,
                spec.target,
                spec.category_mappings,
                scaler=spec.scaler,
            )
            return tensors[:3]  # (x_cont, x_cat, y)

        # Tensors don't depend on the trial config, so build them once rather than once per
        # trial — for a 250-trial search that is the difference between prep dominating the
        # run and disappearing into it.
        prepared = [
            (prep(train_df.iloc[tr].reset_index(drop=True)), prep(train_df.iloc[va].reset_index(drop=True)))
            for tr, va in folds
        ]
        # Trials must train the way the published model does, weights included.
        import torch as _torch

        has_w = "sample_weight" in train_df.columns
        fold_sample_weight = [
            _torch.tensor(train_df.iloc[tr]["sample_weight"].to_numpy(dtype="float32")) if has_w else None
            for tr, _ in folds
        ]

        def trial_fn(config, report):
            trial_spec = replace(spec, hyperparameters=self.merge_config(hyperparameters, config), verbose=False)

            oof_pred, oof_true = [], []
            for fold_idx, (train_tensors, val_tensors) in enumerate(prepared):
                model, _ = train_pytorch_fold(
                    trial_spec,
                    train_tensors,
                    val_tensors,
                    fold_idx=fold_idx,
                    train_sample_weight=fold_sample_weight[fold_idx],
                )

                va_cont, va_cat, va_y = val_tensors
                oof_pred.append(predict(model, va_cont, va_cat).flatten())
                oof_true.append(va_y.numpy().flatten())
                running = pooled_mae(oof_pred, oof_true)
                report(fold_idx + 1, running)

            return running

        return trial_fn


def run_pytorch_hpo(
    train_df,
    base_hyperparameters: dict,
    hpo_block: dict,
    *,
    spec,
    output_dir: str | None = None,
) -> dict:
    """Run the PyTorch hyperparameter search; returns the phase-2 hyperparameters.

    ``spec`` is the template's :class:`~workbench.training.pytorch_core.PyTorchFoldSpec`,
    carrying the fitted preprocessing. See :func:`workbench.training.hpo_runner.run_hpo`
    for the search contract and the ``hpo`` block's keys.
    """
    return run_hpo(
        train_df,
        base_hyperparameters,
        hpo_block,
        adapter=PyTorchAdapter(spec=spec),
        search_space=resolve_search_space(hpo_block.get("search_space")),
        primary_target=spec.target,
        output_dir=output_dir,
    )
