"""PyTorch tabular hyperparameter-search objective + default search space.

The PyTorch adapter for :mod:`workbench.training.hpo_runner`: a default search space
(this module) and, per trial, a full ``n_folds`` ensemble scored as ``holdout_mae`` /
``cv_mae`` (the search *objective*). Scoring a trial in the regime the winner is
published in is the point — a config selected as a lone model does not carry over to an
ensemble. The runner owns everything around that: selection, the re-rank, the artifacts.

Trials build and train through :mod:`workbench.endpoints.pytorch_utils` — the same
``create_model`` / ``train_model`` the template publishes with — so a searched config maps
to the identical architecture and optimizer.

Training-only; imported **only inside the PyTorch template's ``__main__``** (deferred).
"""

from __future__ import annotations

from workbench.training.hpo_harness import Choice, FloatRange
from workbench.training.hpo_runner import HpoAdapter, run_hpo

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
    "lr": {
        "learning_rate": FloatRange(1e-4, 1e-2, log=True, default=1e-3),
        "weight_decay": FloatRange(1e-6, 1e-2, log=True, default=1e-4),
        "batch_size": Choice([32, 64, 128, 256, 512], default=64),
    },
}


def pytorch_search_space(groups=("basic", "lr")) -> dict:
    """Build the default PyTorch search space for the named knob ``groups``.

    Args:
        groups: iterable of group names — ``"basic"`` (architecture capacity) and/or
            ``"lr"`` (optimizer + batch size). Both by default.

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

    Accepts a shorthand string (``"basic"``, ``"basic+lr"``), an iterable of group
    names, or a ready ``{knob: Spec}`` dict (passed through for full custom control).
    Defaults to all groups (``basic+lr``).
    """
    if spec is None:
        return pytorch_search_space()
    if isinstance(spec, str):
        return pytorch_search_space(spec.split("+"))
    if isinstance(spec, dict):
        return spec
    return pytorch_search_space(tuple(spec))


class PyTorchAdapter(HpoAdapter):
    """Trains and scores one PyTorch tabular candidate for :func:`run_hpo`.

    Carries the fitted preprocessing the template built (scaler, category mappings) so a
    trial sees exactly the inputs the published model will. Regression only — the
    objective is MAE.
    """

    def __init__(
        self,
        *,
        target: str,
        continuous_cols: list,
        categorical_cols: list,
        category_mappings: dict,
        categorical_cardinalities: list,
        scaler,
        n_outputs: int = 1,
        task: str = "regression",
        device: str = "cpu",
    ):
        self.target = target
        self.continuous_cols = list(continuous_cols)
        self.categorical_cols = list(categorical_cols)
        self.category_mappings = category_mappings
        self.categorical_cardinalities = list(categorical_cardinalities)
        self.scaler = scaler
        self.n_outputs = n_outputs
        self.task = task
        self.device = device

    def resources_per_trial(self, hpo_block, backend):
        """Two trials per GPU roughly saturates one without spilling. Ray only."""
        if backend == "optuna":
            return None
        return {"gpu": hpo_block.get("gpus_per_trial", 0.5)}

    def make_trial_fn(self, *, train_df, folds, val_df, hyperparameters, metric, concurrency):
        """Build the ensemble PyTorch ``trial_fn`` (closes over the folds and eval data).

        Each trial trains one MLP per fold and scores the ensemble the way it is deployed:

        * ``holdout_mae`` — every member predicts ``val_df``; the objective is the MAE of the
          members' mean prediction, i.e. the real ensemble's held-out error.
        * ``cv_mae`` — each member is scored on its own out-of-fold rows and the objective is
          the mean across folds.
        """
        import numpy as np
        import torch

        from workbench.endpoints.pytorch_utils import create_model, predict, prepare_data, train_model

        def prep(df, with_target=True):
            return prepare_data(
                df,
                self.continuous_cols,
                self.categorical_cols,
                self.target if with_target else None,
                self.category_mappings,
                scaler=self.scaler,
            )

        # Tensors don't depend on the trial config, so build them once rather than once per
        # trial — for a 250-trial search that is the difference between prep dominating the
        # run and disappearing into it.
        prepared = [
            (prep(train_df.iloc[tr].reset_index(drop=True)), prep(train_df.iloc[va].reset_index(drop=True)))
            for tr, va in folds
        ]
        holdout = bool(len(val_df))
        if holdout:
            eval_cont, eval_cat, _, _, _ = prep(val_df, with_target=False)
            eval_y = val_df[self.target].to_numpy(dtype=float)

        def trial_fn(config, report):
            merged = self.merge_config(hyperparameters, config)
            hidden_layers = [int(x) for x in str(merged["layers"]).split("-")]

            member_preds, fold_maes = [], []
            for fold_idx, ((tr_cont, tr_cat, tr_y, _, _), (va_cont, va_cat, va_y, _, _)) in enumerate(prepared):
                torch.manual_seed(merged.get("seed", 42) + fold_idx)
                model = create_model(
                    n_continuous=len(self.continuous_cols),
                    categorical_cardinalities=self.categorical_cardinalities,
                    hidden_layers=hidden_layers,
                    n_outputs=self.n_outputs,
                    task=self.task,
                    dropout=merged["dropout"],
                )
                model, _ = train_model(
                    model,
                    tr_cont,
                    tr_cat,
                    tr_y,
                    va_cont,
                    va_cat,
                    va_y,
                    task=self.task,
                    max_epochs=merged["max_epochs"],
                    patience=merged["early_stopping_patience"],
                    batch_size=merged["batch_size"],
                    learning_rate=merged["learning_rate"],
                    weight_decay=merged["weight_decay"],
                    loss=merged["loss"],
                    device=self.device,
                    restore_best_weights=merged["restore_best_weights"],
                    verbose=False,
                )

                if holdout:
                    member_preds.append(predict(model, eval_cont, eval_cat).flatten())
                    running = float(np.nanmean(np.abs(np.mean(member_preds, axis=0) - eval_y)))
                else:
                    fold_preds = predict(model, va_cont, va_cat).flatten()
                    fold_maes.append(float(np.nanmean(np.abs(fold_preds - va_y.numpy().flatten()))))
                    running = float(np.mean(fold_maes))

                # Fold-granular pruning: a config already off the pace stops here rather than
                # paying for the remaining members.
                report(step=fold_idx + 1, **{metric: running})

            return running

        return trial_fn


def run_pytorch_hpo(
    train_df,
    val_df,
    base_hyperparameters: dict,
    hpo_block: dict,
    *,
    target: str,
    continuous_cols: list,
    categorical_cols: list,
    category_mappings: dict,
    categorical_cardinalities: list,
    scaler,
    n_outputs: int = 1,
    task: str = "regression",
    device: str = "cpu",
    smiles_column: str | None = None,
    output_dir: str | None = None,
) -> dict:
    """Run the PyTorch hyperparameter search; returns the phase-2 hyperparameters.

    See :func:`workbench.training.hpo_runner.run_hpo` for the search/re-rank contract and
    the ``hpo`` block's keys.
    """
    adapter = PyTorchAdapter(
        target=target,
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
        category_mappings=category_mappings,
        categorical_cardinalities=categorical_cardinalities,
        scaler=scaler,
        n_outputs=n_outputs,
        task=task,
        device=device,
    )
    return run_hpo(
        train_df,
        val_df,
        base_hyperparameters,
        hpo_block,
        adapter=adapter,
        search_space=resolve_search_space(hpo_block.get("search_space")),
        primary_target=target,
        smiles_column=smiles_column,
        output_dir=output_dir,
    )
