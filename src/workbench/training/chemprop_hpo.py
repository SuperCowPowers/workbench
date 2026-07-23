"""Chemprop hyperparameter-search objective + default search space.

Drives the framework-agnostic :mod:`workbench.training.hpo_harness` for chemprop:
a default search space (this module) and, per trial, a single-fold chemprop train
scored as held-out ``holdout_mae`` (the search *objective*). Training-only; imported
**only inside the chemprop template's ``__main__``** (deferred).

Parity: each trial builds its model through
:func:`workbench.training.chemprop_core.build_mpnn_model` — the same builder the
template uses to publish the winner — so a searched config maps to the identical
architecture and optimizer schedule.

Only the pure pieces here (the search space and config merge) import without
chemprop; the search entry point and per-trial train (added next) defer their
chemprop/chemprop_core imports so this module stays importable for unit tests.
"""

from __future__ import annotations

import os

from workbench.training.hpo_harness import Choice, FloatRange, IntRange

# Default per-knob search space, grouped like chemprop's own hpopt keywords. The `basic`
# group is chemprop's canonical space verbatim, extended only so ffn_hidden_dim can also
# express a tapered head. Everything else — uq_version, max_epochs, patience, batch_size,
# split_strategy, criterion, seed — stays fixed at its configured value.
_SEARCH_GROUPS = {
    "basic": {
        # depth / hidden_dim / ffn_num_layers ranges are chemprop's canonical search space
        # verbatim. ffn_hidden_dim additionally offers tapered heads (lists), which
        # chemprop's own space can't express — ffn_num_layers is ignored when a tapered
        # list is chosen (the list length sets the depth).
        #
        # `dropout` is deliberately absent even though chemprop searches it: a trial trains
        # a *single* model, while the published model is an `n_folds` ensemble. Dropout
        # chosen without ensembling over-regularizes once the ensemble is stacked on top
        # (DEFAULT_HYPERPARAMETERS keeps it low precisely because the ensemble already
        # regularizes), so it stays at that ensemble-tuned default.
        "depth": IntRange(2, 6, 1),
        "hidden_dim": IntRange(300, 2400, 100),
        "ffn_num_layers": IntRange(1, 3, 1),
        "ffn_hidden_dim": Choice([300, 600, 1200, 1800, 2400, [1024, 256, 64], [512, 128]]),
    },
    # Opt-in ("basic+lr"). init_lr/final_lr are tied to max_lr in merge_best_config
    # rather than searched independently (independent search can produce init > max,
    # which the Noam schedule rejects).
    "lr": {
        "max_lr": FloatRange(1e-4, 5e-3, log=True),
        "warmup_epochs": IntRange(2, 10, 2),
    },
}

# Knobs outside the default space — the working list for when this is revisited:
#
#   * `dropout` — chemprop searches it ({0.0,0.05,…,0.4}); we can't transfer a value found
#     single-model to an ensemble (see the `basic` group note). Searching it honestly needs
#     trials scored in the *deployment* regime — e.g. shortlist cheaply single-fold, then
#     re-rank the top-K configs by training each as a real `n_folds` ensemble and selecting
#     on that. Same treatment would make the capacity knobs transfer better too.
#   * learning rate (`max_lr`, `warmup_epochs`) — already available as the opt-in "lr"
#     group. Chemprop's maintainers describe their recommended search as focusing on
#     learning rate and batch size, which makes this the first candidate to promote into
#     the default.
#   * `batch_size` — searchable in chemprop, not here. It moves memory and throughput as
#     well as accuracy, so a range has to be chosen against the training instance's GPU
#     memory rather than picked from the literature.
#   * `aggregation` (mean/sum/norm) — searchable in chemprop but unreachable from here:
#     `chemprop_core.build_mpnn_model` constructs `NormAggregation` directly, so tuning it
#     requires a model-construction change first.
#   * `activation`, `aggregation_norm` — the remainder of chemprop's "all" keyword.
#
# Each added knob costs trials: the default space is already 5-dimensional, and pruning
# reserves the first PRUNE_STARTUP_TRIALS trials as un-pruned baselines.


def chemprop_search_space(groups=("basic",)) -> dict:
    """Build the default chemprop search space for the named knob ``groups``.

    Args:
        groups: iterable of group names — ``"basic"`` (architecture + dropout) and/or
            ``"lr"`` (the learning-rate schedule).

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
    Defaults to the ``basic`` group.
    """
    if spec is None:
        return chemprop_search_space()
    if isinstance(spec, str):
        return chemprop_search_space(spec.split("+"))
    if isinstance(spec, dict):
        return spec
    return chemprop_search_space(tuple(spec))


def merge_best_config(hyperparameters: dict, best_config: dict) -> dict:
    """Merge the winning search config into the base hyperparameters (phase-2 config).

    Drops the ``hpo`` block (search is done) and, when ``max_lr`` was searched, ties
    ``init_lr``/``final_lr`` to it (one-tenth of ``max_lr``) so the Noam schedule
    stays well-ordered.

    Args:
        hyperparameters: the user's hyperparameters dict (may include the ``hpo`` block).
        best_config: the winning config from the search.

    Returns:
        dict: hyperparameters to publish the tuned model with — no ``hpo`` block.
    """
    merged = {k: v for k, v in hyperparameters.items() if k != "hpo"}
    merged.update(best_config)
    if "max_lr" in best_config:
        merged["init_lr"] = best_config["max_lr"] / 10.0
        merged["final_lr"] = best_config["max_lr"] / 10.0
    return merged


def run_chemprop_hpo(
    train_df,
    val_df,
    base_hyperparameters: dict,
    hpo_block: dict,
    *,
    target_columns,
    smiles_column: str,
    output_dir: str | None = None,
) -> dict:
    """Phase-1 chemprop hyperparameter search; returns the phase-2 hyperparameters.

    The caller passes the already-split training frame and the held-out ``validation``
    frame (e.g. the template's ``split_validation_set`` output). Runs the search —
    single-fold trials scored on the held-out set (``holdout_mae``), or a scaffold split
    of ``train_df`` when ``val_df`` is empty (``cv_mae``) — and merges the winner into
    ``base_hyperparameters``. Writes ``best_config.json`` + ``hpo_trials.csv`` to
    ``output_dir`` when given.

    v1 scope: regression, SMILES-only features; the objective is the primary target's
    held-out MAE. Extra descriptors / bounded loss / multi-task featurization are not
    exercised during search — the phase-2 publish still uses the full template.
    """
    import json

    import pandas as pd

    from workbench.endpoints.inference import get_split_indices
    from workbench.training.hpo_harness import run_search

    if len(val_df):
        eval_df, metric, where = val_df, "holdout_mae", f"held-out validation set ({len(val_df)} rows)"
    else:
        # No designated validation rows -> a single scaffold split held out as a cv_mae
        # proxy. Scaffold is the SMILES-feature default (literature-favored; random splits
        # leak near-duplicate scaffolds across train/val), and reuses the same
        # get_split_indices the template uses for its own CV.
        strategy = base_hyperparameters.get("split_strategy", "scaffold")
        folds = get_split_indices(
            train_df,
            n_splits=1,
            strategy=strategy,
            smiles_column=smiles_column,
            test_size=0.2,
            random_state=base_hyperparameters.get("seed", 42),
            butina_cutoff=base_hyperparameters.get("butina_cutoff", 0.4),
        )
        tr_idx, val_idx = folds[0]
        eval_df = train_df.iloc[val_idx].reset_index(drop=True)
        train_df = train_df.iloc[tr_idx].reset_index(drop=True)
        metric, where = "cv_mae", f"{strategy} split ({len(val_idx)} val rows; no validation_ids)"
    print(f"[hpo] objective = {metric} on {where}; {len(train_df)} training rows")

    space = resolve_search_space(hpo_block.get("search_space"))
    backend = hpo_block.get("backend", "auto")
    trial_fn = _make_trial_fn(train_df, eval_df, base_hyperparameters, list(target_columns), smiles_column, metric)
    result = run_search(
        trial_fn,
        space,
        n_trials=hpo_block.get("n_trials", 40),
        backend=backend,
        max_parallel=hpo_block.get("max_parallel", 1),
        metric=metric,
        mode="min",
        seed=base_hyperparameters.get("seed", 42),
        resources_per_trial={"gpu": 1} if backend != "optuna" else None,
    )
    print(f"[hpo] best {metric}={result.best_value:.4f}  best_config={result.best_config}")

    if output_dir:
        with open(os.path.join(output_dir, "best_config.json"), "w") as fp:
            json.dump(
                {"metric": metric, "best_value": result.best_value, "best_config": result.best_config}, fp, indent=2
            )
        pd.DataFrame(result.trials).to_csv(os.path.join(output_dir, "hpo_trials.csv"), index=False)

    return merge_best_config(base_hyperparameters, result.best_config)


def _make_trial_fn(train_df, eval_df, base_hyperparameters, target_columns, smiles_column, metric):
    """Build the single-fold chemprop ``trial_fn`` (closes over the split data).

    Each trial builds its model via :func:`workbench.training.chemprop_core.build_mpnn_model`
    (parity with the published model), trains on the train rows using the held-out rows as
    Lightning's validation set — so per-epoch ``val_loss`` drives ASHA pruning — and returns
    the final unscaled MAE on the primary target as the objective.
    """
    import numpy as np
    import torch
    from chemprop import data, nn
    from lightning import pytorch as pl
    from rdkit import Chem

    from workbench.training.chemprop_core import build_mpnn_model

    n_targets = len(target_columns)
    num_workers = min(os.cpu_count() or 4, 8)

    def _dataset(df):
        """Featurize a frame -> (MoleculeDataset, aligned original-scale targets)."""
        smis = df[smiles_column].tolist()
        y = df[target_columns].to_numpy(dtype=float)
        dps, ys = [], []
        for i, smi in enumerate(smis):
            if Chem.MolFromSmiles(smi) is None:
                continue
            dps.append(data.MoleculeDatapoint.from_smi(smi, y=y[i].tolist()))
            ys.append(y[i])
        return data.MoleculeDataset(dps), np.asarray(ys, dtype=float)

    def trial_fn(config, report):
        hp = merge_best_config(base_hyperparameters, config)
        pl.seed_everything(hp.get("seed", 42))

        train_ds, _ = _dataset(train_df)
        eval_ds, eval_y = _dataset(eval_df)
        target_scaler = train_ds.normalize_targets()
        eval_ds.normalize_targets(target_scaler)
        output_transform = nn.UnscaleTransform.from_standard_scaler(target_scaler)
        # `val_loss` is the criterion on *normalized* targets, but the value we return is
        # unscaled MAE. Rescale per-epoch reports into the objective's units so the pruning
        # signal, the recorded per-trial values, and the final value are all one quantity —
        # otherwise pruned trials record a smaller number than completed ones and look
        # better than they are. Exact for the default MAE criterion (MAE is linear in the
        # target scale); a monotone proxy for others.
        target_scale = float(getattr(target_scaler, "scale_", [1.0])[0])
        train_ds.cache = eval_ds.cache = True

        batch_size = hp.get("batch_size", 64)
        train_loader = data.build_dataloader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        eval_loader = data.build_dataloader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        mpnn = build_mpnn_model(hp, task="regression", n_targets=n_targets, output_transform=output_transform)

        class _ReportPruning(pl.Callback):
            # Report per-epoch held-out val_loss so the harness can prune weak trials (ASHA).
            def on_validation_epoch_end(self, trainer, module):
                # Lightning's pre-training sanity-check pass fires this at epoch 0 too;
                # reporting it would duplicate step 0 with an untrained model's value.
                if trainer.sanity_checking:
                    return
                v = trainer.callback_metrics.get("val_loss")
                if v is not None:
                    report(step=trainer.current_epoch, **{metric: float(v) * target_scale})

        trainer = pl.Trainer(
            accelerator="auto",
            max_epochs=hp.get("max_epochs", 400),
            precision="16-mixed",
            logger=False,
            enable_progress_bar=False,
            callbacks=[
                _ReportPruning(),
                pl.callbacks.EarlyStopping(monitor="val_loss", patience=hp.get("patience", 40), mode="min"),
            ],
        )
        trainer.fit(mpnn, train_loader, eval_loader)

        mpnn.eval()
        with torch.inference_mode():
            preds = np.concatenate([p.numpy() for p in trainer.predict(mpnn, eval_loader)], axis=0)
        preds = preds.reshape(len(eval_y), -1)[:, 0]  # primary target; output_transform already unscaled
        return float(np.mean(np.abs(preds - eval_y[:, 0])))

    return trial_fn
