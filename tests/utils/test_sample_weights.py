"""`sample_weights` must reach the loss in every framework, not just XGBoost.

`Model.to_model(sample_weights=...)` builds a training view carrying a `sample_weight`
column. XGBoost consumed it; chemprop and PyTorch read the same view and dropped it, so
passing weights to those was a silent no-op — the model trained unweighted and nothing said
so. These pin the wiring at the point where the weight meets the loss.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_chemprop_datapoints_carry_the_weight():
    from workbench.endpoints.chemprop_utils import create_molecule_datapoints

    smis = ["CCO", "c1ccccc1", "CC(=O)O"]
    y = np.array([[1.0], [2.0], [3.0]])

    unweighted, _ = create_molecule_datapoints(smis, y)
    assert [d.weight for d in unweighted] == [1.0, 1.0, 1.0]

    weighted, _ = create_molecule_datapoints(smis, y, sample_weight=np.array([1.0, 5.0, 0.5]))
    assert [d.weight for d in weighted] == [1.0, 5.0, 0.5]


def test_chemprop_criterion_honours_a_per_row_weight():
    from chemprop.nn.metrics import MAE

    preds = torch.tensor([[2.0], [2.0], [2.0]])
    targets = torch.tensor([[1.0], [2.0], [3.0]])
    mask = torch.ones_like(targets, dtype=torch.bool)
    zeros = torch.zeros_like(mask)

    losses = []
    for w in (torch.tensor([1.0, 1.0, 1.0]), torch.tensor([1.0, 1.0, 9.0])):
        crit = MAE()
        crit.update(preds, targets, mask, w, zeros, zeros)
        losses.append(crit.compute().item())
    assert losses[1] > losses[0]


def _fit_subgroup_error(train_sample_weight, seed=0):
    """Train a tiny MLP where 50 rows carry a distinctive offset; report error on those rows."""
    from workbench.endpoints.pytorch_utils import create_model, train_model

    torch.manual_seed(seed)
    np.random.seed(seed)
    n = 400
    x = torch.linspace(-2, 2, n).reshape(-1, 1)
    y = (3.0 * x).clone()
    y[:50] += 10.0
    cat = torch.zeros(n, 0, dtype=torch.long)
    model = create_model(
        n_continuous=1,
        categorical_cardinalities=[],
        hidden_layers=[16],
        n_outputs=1,
        task="regression",
        dropout=0.0,
    )
    model, _ = train_model(
        model,
        x,
        cat,
        y,
        x,
        cat,
        y,
        task="regression",
        max_epochs=60,
        patience=60,
        batch_size=64,
        learning_rate=0.02,
        verbose=False,
        train_sample_weight=train_sample_weight,
    )
    with torch.no_grad():
        return float((model(x, None)[:50] - y[:50]).abs().mean())


def test_pytorch_uniform_weights_change_nothing():
    # The weighted reduction must collapse to the plain mean, or every existing model moves.
    assert _fit_subgroup_error(None) == pytest.approx(_fit_subgroup_error(torch.ones(400)), abs=1e-9)


def test_pytorch_upweighting_pulls_the_fit_toward_the_subgroup():
    baseline = _fit_subgroup_error(None)
    upweighted = _fit_subgroup_error(torch.cat([torch.full((50,), 20.0), torch.ones(350)]))
    assert upweighted < baseline / 2


def test_every_framework_trainer_accepts_the_argument():
    """A silent no-op is the failure this guards: the argument must exist on all three."""
    import inspect

    from workbench.training.chemprop_core import train_chemprop_fold
    from workbench.training.pytorch_core import train_pytorch_fold
    from workbench.training.xgb_core import train_xgb_fold

    assert "train_sample_weight" in inspect.signature(train_chemprop_fold).parameters
    assert "train_sample_weight" in inspect.signature(train_pytorch_fold).parameters
    assert "sample_weight" in inspect.signature(train_xgb_fold).parameters
