"""Which instance ladder a training job asks for, given its `hpo` block.

The choice is made client-side, before the job exists, so it cannot consult the box it
will land on — these are the only rules deciding whether a search gets multiple GPUs.
"""

import pytest

# Workbench Imports
from workbench.core.transforms.features_to_model.features_to_model import INSTANCE_LADDERS, training_workload


@pytest.mark.parametrize("hyperparameters", [None, {}, {"uq_version": "v1"}])
def test_a_plain_model_never_asks_for_the_parallel_box(hyperparameters):
    """No `hpo` block means no search, so a multi-GPU instance would sit idle."""
    assert training_workload(hyperparameters, gpu_framework=True) == "gpu"
    assert training_workload(hyperparameters, gpu_framework=False) == "cpu"


def test_an_empty_hpo_block_still_gets_a_search_box():
    """`{}` asks for a search on every default — the promise that defaults work."""
    assert training_workload({"hpo": {}}, gpu_framework=True) == "gpu_parallel_hpo"
    assert training_workload({"hpo": {}}, gpu_framework=False) == "cpu_hpo"


def test_a_gpu_search_gets_the_parallel_ladder():
    """Concurrency is derived from the cards, so the box has to have cards to derive from."""
    assert training_workload({"hpo": {"n_trials": 60}}, gpu_framework=True) == "gpu_parallel_hpo"


@pytest.mark.parametrize("block", [{"backend": "optuna"}, {"max_parallel": 1}])
def test_a_deliberately_serial_search_stays_on_one_gpu(block):
    """Optuna is serial by construction, and max_parallel=1 says so outright."""
    assert training_workload({"hpo": block}, gpu_framework=True) == "gpu"


def test_every_workload_names_a_real_ladder():
    """A returned key that isn't in the table would KeyError at submit time."""
    cases = [(None, True), (None, False), ({"hpo": {}}, True), ({"hpo": {}}, False)]
    for hyperparameters, gpu in cases:
        assert training_workload(hyperparameters, gpu_framework=gpu) in INSTANCE_LADDERS
