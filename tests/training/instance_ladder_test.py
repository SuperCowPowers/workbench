"""Which instance ladder a training job asks for, given its framework, type, and `hpo` block.

The choice is made client-side, before the job exists, so it cannot consult the box it
will land on — these are the only rules deciding a job's device and whether a search gets
multiple GPUs.
"""

import pytest

# Workbench Imports
from workbench.core.model_types import ModelFramework, ModelType
from workbench.core.transforms.features_to_model.features_to_model import (
    FRAMEWORK_WORKLOAD,
    INSTANCE_LADDERS,
    training_workload,
)

GPU = ModelFramework.CHEMPROP
CPU = ModelFramework.XGBOOST
REG = ModelType.REGRESSOR


@pytest.mark.parametrize("hyperparameters", [None, {}, {"uq_version": "v1"}])
def test_a_plain_model_never_asks_for_the_parallel_box(hyperparameters):
    """No `hpo` block means no search, so a multi-GPU instance would sit idle."""
    assert training_workload(GPU, REG, hyperparameters) == "gpu"
    assert training_workload(CPU, REG, hyperparameters) == "cpu"


def test_an_empty_hpo_block_still_gets_a_search_box():
    """`{}` asks for a search on every default — the promise that defaults work."""
    assert training_workload(GPU, REG, {"hpo": {}}) == "gpu_hpo"
    assert training_workload(CPU, REG, {"hpo": {}}) == "cpu_hpo"


def test_a_gpu_search_gets_the_parallel_ladder():
    """Concurrency is derived from the cards, so the box has to have cards to derive from."""
    assert training_workload(GPU, REG, {"hpo": {"n_trials": 60}}) == "gpu_hpo"


@pytest.mark.parametrize("block", [{"backend": "optuna"}, {"max_parallel": 1}])
def test_a_deliberately_serial_search_stays_on_one_gpu(block):
    """Optuna is serial by construction, and max_parallel=1 says so outright."""
    assert training_workload(GPU, REG, {"hpo": block}) == "gpu"


def test_a_transformer_model_trains_on_cpu():
    """A TRANSFORMER model computes features and trains nothing, whatever its framework."""
    assert training_workload(GPU, ModelType.TRANSFORMER, None) == "cpu"


def test_tabicl_trains_on_gpu():
    """A CPU fit saves a float32 cache, about 1 GB more serving memory than CUDA's float16."""
    assert training_workload(ModelFramework.TABICL, ModelType.UQ_REGRESSOR, None) == "gpu"


def test_a_parallel_search_never_lands_on_a_single_gpu_box():
    """Every rung must have multiple cards.

    A single-card rung does not fail — it succeeds into a run that takes days and can
    exhaust host RAM on a real dataset, hours past the point where it could have failed
    cleanly. No capacity should stop the search, not shrink it.
    """
    from workbench.utils.model_utils import model_instance_info

    gpus = model_instance_info().set_index("Instance Name")["Num GPUs"]
    for instance in INSTANCE_LADDERS["gpu_hpo"]:
        assert gpus.get(instance, 0) > 1, f"{instance} is a single-GPU rung"


def test_every_framework_has_a_workload():
    """A framework missing from the map would KeyError at submit time."""
    assert set(FRAMEWORK_WORKLOAD) == set(ModelFramework)


def test_every_workload_names_a_real_ladder():
    """A returned key that isn't in the table would KeyError at submit time."""
    for framework in ModelFramework:
        for hyperparameters in [None, {"hpo": {}}]:
            assert training_workload(framework, REG, hyperparameters) in INSTANCE_LADDERS
