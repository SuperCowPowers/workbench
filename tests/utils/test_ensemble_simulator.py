"""Target/capture resolution for EnsembleSimulator.

A multi-target model keeps one out-of-fold capture per target, so the simulator has to be
told which one to analyze. Silently picking the primary would mean analyzing one of four
targets without saying so, which is why the multi-target-without-target case raises.

These cover resolution only — they stub the Model rather than reaching AWS.
"""

from types import SimpleNamespace

import pytest

from workbench.utils.ensemble_simulator import EnsembleSimulator

MULTI = SimpleNamespace(name="cyp-mt", target=lambda: ["cyp1a2_pic50", "cyp2d6_pic50"])
SINGLE = SimpleNamespace(name="solo", target=lambda: "solubility")
# A model built with target_column=[x] declares a one-element list. EndpointCore gates on
# len(targets) > 1, so this captures under full_cross_fold and must resolve as single-target.
SINGLE_AS_LIST = SimpleNamespace(name="solo-list", target=lambda: ["cyp2d6_pic50"])


def test_multi_target_requires_a_target():
    with pytest.raises(ValueError, match="multi-target") as err:
        EnsembleSimulator._resolve_target(MULTI, None)
    # The available targets belong in the message — that is the whole point of raising.
    assert "cyp1a2_pic50" in str(err.value) and "cyp2d6_pic50" in str(err.value)


def test_multi_target_accepts_a_declared_target():
    assert EnsembleSimulator._resolve_target(MULTI, "cyp2d6_pic50") == "cyp2d6_pic50"


def test_multi_target_rejects_an_undeclared_target():
    with pytest.raises(ValueError, match="not among"):
        EnsembleSimulator._resolve_target(MULTI, "cyp3a4_pic50")


def test_single_target_needs_no_target():
    assert EnsembleSimulator._resolve_target(SINGLE, None) == "solubility"


def test_single_target_accepts_its_own_target():
    assert EnsembleSimulator._resolve_target(SINGLE, "solubility") == "solubility"


def test_single_target_rejects_a_different_target():
    with pytest.raises(ValueError, match="single target"):
        EnsembleSimulator._resolve_target(SINGLE, "logd")


def test_one_element_list_is_single_target():
    assert EnsembleSimulator._resolve_target(SINGLE_AS_LIST, None) == "cyp2d6_pic50"
    assert EnsembleSimulator._resolve_target(SINGLE_AS_LIST, "cyp2d6_pic50") == "cyp2d6_pic50"


def test_one_element_list_rejects_a_different_target():
    with pytest.raises(ValueError, match="single target"):
        EnsembleSimulator._resolve_target(SINGLE_AS_LIST, "cyp3a4_pic50")
