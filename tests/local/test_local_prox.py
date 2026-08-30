"""Tests for proximity on local artifacts (no AWS required).

`fs.prox()` is pure DataFrame work and runs in the quick tier. `model.prox()` needs a
trained model to delegate from, so those trains the real model script and are marked
medium alongside the rest of the local training tests.
"""

import pandas as pd
import pytest

from workbench.algorithms.dataframe.feature_space_proximity import FeatureSpaceProximity
from workbench.algorithms.dataframe.fingerprint_proximity import FingerprintProximity
from workbench.local import DataSource, ModelType, ModelFramework
from workbench.utils.chem_utils.mol_descriptors import compute_descriptors
from workbench.utils.config_manager import ConfigManager

# Enough distinct scaffolds that a scaffold-grouped 5-fold split has groups to work with
SCAFFOLDS = [
    "c1ccccc1",
    "c1ccncc1",
    "c1ccc2ccccc2c1",
    "c1ccsc1",
    "c1cc[nH]c1",
    "C1CCCCC1",
    "C1CCNCC1",
    "c1ccc(cc1)-c1ccccc1",
    "c1cnc2ccccc2n1",
    "C1CCOC1",
    "c1ccc2[nH]ccc2c1",
    "c1cnccn1",
]
SUBSTITUENTS = ["C", "CC", "CCC", "CO", "CN"]
DESCRIPTORS = ["molwt", "mollogp", "tpsa"]


@pytest.fixture(autouse=True)
def local_storage(tmp_path):
    """Point local storage at a temp directory for every test"""
    cm = ConfigManager()
    original = cm.config.get("WORKBENCH_LOCAL_PATH")
    cm.set_config("WORKBENCH_LOCAL_PATH", str(tmp_path))
    yield tmp_path
    cm.set_config("WORKBENCH_LOCAL_PATH", original)


@pytest.fixture
def feature_set():
    """A small featurized compound set with SMILES and 2D descriptors"""
    rows = [
        {"id": f"c{i}_{j}", "smiles": sub + scaffold, "logs": float((i * len(SUBSTITUENTS) + j) % 19)}
        for i, scaffold in enumerate(SCAFFOLDS)
        for j, sub in enumerate(SUBSTITUENTS)
    ]
    df = compute_descriptors(pd.DataFrame(rows))
    return DataSource(df, name="prox_data").to_features("prox_features", id_column="id")


#
# FeatureSet proximity -- the pre-model path
#


class TestFeatureSetProx:
    def test_fingerprint_space(self, feature_set):
        """Structural neighbors come back with a similarity"""
        prox = feature_set.prox("fingerprint", target="logs")
        assert isinstance(prox, FingerprintProximity)
        assert prox.space == "fingerprint"
        assert prox.id_column == "id"

        nbrs = prox.neighbors("c3_1", n_neighbors=3)
        assert set(nbrs.columns) >= {"id", "neighbor_id", "similarity", "logs"}
        # The query is its own first neighbor, at similarity 1.0
        assert nbrs.iloc[0]["neighbor_id"] == "c3_1"
        assert nbrs.iloc[0]["similarity"] == pytest.approx(1.0)

    def test_feature_space(self, feature_set):
        """Descriptor neighbors come back with a distance"""
        prox = feature_set.prox("features", feature_list=DESCRIPTORS, target="logs")
        assert isinstance(prox, FeatureSpaceProximity)
        assert prox.space == "features"
        assert "distance" in feature_set.prox("features", feature_list=DESCRIPTORS, target="logs").neighbors("c3_1")

    def test_cached_per_key(self, feature_set):
        """The same (space, feature_list, target) returns the same model"""
        first = feature_set.prox("fingerprint", target="logs")
        assert feature_set.prox("fingerprint", target="logs") is first
        assert feature_set.prox("fingerprint", target=None) is not first

    def test_invalid_space(self, feature_set):
        """An unknown space is a caller error, not a silent default"""
        with pytest.raises(ValueError, match="fingerprint"):
            feature_set.prox("euclidean")

    def test_feature_space_requires_a_feature_list(self, feature_set):
        """Feature-space proximity has nothing to measure without columns"""
        with pytest.raises(ValueError, match="feature_list"):
            feature_set.prox("features")


#
# Model proximity -- delegates to the FeatureSet it trained on
#


@pytest.mark.medium
class TestModelProx:
    @pytest.fixture
    def model(self, feature_set):
        """A trained descriptor model"""
        return feature_set.to_model(
            "prox-reg-model",
            model_type=ModelType.REGRESSOR,
            model_framework=ModelFramework.XGBOOST,
            target_column="logs",
            feature_list=DESCRIPTORS,
        )

    def test_fingerprint_space(self, model):
        """Neighbors carry the model's own target, which is what the graph plots"""
        prox = model.prox("fingerprint")
        assert isinstance(prox, FingerprintProximity)
        assert prox.id_column == "id"

        nbrs = prox.neighbors("c3_1", n_neighbors=3)
        assert set(nbrs.columns) >= {"id", "neighbor_id", "similarity", "logs"}
        assert nbrs.iloc[0]["neighbor_id"] == "c3_1"

    def test_feature_space(self, model):
        """A descriptor model has a meaningful feature space"""
        assert isinstance(model.prox("features"), FeatureSpaceProximity)

    def test_cached_per_space(self, model):
        """Repeated calls return the same model rather than rebuilding"""
        first = model.prox("fingerprint")
        assert model.prox("fingerprint") is first

    def test_invalid_space(self, model):
        """An unknown space is a caller error, not a silent default"""
        with pytest.raises(ValueError, match="fingerprint"):
            model.prox("euclidean")

    def test_structure_model_has_no_feature_space(self, model, monkeypatch):
        """A model trained on SMILES has no descriptor space to measure"""
        meta = dict(model.workbench_meta())
        meta["workbench_model_features"] = ["smiles"]
        monkeypatch.setattr(model, "workbench_meta", lambda: meta)
        assert model.prox("features") is None

    def test_missing_feature_set(self, model, monkeypatch):
        """A model whose FeatureSet is gone has nothing to build from"""
        monkeypatch.setattr(model, "parent", lambda: None)
        assert model.prox("fingerprint") is None
