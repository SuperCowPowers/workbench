"""Tests for decompress_features: bitstrings, count vectors, and float vectors (no AWS needed)"""

import numpy as np
import pandas as pd

from workbench.endpoints.inference import decompress_features


def test_bitstring():
    df = pd.DataFrame({"fp": ["1010", "0111"], "x": [1.0, 2.0]})
    out, features = decompress_features(df, ["fp", "x"], ["fp"])
    assert features == ["x", "fp_0", "fp_1", "fp_2", "fp_3"]
    assert out["fp_0"].dtype == np.uint8
    assert out[["fp_0", "fp_1", "fp_2", "fp_3"]].to_numpy().tolist() == [[1, 0, 1, 0], [0, 1, 1, 1]]


def test_count_vector():
    df = pd.DataFrame({"fingerprint": ["0,3,0", "5,0,1"]})
    out, features = decompress_features(df, ["fingerprint"], ["fingerprint"])
    assert features == ["fin_0", "fin_1", "fin_2"]
    assert out["fin_0"].dtype == np.uint8
    assert out.to_numpy().tolist() == [[0, 3, 0], [5, 0, 1]]


def test_float_vector_with_missing_row():
    """Embeddings: negative/continuous values parse as float32, and a missing row is all NaN"""
    vectors = np.array([[0.0123, -1.5, 2.25e-3], [-0.5, 0.0, 1.0]], dtype=np.float32)
    packed = [",".join(f"{v:.7g}" for v in row) for row in vectors]
    df = pd.DataFrame({"monroe": [packed[0], None, packed[1]], "lfc": [0.1, 0.2, 0.3]})
    out, features = decompress_features(df, ["monroe", "lfc"], ["monroe"])
    assert features == ["lfc", "mon_0", "mon_1", "mon_2"]
    assert out["mon_0"].dtype == np.float32
    values = out[["mon_0", "mon_1", "mon_2"]].to_numpy()
    np.testing.assert_allclose(values[[0, 2]], vectors, rtol=1e-6)
    assert np.isnan(values[1]).all()
