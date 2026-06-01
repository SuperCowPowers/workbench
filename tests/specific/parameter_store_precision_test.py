"""Tests for compact Parameter Store serialization."""

import base64
import json
import zlib

from workbench.core.parameter_store_core import ParameterStoreCore


def _decode_compressed(value):
    payload = value[len("COMPRESSED:") :]
    return json.loads(zlib.decompress(base64.b64decode(payload)).decode("utf-8"))


def test_zero_precision_compression_rounds_float_values():
    value = [("feature_a", 0.0275759007781744), ("feature_b", 0.9999)]

    compressed = ParameterStoreCore._compress_value(value, precision=0)
    decoded = _decode_compressed(compressed)

    assert decoded == [["feature_a", 0.0], ["feature_b", 1.0]]


def test_progressive_precision_preserves_all_items_before_clipping():
    value = [(f"feature_{i:03d}", i / 123456.789) for i in range(350)]

    lower_precision_sizes = [
        len(ParameterStoreCore._compress_value(value, precision=precision))
        for precision in ParameterStoreCore._reduced_precision_steps(3)
    ]
    clipped = _decode_compressed(ParameterStoreCore._compress_value(ParameterStoreCore._clip_data(value), precision=0))

    assert len(clipped) == 100
    assert any(size <= 4096 for size in lower_precision_sizes)
