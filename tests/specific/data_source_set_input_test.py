from pathlib import Path

import pytest

from workbench.api.data_source import DataSource


class _LogRecorder:
    def __init__(self):
        self.messages = []

    def important(self, message):
        self.messages.append(message)


def _data_source_double():
    data_source = object.__new__(DataSource)
    data_source.name = "sample_data"
    data_source.log = _LogRecorder()
    data_source.upserted_meta = []
    data_source.upsert_workbench_meta = data_source.upserted_meta.append
    return data_source


def test_set_input_stores_s3_source_reference():
    data_source = _data_source_double()

    data_source.set_input("s3://bucket/raw/sample.csv")

    assert data_source.upserted_meta == [{"workbench_input": "s3://bucket/raw/sample.csv"}]


def test_set_input_accepts_pathlike_source_reference():
    data_source = _data_source_double()
    input_path = Path("raw") / "sample.csv"

    data_source.set_input(input_path)

    assert data_source.upserted_meta == [{"workbench_input": str(input_path)}]


def test_set_input_rejects_empty_source_reference():
    data_source = _data_source_double()

    with pytest.raises(ValueError, match="must not be empty"):
        data_source.set_input("")
