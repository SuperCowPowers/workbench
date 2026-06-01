from workbench.utils import model_utils


def test_hyperparameters_are_cached_by_model_artifact_uri(monkeypatch):
    calls = []

    def fake_load(model_artifact_uri):
        calls.append(model_artifact_uri)
        return {"source": model_artifact_uri, "nested": {"calls": len(calls)}}

    model_utils._load_hyperparameters_from_s3_cached.cache_clear()
    monkeypatch.setattr(model_utils, "_load_hyperparameters_from_s3_uncached", fake_load)

    try:
        first = model_utils.load_hyperparameters_from_s3("s3://bucket/model.tar.gz")
        first["nested"]["calls"] = 999

        second = model_utils.load_hyperparameters_from_s3("s3://bucket/model.tar.gz")
        third = model_utils.load_hyperparameters_from_s3("s3://bucket/other-model.tar.gz")
    finally:
        model_utils._load_hyperparameters_from_s3_cached.cache_clear()

    assert calls == ["s3://bucket/model.tar.gz", "s3://bucket/other-model.tar.gz"]
    assert second == {"source": "s3://bucket/model.tar.gz", "nested": {"calls": 1}}
    assert third == {"source": "s3://bucket/other-model.tar.gz", "nested": {"calls": 2}}
