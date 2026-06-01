import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


def install_module(name, **attrs):
    module = ModuleType(name)
    for attr_name, value in attrs.items():
        setattr(module, attr_name, value)
    sys.modules[name] = module
    return module


def load_model_core_module():
    stub_names = [
        "botocore",
        "botocore.exceptions",
        "pandas",
        "awswrangler",
        "awswrangler.exceptions",
        "sagemaker",
        "sagemaker.core",
        "sagemaker.core.resources",
        "workbench",
        "workbench.core",
        "workbench.core.artifacts",
        "workbench.core.artifacts.artifact",
        "workbench.utils",
        "workbench.utils.aws_utils",
        "workbench.utils.metrics_utils",
        "workbench.utils.s3_utils",
        "workbench.utils.shap_utils",
        "workbench.utils.deprecated_utils",
        "workbench.utils.model_utils",
    ]
    missing = object()
    original_modules = {name: sys.modules.get(name, missing) for name in stub_names}

    class ClientError(Exception):
        pass

    class Artifact:
        pass

    def identity_decorator(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    exceptions = SimpleNamespace(ClientError=ClientError, NoFilesFound=type("NoFilesFound", (Exception,), {}))
    botocore = install_module("botocore", exceptions=exceptions)
    install_module("botocore.exceptions", ClientError=ClientError)
    install_module("pandas", DataFrame=lambda *args, **kwargs: None)
    install_module("awswrangler", s3=SimpleNamespace())
    install_module("awswrangler.exceptions", NoFilesFound=exceptions.NoFilesFound)
    install_module("sagemaker", core=SimpleNamespace())
    install_module("sagemaker.core", resources=SimpleNamespace(), shapes=SimpleNamespace())
    install_module("sagemaker.core.resources", TrainingJob=object, ModelPackageGroup=object)
    install_module("workbench")
    install_module("workbench.core")
    install_module("workbench.core.artifacts")
    install_module("workbench.core.artifacts.artifact", Artifact=Artifact)
    install_module("workbench.utils")
    install_module(
        "workbench.utils.aws_utils", newest_path=lambda *args, **kwargs: None, pull_s3_data=lambda *a, **k: None
    )
    install_module(
        "workbench.utils.metrics_utils", reorder_cm_df=lambda df, labels: df, reorder_metrics_df=lambda df, labels: df
    )
    install_module("workbench.utils.s3_utils", compute_s3_object_hash=lambda *args, **kwargs: None)
    install_module(
        "workbench.utils.shap_utils",
        get_shap_importance=lambda *a, **k: None,
        get_shap_values=lambda *a, **k: None,
        get_shap_feature_values=lambda *a, **k: None,
    )
    install_module("workbench.utils.deprecated_utils", deprecated=identity_decorator)
    install_module(
        "workbench.utils.model_utils",
        published_proximity_model=lambda *a, **k: None,
        get_model_hyperparameters=lambda *a, **k: None,
    )

    model_core_path = Path(__file__).parents[2] / "src" / "workbench" / "core" / "artifacts" / "model_core.py"
    spec = importlib.util.spec_from_file_location("model_core_under_test", model_core_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        assert sys.modules["botocore"] is botocore
    finally:
        for name, original in original_modules.items():
            if original is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
    return module


MODEL_CORE = load_model_core_module()
CLASS_LABELS_META_KEY = MODEL_CORE.CLASS_LABELS_META_KEY
LEGACY_CLASS_LABELS_META_KEY = MODEL_CORE.LEGACY_CLASS_LABELS_META_KEY
ModelCore = MODEL_CORE.ModelCore
ModelType = MODEL_CORE.ModelType


def model_core_stub(model_type=ModelType.CLASSIFIER, metadata=None):
    model = object.__new__(ModelCore)
    model.model_type = model_type
    model.model_name = "test-model"
    model.log = SimpleNamespace(error=lambda *args, **kwargs: None)
    model._metadata = {} if metadata is None else dict(metadata)
    model._reorder_calls = []
    model.workbench_meta = lambda: model._metadata
    model.upsert_workbench_meta = lambda updates: model._metadata.update(updates)
    model._reorder_class_artifacts = lambda labels: model._reorder_calls.append(labels)
    return model


def test_set_class_labels_updates_new_and_legacy_metadata():
    model = model_core_stub()

    model.set_class_labels(["low", "medium", "high"])

    assert model.workbench_meta()[CLASS_LABELS_META_KEY] == ["low", "medium", "high"]
    assert model.workbench_meta()[LEGACY_CLASS_LABELS_META_KEY] == ["low", "medium", "high"]
    assert model.class_labels() == ["low", "medium", "high"]
    assert model.labels() == ["low", "medium", "high"]
    assert model._reorder_calls == [["low", "medium", "high"]]


def test_labels_reads_legacy_metadata_for_existing_models():
    model = model_core_stub(metadata={LEGACY_CLASS_LABELS_META_KEY: ["spam", "ham"]})

    assert model.labels() == ["spam", "ham"]


def test_labels_are_classifier_only():
    model = model_core_stub(model_type=ModelType.REGRESSOR)

    model.set_class_labels(["ignored"])

    assert model.labels() is None
    assert CLASS_LABELS_META_KEY not in model.workbench_meta()
    assert LEGACY_CLASS_LABELS_META_KEY not in model.workbench_meta()
