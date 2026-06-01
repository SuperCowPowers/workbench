import tomllib
from pathlib import Path


def test_xgboost_source_build_dependency_is_declared_before_xgboost():
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    metadata = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    dependencies = metadata["project"]["dependencies"]

    cmake_index = next(i for i, spec in enumerate(dependencies) if spec.startswith("cmake "))
    xgboost_index = next(i for i, spec in enumerate(dependencies) if spec.startswith("xgboost "))

    assert "cmake >= 3.18" in dependencies
    assert cmake_index < xgboost_index
