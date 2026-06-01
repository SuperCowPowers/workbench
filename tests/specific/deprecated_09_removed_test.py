"""Guard that Workbench 0.9 deprecated APIs have been removed."""

from pathlib import Path


def test_no_09_deprecated_api_surface_remains():
    repo_root = Path(__file__).resolve().parents[2]
    src_root = repo_root / "src" / "workbench"
    offenders = []

    for path in src_root.rglob("*.py"):
        if path.name == "deprecated_utils.py":
            continue
        text = path.read_text(encoding="utf-8")
        if '@deprecated(version="0.9")' in text or "@deprecated(version='0.9')" in text:
            offenders.append(str(path.relative_to(repo_root)))

    assert offenders == []
