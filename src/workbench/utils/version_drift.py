"""Compare local package versions against the versions a training image carries.

Local training uses whatever is installed in the current environment; AWS training
uses the image's pinned set. Where those differ, a model trained locally and one
trained in AWS can differ too. This reports the gap so publishing can warn.
"""

import logging
import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Union

log = logging.getLogger("workbench")

# Packages whose version can actually change a model's outputs
TRACKED = ["xgboost", "scikit-learn", "torch", "chemprop", "rdkit", "numpy", "pandas", "lightgbm"]

_PIN_RE = re.compile(r"^([A-Za-z0-9_.\-]+)==([^\s;#]+)")


def images_dir() -> Union[Path, None]:
    """Locate the sagemaker_images directory, walking up from this module.

    Only a source checkout carries the image definitions; an installed workbench
    does not ship them, so the drift check cannot run there.

    Returns:
        Union[Path, None]: The directory, or None when it isn't available
    """
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "sagemaker_images"
        if candidate.is_dir():
            return candidate
    return None


def image_lock_path(model_framework: str, stage: str = "training") -> Union[Path, None]:
    """Path to the requirements.lock for the image a framework trains on.

    Args:
        model_framework (str): The model framework value (e.g. "chemprop", "xgboost")
        stage (str): "training" or "inference" (default: "training")

    Returns:
        Union[Path, None]: The lock file path, or None when image definitions aren't available
    """
    images = images_dir()
    if images is None:
        return None
    image = "pytorch_chem" if model_framework in ("pytorch", "chemprop") else "base"
    return images / image / stage / "requirements.lock"


def parse_lock(lock_path: Union[Path, None]) -> dict:
    """Read the pinned versions out of a requirements.lock.

    Args:
        lock_path (Union[Path, None]): The lock file to read

    Returns:
        dict: package name (lowercased) -> pinned version, empty if unreadable
    """
    if lock_path is None:
        return {}
    try:
        lines = lock_path.read_text().splitlines()
    except OSError:
        log.warning(f"Could not read image lock file: {lock_path}")
        return {}

    pins = {}
    for line in lines:
        match = _PIN_RE.match(line.strip())
        if match:
            pins[match.group(1).lower()] = match.group(2)
    return pins


def base_version(raw: str) -> str:
    """Strip build/local suffixes so a CPU build compares equal to a CUDA one.

    Args:
        raw (str): A version string (e.g. "2.13.0+cu130")

    Returns:
        str: The version proper (e.g. "2.13.0")
    """
    return raw.split("+")[0]


def image_workbench_version(model_framework: str, stage: str = "training") -> str:
    """The workbench version an image installs, read from its Dockerfile.

    The images pip-install a released workbench from PyPI at a pinned version, so the
    model scripts they run import *that* workbench, not the local one. A template that
    calls something newer fails at import inside the container.

    Args:
        model_framework (str): The model framework value (e.g. "chemprop", "xgboost")
        stage (str): "training" or "inference" (default: "training")

    Returns:
        str: The pinned version, or "" if the Dockerfile can't be read
    """
    lock_path = image_lock_path(model_framework, stage)
    if lock_path is None:
        return ""
    dockerfile = lock_path.parent / "Dockerfile"
    try:
        text = dockerfile.read_text()
    except OSError:
        return ""
    match = re.search(r"^ARG WORKBENCH_VERSION=(\S+)", text, re.MULTILINE)
    return match.group(1) if match else ""


def check_workbench_drift(model_framework: str, stage: str = "training") -> dict:
    """Compare the local workbench version against the one the image installs.

    Args:
        model_framework (str): The model framework the model trains with
        stage (str): "training" or "inference" (default: "training")

    Returns:
        dict: {package, local, image} when they differ, empty dict when they match
    """
    image_version = image_workbench_version(model_framework, stage)
    if not image_version:
        return {}
    try:
        local_version = version("workbench")
    except PackageNotFoundError:
        return {}
    if base_version(local_version) == base_version(image_version):
        return {}
    return {"package": "workbench", "local": local_version, "image": image_version}


def check_drift(model_framework: str, packages: list = None) -> list[dict]:
    """Compare installed versions against the image's pinned versions.

    Args:
        model_framework (str): The model framework the model trains with
        packages (list, optional): Packages to check (defaults to TRACKED)

    Returns:
        list[dict]: One row per mismatch: {package, local, image}. Packages that are
            absent locally or unpinned in the image are skipped, not reported.
    """
    pins = parse_lock(image_lock_path(model_framework))
    if not pins:
        return []

    drift = []
    for package in packages or TRACKED:
        image_version = pins.get(package.lower())
        if not image_version:
            continue
        try:
            local_version = version(package)
        except PackageNotFoundError:
            continue
        if base_version(local_version) != base_version(image_version):
            drift.append({"package": package, "local": local_version, "image": image_version})
    return drift


def drift_summary(model_framework: str) -> str:
    """A human-readable drift report for a framework.

    Args:
        model_framework (str): The model framework the model trains with

    Returns:
        str: One line per mismatched package, "" when everything matches, or a note
            that the check could not run when image definitions aren't available
    """
    # Silence here would read as "versions match", which is the opposite of what we know
    if images_dir() is None:
        return (
            "Could not verify package versions against the training image: "
            "sagemaker_images is only present in a source checkout."
        )

    workbench_drift = check_workbench_drift(model_framework)
    drift = check_drift(model_framework)
    if not workbench_drift and not drift:
        return ""

    lines = [f"  {d['package']}: local {d['local']} vs image {d['image']}" for d in [workbench_drift] + drift if d]
    summary = "Package versions differ from the training image:\n" + "\n".join(lines)

    # The model scripts import workbench from inside the image, so this one is not a
    # subtle numerical difference -- a script calling something newer fails at import.
    if workbench_drift:
        summary += (
            "\n  The model script imports workbench from the image. If it uses anything"
            "\n  added since the image's version, training fails at import."
        )
    return summary
