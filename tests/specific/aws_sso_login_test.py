"""Tests for AWS SSO login renewal helpers."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


class FakeLog:
    def __init__(self):
        self.messages = []

    def important(self, message):
        self.messages.append(("important", message))

    def warning(self, message):
        self.messages.append(("warning", message))


def load_aws_session_module(monkeypatch):
    """Load aws_session.py with stubs for optional AWS dependencies."""
    boto3_module = ModuleType("boto3")
    boto3_module.client = lambda *_args, **_kwargs: None
    boto3_module.Session = lambda *_args, **_kwargs: SimpleNamespace(region_name="us-east-1")
    monkeypatch.setitem(sys.modules, "boto3", boto3_module)

    botocore_module = ModuleType("botocore")
    exceptions_module = ModuleType("botocore.exceptions")
    for name in ["ClientError", "UnauthorizedSSOTokenError", "TokenRetrievalError", "SSOTokenLoadError"]:
        setattr(exceptions_module, name, type(name, (Exception,), {}))
    monkeypatch.setitem(sys.modules, "botocore", botocore_module)
    monkeypatch.setitem(sys.modules, "botocore.exceptions", exceptions_module)

    credentials_module = ModuleType("botocore.credentials")
    credentials_module.RefreshableCredentials = SimpleNamespace(create_from_metadata=lambda **_kwargs: object())
    monkeypatch.setitem(sys.modules, "botocore.credentials", credentials_module)

    session_module = ModuleType("botocore.session")
    session_module.get_session = lambda: SimpleNamespace(_credentials=None)
    monkeypatch.setitem(sys.modules, "botocore.session", session_module)

    config_module = ModuleType("workbench.utils.config_manager")
    config_module.ConfigManager = object
    monkeypatch.setitem(sys.modules, "workbench.utils.config_manager", config_module)

    env_module = ModuleType("workbench.utils.execution_environment")
    env_module.running_as_service = lambda: False
    monkeypatch.setitem(sys.modules, "workbench.utils.execution_environment", env_module)

    ipython_module = ModuleType("workbench.utils.ipython_utils")
    ipython_module.is_running_in_ipython = lambda: False
    ipython_module.display_error_and_raise = lambda message: (_ for _ in ()).throw(RuntimeError(message))
    monkeypatch.setitem(sys.modules, "workbench.utils.ipython_utils", ipython_module)

    module_path = Path(__file__).parents[2] / "src" / "workbench" / "core" / "cloud_platform" / "aws" / "aws_session.py"
    spec = importlib.util.spec_from_file_location("aws_session_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_session(module, profile="dev"):
    session = module.AWSSession.__new__(module.AWSSession)
    session.profile = profile
    session.log = FakeLog()
    return session


def test_renew_sso_login_runs_aws_cli(monkeypatch):
    module = load_aws_session_module(monkeypatch)
    session = make_session(module)
    calls = []

    monkeypatch.setattr(module.shutil, "which", lambda command: "aws.exe" if command == "aws" else None)
    monkeypatch.setattr(
        module.subprocess, "run", lambda command, check: calls.append((command, check)) or SimpleNamespace(returncode=0)
    )

    assert session.renew_sso_login() is True
    assert calls == [(["aws", "sso", "login", "--profile", "dev"], False)]


def test_renew_sso_login_skips_without_profile(monkeypatch):
    module = load_aws_session_module(monkeypatch)
    session = make_session(module, profile=None)

    monkeypatch.setattr(module.shutil, "which", lambda _command: "aws.exe")

    assert session.renew_sso_login() is False


def test_renew_sso_login_skips_without_aws_cli(monkeypatch):
    module = load_aws_session_module(monkeypatch)
    session = make_session(module)

    monkeypatch.setattr(module.shutil, "which", lambda _command: None)

    assert session.renew_sso_login() is False
    assert ("warning", "AWS CLI not found; cannot open AWS SSO login automatically.") in session.log.messages
