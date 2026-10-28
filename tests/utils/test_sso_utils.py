"""Tests for workbench.utils.sso_utils: the SSO renewal prompt and the legacy-profile migration.

The migration edits a user's own ~/.aws/config, so the invariants guarded here are that
it changes only the target profile's SSO lines, keeps everything else byte-for-byte
(comments, CRLF endings), leaves a .bak, and produces a file botocore reads as an
sso-session profile.
"""

import json
from types import SimpleNamespace

import pytest
from botocore.configloader import load_config
from botocore.exceptions import TokenRetrievalError, UnauthorizedSSOTokenError

from workbench.utils import sso_utils
from workbench.utils.config_manager import ConfigManager

LEGACY = """\
# keep this comment
[profile idb-dev]
sso_start_url = https://ideayabio.awsapps.com/start
sso_region = us-west-2
sso_account_id = 111111111111
sso_role_name = Developer
region = us-west-2

[profile idb-dev-admin]
sso_start_url = https://ideayabio.awsapps.com/start
sso_region = us-west-2
sso_account_id = 111111111111
sso_role_name = Admin
"""


def write_config(tmp_path, text: str, newline: str = "\n") -> str:
    path = tmp_path / "config"
    path.write_bytes(text.replace("\n", newline).encode("utf-8"))
    return str(path)


def test_migrate_legacy_profile(tmp_path):
    path = write_config(tmp_path, LEGACY)
    assert sso_utils.migrate_legacy_sso_profile("idb-dev", path)

    config = load_config(path)
    assert config["profiles"]["idb-dev"] == {
        "sso_session": "ideayabio",
        "sso_account_id": "111111111111",
        "sso_role_name": "Developer",
        "region": "us-west-2",
    }
    assert config["sso_sessions"]["ideayabio"] == {
        "sso_start_url": "https://ideayabio.awsapps.com/start",
        "sso_region": "us-west-2",
        "sso_registration_scopes": "sso:account:access",
    }

    # Only the target profile moved; the sibling and the comment are untouched
    text = open(path, encoding="utf-8").read()
    assert text.startswith("# keep this comment\n")
    assert "sso_start_url" in config["profiles"]["idb-dev-admin"]
    assert open(path + ".bak", encoding="utf-8").read() == LEGACY


def test_migrate_reuses_matching_session(tmp_path):
    path = write_config(tmp_path, LEGACY)
    sso_utils.migrate_legacy_sso_profile("idb-dev", path)
    assert sso_utils.migrate_legacy_sso_profile("idb-dev-admin", path)

    text = open(path, encoding="utf-8").read()
    assert text.count("[sso-session") == 1
    assert load_config(path)["profiles"]["idb-dev-admin"]["sso_session"] == "ideayabio"


def test_migrate_session_name_collision(tmp_path):
    other = "[sso-session ideayabio]\nsso_start_url = https://other.example.com/start\nsso_region = us-east-1\n\n"
    path = write_config(tmp_path, other + LEGACY)
    sso_utils.migrate_legacy_sso_profile("idb-dev", path)
    assert load_config(path)["profiles"]["idb-dev"]["sso_session"] == "ideayabio-2"


def test_migrate_keeps_crlf(tmp_path):
    path = write_config(tmp_path, LEGACY, newline="\r\n")
    sso_utils.migrate_legacy_sso_profile("idb-dev", path)

    raw = open(path, "rb").read()
    assert b"\n" not in raw.replace(b"\r\n", b"")
    assert load_config(path)["profiles"]["idb-dev"]["sso_session"] == "ideayabio"


@pytest.mark.parametrize(
    "text",
    [
        "[profile scp]\nsso_session = scp\nsso_account_id = 1\n\n[sso-session scp]\nsso_start_url = https://x/start\n",
        "[profile keys]\naws_access_key_id = AKIA\n",
    ],
)
def test_migrate_leaves_other_profiles_alone(tmp_path, text):
    path = write_config(tmp_path, text)
    profile = text.split("]")[0].split()[-1]
    assert not sso_utils.migrate_legacy_sso_profile(profile, path)
    assert open(path, encoding="utf-8").read() == text
    assert not (tmp_path / "config.bak").exists()


class FakeLogin:
    """Stands in for the `aws sso login` process, as the world says it should behave."""

    def __init__(self, world, command, **kwargs):
        world.logins.append(command)
        self.world = world
        self.returncode = None

    def communicate(self):
        if self.world.interrupt:
            self.world.interrupt = False
            raise KeyboardInterrupt
        if self.returncode is None:
            self.returncode = self.world.returncode
        return CLI_OUTPUT, None

    def kill(self):
        self.returncode = -9


CLI_OUTPUT = "If the browser does not open, open the following URL: https://oidc.example/authorize"


@pytest.fixture
def renew_world(monkeypatch):
    """A terminal with the AWS CLI on PATH, where `aws sso login` is recorded rather than run."""
    world = SimpleNamespace(answers=[], logins=[], saved={}, auto=False, returncode=0, interrupt=False)
    monkeypatch.setattr(sso_utils.shutil, "which", lambda name: "/usr/local/bin/aws")
    monkeypatch.setattr(sso_utils, "running_as_service", lambda: False)
    monkeypatch.setattr(sso_utils.sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr("builtins.input", lambda prompt: world.answers.pop(0))
    monkeypatch.setattr(sso_utils, "migrate_legacy_sso_profile", lambda profile: False)
    monkeypatch.setattr(sso_utils.subprocess, "Popen", lambda command, **kwargs: FakeLogin(world, command))
    monkeypatch.setattr(sso_utils.boto3, "setup_default_session", lambda: None)
    monkeypatch.setattr(
        ConfigManager,
        "get_config",
        lambda self, key, default=None: world.auto if key == "AUTO_SSO_RENEW" else default,
    )
    monkeypatch.setattr(ConfigManager, "save_config", lambda self, key, value: world.saved.update({key: value}))
    return world


@pytest.mark.parametrize("answer", ["", "y", "YES"])
def test_renew_yes(renew_world, answer, capsys):
    renew_world.answers = [answer]
    assert sso_utils.sso_renew("idb-dev")
    assert renew_world.logins == [["/usr/local/bin/aws", "sso", "login", "--profile", "idb-dev"]]
    assert renew_world.saved == {}
    assert CLI_OUTPUT not in capsys.readouterr().out


@pytest.mark.parametrize("returncode, interrupt", [(1, False), (0, True)])
def test_renew_failure_shows_cli_output(renew_world, returncode, interrupt, capsys):
    renew_world.answers = ["y"]
    renew_world.returncode, renew_world.interrupt = returncode, interrupt
    assert not sso_utils.sso_renew("idb-dev")
    assert CLI_OUTPUT in capsys.readouterr().out


def test_renew_no(renew_world):
    renew_world.answers = ["n"]
    assert not sso_utils.sso_renew("idb-dev")
    assert renew_world.logins == []


def test_renew_always_saves_flag(renew_world):
    renew_world.answers = ["a"]
    assert sso_utils.sso_renew("idb-dev")
    assert renew_world.saved == {"AUTO_SSO_RENEW": True}


def test_renew_auto_skips_prompt(renew_world):
    renew_world.auto = True
    assert sso_utils.sso_renew("idb-dev")  # input() would raise IndexError on the empty answers list
    assert len(renew_world.logins) == 1


def test_renew_needs_a_terminal(renew_world, monkeypatch):
    monkeypatch.setattr(sso_utils.sys.stdin, "isatty", lambda: False, raising=False)
    assert not sso_utils.sso_renew("idb-dev")
    assert renew_world.logins == []


def test_with_sso_renewal_retries_once(monkeypatch):
    monkeypatch.setattr(sso_utils, "sso_renew", lambda profile=None: True)
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise TokenRetrievalError(provider="sso", error_msg="expired")
        return "ok"

    assert sso_utils.with_sso_renewal(flaky) == "ok"
    assert len(calls) == 2


def test_with_sso_renewal_reraises_when_declined(monkeypatch):
    monkeypatch.setattr(sso_utils, "sso_renew", lambda profile=None: False)

    def expired():
        raise UnauthorizedSSOTokenError()

    with pytest.raises(UnauthorizedSSOTokenError):
        sso_utils.with_sso_renewal(expired)


def test_save_config_writes_only_that_key(tmp_path, monkeypatch):
    site = tmp_path / "site.json"
    site.write_text('{"WORKBENCH_BUCKET": "bucket", "AWS_PROFILE": "idb-dev"}', encoding="utf-8")
    cm = ConfigManager()
    monkeypatch.setattr(cm, "config", {"WORKBENCH_BUCKET": "bucket", "FROM_ENV": "not-on-disk"})
    monkeypatch.setattr(cm, "site_config_path", str(site))
    monkeypatch.setattr(cm, "using_default_config", False)

    cm.save_config("AUTO_SSO_RENEW", True)
    assert cm.config["AUTO_SSO_RENEW"] is True
    assert json.loads(site.read_text(encoding="utf-8")) == {
        "WORKBENCH_BUCKET": "bucket",
        "AWS_PROFILE": "idb-dev",
        "AUTO_SSO_RENEW": True,
    }


def test_renew_without_stdin(renew_world, monkeypatch):
    monkeypatch.setattr(sso_utils.sys, "stdin", None)
    assert not sso_utils.sso_renew("idb-dev")


def test_renew_survives_failed_migration(renew_world, monkeypatch):
    def locked(profile):
        raise PermissionError("config is open in another program")

    monkeypatch.setattr(sso_utils, "migrate_legacy_sso_profile", locked)
    renew_world.answers = ["y"]
    assert sso_utils.sso_renew("idb-dev")
    assert len(renew_world.logins) == 1


def test_role_refresh_passes_sso_errors_through(monkeypatch):
    """The refresh callback must not wrap SSO errors, or the REPL hook and Bosco never see them."""
    from workbench.core.cloud_platform.aws import aws_session

    class ExpiredSTS:
        def assume_role(self, **kwargs):
            raise TokenRetrievalError(provider="sso", error_msg="Token has expired and refresh failed")

    monkeypatch.setattr(aws_session.boto3, "client", lambda service: ExpiredSTS())
    fake_session = SimpleNamespace(get_workbench_execution_role_arn=lambda: "arn:aws:iam::111111111111:role/Workbench")
    with pytest.raises(TokenRetrievalError):
        aws_session.AWSSession._assume_workbench_role_session_credentials(fake_session)
