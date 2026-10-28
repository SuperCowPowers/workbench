"""AWS SSO renewal: run `aws sso login` for the user when their token expires

Boto refreshes SSO tokens on its own while the Identity Center session lasts. Past
that, only an interactive (browser) login renews it, so these helpers offer that login
-- yes / no / always -- instead of just naming the command. "Always" is saved to the
Workbench config as AUTO_SSO_RENEW, and from then on the login runs without asking.
"""

import os
import re
import sys
import shutil
import subprocess
from typing import Callable, Optional

import boto3
from botocore.exceptions import SSOError, TokenRetrievalError

# Workbench Imports
from workbench.utils.config_manager import ConfigManager
from workbench.utils.execution_environment import running_as_service
from workbench.utils.repl_color_utils import cprint

# Botocore's errors for an SSO token that is missing, expired, or past its refresh window
SSO_TOKEN_ERRORS = (SSOError, TokenRetrievalError)

_SECTION_RE = re.compile(r"^\s*\[\s*([^\]]+?)\s*\]\s*$")
_KEY_RE = re.compile(r"^\s*([A-Za-z0-9_]+)\s*=\s*(.*?)\s*$")


def sso_login_hint(profile: str = None) -> str:
    """The `aws sso login` command for the profile in play.

    Args:
        profile (str, optional): Profile name; falls back to the AWS_PROFILE env var.

    Returns:
        str: The login command, with no profile flag when none is known.
    """
    profile = profile or os.environ.get("AWS_PROFILE")
    return f"aws sso login --profile {profile}" if profile else "aws sso login"


def sso_renew(profile: str = None) -> bool:
    """Renew an expired SSO token with `aws sso login`, asking first unless AUTO_SSO_RENEW is set.

    Needs an interactive terminal and the AWS CLI; without either it returns False and
    the caller reports the expired token.

    Args:
        profile (str, optional): Profile to renew; falls back to the AWS_PROFILE env var.

    Returns:
        bool: True if the login ran and succeeded.
    """
    profile = profile or os.environ.get("AWS_PROFILE")
    aws_cli = shutil.which("aws")
    if aws_cli is None or running_as_service() or not (sys.stdin and sys.stdin.isatty()):
        return False

    cm = ConfigManager()
    auto = str(cm.get_config("AUTO_SSO_RENEW", False)).strip().lower() in ("true", "1", "yes")
    if auto:
        cprint("lightpurple", f"AWS SSO session expired for '{profile}'")
    else:
        answer = input(f"AWS SSO session expired for '{profile}'. Renew now? [Y]es/[n]o/[a]lways: ").strip().lower()
        if answer in ("a", "always"):
            cm.save_config("AUTO_SSO_RENEW", True)
        elif answer not in ("", "y", "yes"):
            return False

    # Best effort: a config file we can't read or replace still leaves the login to run
    try:
        migrate_legacy_sso_profile(profile or "default")
    except (OSError, UnicodeError) as e:
        cprint("darkyellow", f"Could not update the AWS config for silent refresh: {e}")

    # The CLI's own output (fallback URL, device-code advice) is shown only if the login
    # fails or is interrupted
    cprint("lightpurple", "Opening your browser to sign in...")
    command = [aws_cli, "sso", "login"] + (["--profile", profile] if profile else [])
    login = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    try:
        output, _ = login.communicate()
    except KeyboardInterrupt:
        login.kill()
        output, _ = login.communicate()
    if login.returncode != 0:
        print(output)
        return False

    # A fresh default session re-reads ~/.aws/config, so a just-migrated profile is seen
    boto3.setup_default_session()
    cprint("lightgreen", "AWS SSO renewed")
    return True


def with_sso_renewal(fn: Callable, profile: str = None):
    """Run fn(); if an expired SSO token stops it, offer a renewal and run it once more.

    Args:
        fn (Callable): Zero-argument callable to run.
        profile (str, optional): Profile to renew; falls back to the AWS_PROFILE env var.

    Returns:
        Whatever fn returns. The SSO error is re-raised when the renewal is declined or fails.
    """
    try:
        return fn()
    except SSO_TOKEN_ERRORS:
        if not sso_renew(profile):
            raise
        return fn()


def install_sso_handler() -> None:
    """Offer an SSO renewal whenever code typed at the REPL hits an expired token."""
    from IPython import get_ipython

    get_ipython().set_custom_exc(SSO_TOKEN_ERRORS, _sso_exception_handler)


def _sso_exception_handler(shell, etype, value, tb, tb_offset=None):
    """IPython custom exception handler: a renewal offer in place of the traceback."""
    if sso_renew():
        cprint("lightgreen", "Re-run the last command")
    else:
        cprint("darkyellow", f"AWS SSO session expired: renew with '{sso_login_hint()}', then re-run the last command")


def migrate_legacy_sso_profile(profile: str, config_path: Optional[str] = None) -> bool:
    """Move a profile's inline SSO settings into a shared [sso-session] block.

    Inline (legacy) SSO profiles get no refresh token, so every token expiry needs a
    browser login; sso-session profiles refresh silently for the whole Identity Center
    session. The edit swaps the profile's sso_start_url/sso_region lines for one
    sso_session line and appends the session block (or reuses one with the same start
    URL). Everything else in the file, comments and line endings included, is untouched,
    and the prior file is kept beside it as config.bak.

    Args:
        profile (str): Profile name ("default" for the [default] section).
        config_path (str, optional): AWS config file; defaults to AWS_CONFIG_FILE or ~/.aws/config.

    Returns:
        bool: True if the file was changed.
    """
    path = config_path or os.path.expanduser(os.environ.get("AWS_CONFIG_FILE", "~/.aws/config"))
    if not os.path.exists(path):
        return False
    with open(path, encoding="utf-8", newline="") as f:
        lines = f.read().splitlines(keepends=True)
    eol = "\r\n" if any(line.endswith("\r\n") for line in lines) else "\n"

    # Each section's name and its key -> (line index, value)
    sections = {}
    current = None
    for i, line in enumerate(lines):
        header = _SECTION_RE.match(line.rstrip("\r\n"))
        if header:
            current = " ".join(header.group(1).split())
            sections.setdefault(current, {})
        elif current is not None:
            kv = _KEY_RE.match(line.rstrip("\r\n"))
            if kv:
                sections[current][kv.group(1)] = (i, kv.group(2))

    target = "default" if profile == "default" else f"profile {profile}"
    keys = sections.get(target, {})
    if "sso_session" in keys or "sso_start_url" not in keys or "sso_region" not in keys:
        return False
    start_url = keys["sso_start_url"][1]
    sso_region = keys["sso_region"][1]

    # Reuse a session with the same start URL, else name one after the portal host
    sessions = {name.split(" ", 1)[1]: kv for name, kv in sections.items() if name.startswith("sso-session ")}
    session = next((name for name, kv in sessions.items() if kv.get("sso_start_url", (0, None))[1] == start_url), None)
    new_session = session is None
    if new_session:
        host = re.match(r"https?://([^./]+)", start_url)
        base = host.group(1) if host else "workbench-sso"
        session, n = base, 2
        while session in sessions:
            session, n = f"{base}-{n}", n + 1

    # Swap the start URL line for sso_session (keeping its line ending) and drop sso_region
    url_idx, region_idx = keys["sso_start_url"][0], keys["sso_region"][0]
    url_line = lines[url_idx]
    lines[url_idx] = f"sso_session = {session}" + url_line[len(url_line.rstrip("\r\n")) :]
    del lines[region_idx]

    if new_session:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += eol
        lines += [
            eol,
            f"[sso-session {session}]{eol}",
            f"sso_start_url = {start_url}{eol}",
            f"sso_region = {sso_region}{eol}",
            f"sso_registration_scopes = sso:account:access{eol}",
        ]

    shutil.copy2(path, path + ".bak")
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        f.write("".join(lines))
    shutil.copymode(path, tmp_path)
    os.replace(tmp_path, path)
    cprint(
        "lightpurple", f"Moved profile '{profile}' to [sso-session {session}] for silent refresh (backup: {path}.bak)"
    )
    return True
