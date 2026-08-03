"""Bosco session reports: a distilled summary of where a session ended up.

A session report is prose, not a transcript and not a variable dump. It records the
goal, the artifacts involved (by name, since those are re-derivable), what was
concluded, and what is still open. Reports live in the Parameter Store so any user
on the account can recall one, including someone else's.
"""

import logging

from workbench.api import ParameterStore
from workbench.utils.aws_utils import current_user, slugify

log = logging.getLogger("workbench")

SESSION_ROOT = "/workbench/bosco/sessions"

# Reports are distilled, not dumped. Past this the content belongs in a DFStore frame
# that the report points at. The Parameter Store compresses above 4KB on its own.
MAX_REPORT_CHARS = 5000


def session_path(name: str, user: str = None) -> str:
    """Full Parameter Store path for a session report.

    Args:
        name (str): Session name, or "<user>/<name>" to address someone else's.
        user (str): Owner of the session. Defaults to the current user.

    Returns:
        str: e.g. "/workbench/bosco/sessions/briford/logd-cleanup"
    """
    if "/" in name:
        user, name = name.rsplit("/", 1)
    return f"{SESSION_ROOT}/{slugify(user or current_user())}/{slugify(name)}"


def save_session(name: str, report: str, user: str = None) -> str:
    """Save a session report.

    Args:
        name (str): Session name, e.g. "logd-cleanup".
        report (str): The report markdown.
        user (str): Owner. Defaults to the current user.

    Returns:
        str: The path it was saved to.

    Raises:
        ValueError: If the report exceeds MAX_REPORT_CHARS.
    """
    if len(report) > MAX_REPORT_CHARS:
        raise ValueError(
            f"Report is {len(report)} chars, over the {MAX_REPORT_CHARS} limit. "
            "Name artifacts instead of restating them, and park bulk findings in a DFStore frame."
        )
    path = session_path(name, user)
    ParameterStore().upsert(path, report)
    return path


def read_session(name: str, user: str = None) -> str:
    """Read a session report back.

    Args:
        name (str): Session name, or "<user>/<name>" for someone else's.
        user (str): Owner. Defaults to the current user.

    Returns:
        str: The report markdown, or a message naming what is available.
    """
    path = session_path(name, user)
    report = ParameterStore().get(path)
    if report is None:
        available = list_sessions(all_users=True)
        listing = ", ".join(available) if available else "none saved yet"
        return f"No session at '{path}'. Available: {listing}"
    return report


def list_sessions(all_users: bool = False) -> list:
    """Saved session reports, as "<user>/<name>".

    Args:
        all_users (bool): Include other users' sessions. Defaults to False.

    Returns:
        list[str]: e.g. ["briford/logd-cleanup", "alice/pxr-hpo"]
    """
    prefix = SESSION_ROOT if all_users else f"{SESSION_ROOT}/{current_user()}"
    return [p[len(SESSION_ROOT) + 1 :] for p in ParameterStore().list(prefix)]
