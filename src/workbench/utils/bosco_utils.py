"""Support for the Bosco agent: session reports and turn-time notifications.

A session report is prose, not a transcript and not a variable dump. It records the
goal, the artifacts involved (by name, since those are re-derivable), what was
concluded, and what is still open. Reports live in the Parameter Store so any user
on the account can recall one, including someone else's.
"""

import logging
from datetime import datetime, timezone

import pandas as pd

from workbench.api import ParameterStore
from workbench.utils.aws_utils import current_user, slugify

log = logging.getLogger("workbench")

SESSION_ROOT = "/workbench/bosco/sessions"

# Reports are distilled, not dumped. Past this the content belongs in a DFStore frame
# that the report points at. The Parameter Store compresses above 4KB on its own.
MAX_REPORT_CHARS = 6000


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
        available = list_sessions(all_users=True)["session"].tolist()
        listing = ", ".join(available) if available else "none saved yet"
        return f"No session at '{path}'. Available: {listing}"
    return report


def list_sessions(all_users: bool = False) -> pd.DataFrame:
    """Saved session reports, most recently written first.

    Args:
        all_users (bool): Include other users' sessions. Defaults to False.

    Returns:
        pd.DataFrame: columns [session, modified], where session is "<user>/<name>".
    """
    prefix = SESSION_ROOT if all_users else f"{SESSION_ROOT}/{current_user()}"
    rows = [
        {"session": p["name"][len(SESSION_ROOT) + 1 :], "modified": p["modified"]}
        for p in ParameterStore().list(prefix, details=True)
    ]
    df = pd.DataFrame(rows, columns=["session", "modified"])
    return df.sort_values("modified", ascending=False, na_position="last").reset_index(drop=True)


def _age(when) -> str:
    """Compact age for a timestamp: "just now", "20m ago", "3h ago", "5d ago"."""
    if pd.isna(when):
        return "unknown"
    minutes = int((datetime.now(timezone.utc) - when).total_seconds() // 60)
    if minutes < 1:
        return "just now"
    if minutes < 60:
        return f"{minutes}m ago"
    if minutes < 60 * 24:
        return f"{minutes // 60}h ago"
    return f"{minutes // (60 * 24)}d ago"


def recent_sessions(all_users: bool = False) -> pd.DataFrame:
    """Saved sessions in a form worth showing a person.

    Same rows as ``list_sessions``, with the timestamp rendered as a relative age and
    a minute-resolution date instead of a microsecond one.

    Args:
        all_users (bool): Include other users' sessions. Defaults to False.

    Returns:
        pd.DataFrame: columns [session, saved, when]. Empty when nothing is saved.
    """
    df = list_sessions(all_users)
    if df.empty:
        return pd.DataFrame(columns=["session", "saved", "when"])
    return pd.DataFrame(
        {
            "session": df["session"],
            "saved": df["modified"].apply(_age),
            "when": df["modified"].dt.strftime("%Y-%m-%d %H:%M"),
        }
    )


def batch_updates(prompt: str) -> str:
    """Prefix a turn with any Batch jobs that finished since the last one.

    The watcher already printed a banner, but that scrolls past. This is what puts
    the outcome in front of the agent so it can speak to it and go look at what the
    job produced.

    Args:
        prompt (str): The user's prompt for this turn.

    Returns:
        str: The prompt, preceded by one bracketed line per finished job.
    """
    from workbench.utils.batch_utils import drain_completed

    rows = drain_completed()
    if not rows:
        return prompt
    updates = "\n".join(
        f"[Batch update: {r['name']} {r['status']}"
        + (f" after {r['runtime']}" if r.get("runtime") else "")
        + (f" -- {r['reason']}" if r.get("reason") else "")
        + "]"
        for r in rows
    )
    return f"{updates}\n\n{prompt}"
