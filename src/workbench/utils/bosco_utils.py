"""Support for the Bosco agent: session reports and turn-time notifications.

A session report is prose, not a transcript and not a variable dump. It records the
goal, the artifacts involved (by name, since those are re-derivable), what was
concluded, and what is still open. Reports live in the Parameter Store: on an AWS
account any user can recall one, including someone else's; in local mode they are
files on the user's own disk.

The last session is the implicit counterpart: a rolling record of the user's recent
prompts and Bosco's replies, rewritten after every turn so the next REPL can pick up
where this one stopped.
"""

import json
import logging
from datetime import datetime, timezone
from functools import lru_cache

import pandas as pd
from botocore.exceptions import BotoCoreError, ClientError

from workbench.utils.aws_utils import current_user, slugify
from workbench.utils.config_manager import ConfigManager
from workbench.utils.log_utils import silence_logs
from workbench.utils.repl_color_utils import cprint, render_markdown

log = logging.getLogger("workbench")

SESSION_ROOT = "/workbench/bosco/sessions"

# Outside SESSION_ROOT, so session listings show only reports someone chose to save
LAST_SESSION_ROOT = "/workbench/bosco/last_session"

# Record size, as JSON characters. The store caps the compressed, base64-encoded value
# at 4KB; this leaves room for text that compresses poorly (SMILES, ids).
LAST_SESSION_CHARS = 5000
PROMPT_CHARS = 500
REPLY_CHARS = 800

# What a failed read or write of the record can raise: AWS (including an expired SSO
# token), the local store's disk, or a value the store can't fit
_STORE_ERRORS = (BotoCoreError, ClientError, OSError, ValueError)


def param_store():
    """The parameter store backing session reports.

    An AWS account gets the shared store, so a report is recallable by the whole
    team. Local mode has no account to share through, so reports stay on disk.
    """
    if ConfigManager().config_okay():
        from workbench.api import ParameterStore

        return ParameterStore()
    from workbench.local import ParameterStore as LocalParameterStore

    return LocalParameterStore()


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
        ValueError: If the report is too large for the store to hold.
    """
    path = session_path(name, user)
    try:
        param_store().upsert(path, report)
    except ValueError as e:
        # The store owns the size limit; it reports the encoded byte count in the cause
        raise ValueError(
            f"Report is too large to store ({len(report)} chars): distill it, and name artifacts "
            f"instead of restating them. {e}"
        ) from e
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
    report = param_store().get(path)
    if report is None:
        available = list_sessions(all_users=True)["session"].tolist()
        listing = ", ".join(available) if available else "none saved yet"
        return f"No session at '{path}'. Available: {listing}"
    return report


def show_session(name: str = None) -> None:
    """Print a session report as rendered markdown.

    Args:
        name (str): Session name, or "<user>/<name>" for someone else's. Defaults to
            the most recently saved session (any user).
    """
    if name is None:
        recent = list_sessions(all_users=True)
        if recent.empty:
            cprint("darkyellow", "No sessions saved yet.")
            return
        name = recent.iloc[0]["session"]
    cprint("lightpurple", f"\n{name}")
    render_markdown(read_session(name))


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
        for p in param_store().list(prefix, details=True)
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


def last_session_path(user: str = None) -> str:
    """Parameter Store path for a user's last-session record.

    Args:
        user (str): Owner. Defaults to the current user.

    Returns:
        str: e.g. "/workbench/bosco/last_session/briford"
    """
    return f"{LAST_SESSION_ROOT}/{slugify(user or current_user())}"


def exchange(prompt: str, reply: str) -> dict:
    """One prompt/reply pair, each clipped so a single long one can't crowd out the rest."""
    return {"prompt": prompt[:PROMPT_CHARS], "reply": reply[:REPLY_CHARS]}


def save_last_session(exchanges: list) -> None:
    """Rewrite the user's last-session record from this REPL's exchanges.

    Drops the oldest exchanges from ``exchanges`` in place until the record fits, so the
    list stays bounded however long the REPL runs. Best effort: a failed write (an
    expired token, a store hiccup) never interrupts the user.

    Args:
        exchanges (list): ``exchange()`` dicts, oldest first.
    """
    record = {"saved": datetime.now(timezone.utc).isoformat(timespec="seconds"), "exchanges": exchanges}
    while len(exchanges) > 1 and len(json.dumps(record)) > LAST_SESSION_CHARS:
        exchanges.pop(0)
    try:
        with silence_logs():
            param_store().upsert(last_session_path(), record)
    except _STORE_ERRORS as e:
        log.debug(f"Last session not saved: {e}")


@lru_cache(maxsize=1)
def previous_session() -> dict:
    """The last-session record as it stood when this process first asked.

    Cached, so once this REPL starts rewriting the record, the previous session is still
    what comes back.

    Returns:
        dict: {"saved": ISO timestamp, "exchanges": [...]}, or None when there is none.
    """
    try:
        with silence_logs():
            return param_store().get(last_session_path(), warn=False)
    except _STORE_ERRORS as e:
        log.debug(f"Last session not read: {e}")
        return None


def previous_session_age() -> str:
    """How long ago the previous session's record was saved ("3h ago"), or None without one."""
    record = previous_session()
    return _age(pd.Timestamp(record["saved"])) if record else None


def read_last_session() -> str:
    """The previous session's prompts and replies, as markdown for Bosco to read.

    Returns:
        str: The exchanges oldest first, or a note that there is no previous session.
    """
    record = previous_session()
    if not record or not record.get("exchanges"):
        return "No previous session recorded."
    lines = [f"Previous session, last saved {previous_session_age()} (replies clipped):"]
    for ex in record["exchanges"]:
        lines += ["", f"**User:** {ex['prompt']}", f"**Bosco:** {ex['reply'] or '(no reply)'}"]
    return "\n".join(lines)
