"""Track long-running jobs so the REPL and the agent can report them.

Job kinds register here and report their own outcome: AWS Batch jobs
(``batch_utils``) and local training subprocesses (``workbench.local``). One
watcher set feeds one set of prompt lights and one completion queue, so a
finished job reaches the user the same way no matter where it ran.
"""

import atexit
import logging
import threading
import time
from collections import deque

# Workbench Imports
from workbench.utils.repl_utils import cprint_above_prompt, set_rprompt

log = logging.getLogger("workbench")

# Watchers are daemon threads, so the interpreter exits without them; the event wakes
# them out of a poll interval first rather than tearing down mid-sleep.
shutdown = threading.Event()
atexit.register(shutdown.set)

# Finished jobs waiting to be reported into an agent turn. Bounded: nothing guarantees
# anyone ever drains it.
_completed = deque(maxlen=20)

# Jobs launched this session, in launch order, for the REPL's right-prompt lights.
_watched = {}
_LIGHT_TOKENS = {"running": "Darkyellow", "completed": "Lightgreen", "failed": "Red"}

# How often a subprocess watcher checks on its child
SUBPROCESS_INTERVAL = 5


def register(name: str) -> None:
    """Mark a job as running, lighting its dot on the prompt.

    Args:
        name (str): The job name
    """
    _watched[name] = "running"


def report(row: dict, success: bool) -> None:
    """Announce a finished job and queue it for the next agent turn.

    Reporting happens twice, because the launcher may be looking at neither: a banner
    is printed to the terminal, and the outcome is queued for `drain_completed`.

    Args:
        row (dict): Job outcome with keys name, status, and optionally kind/runtime/reason
        success (bool): Did the job succeed? Job kinds have their own status vocabularies.
    """
    _watched[row["name"]] = "completed" if success else "failed"
    _completed.append(row)

    kind = row.get("kind", "Job")
    runtime = f" after {row['runtime']}" if row.get("runtime") else ""
    reason = f" -- {row['reason']}" if row.get("reason") else ""
    color = "lightgreen" if success else "red"
    cprint_above_prompt(color, f"\n{kind} {row['name']} {row['status']}{runtime}{reason}")


def drain_completed() -> list:
    """Pop every job that has finished since the last call.

    Returns:
        list[dict]: Job rows (kind, name, status, runtime, reason), oldest first.
    """
    rows = []
    while _completed:
        rows.append(_completed.popleft())
    return rows


def job_updates(prompt: str) -> str:
    """Prefix an agent turn with any jobs that finished since the last one.

    Covers every tracked job kind (AWS Batch, local training). The watcher already
    printed a banner, but that scrolls past. This is what puts the outcome in front
    of the agent so it can speak to it and go look at what the job produced.

    Args:
        prompt (str): The user's prompt for this turn.

    Returns:
        str: The prompt, preceded by one bracketed line per finished job.
    """
    rows = drain_completed()
    if not rows:
        return prompt
    updates = "\n".join(
        f"[{r.get('kind', 'Job')} update: {r['name']} {r['status']}"
        + (f" after {r['runtime']}" if r.get("runtime") else "")
        + (f" -- {r['reason']}" if r.get("reason") else "")
        + "]"
        for r in rows
    )
    return f"{updates}\n\n{prompt}"


def job_lights():
    """`Jobs [***]` -- a dot per job launched this session, for the right prompt.

    Yellow while a job runs, green when it succeeds, red when it fails. Reads the
    watchers' state, so it costs nothing per render.

    Returns:
        list[(Token, str)]: Pygments tokens, empty when nothing has been launched.
    """
    from pygments.token import Token

    if not _watched:
        return []
    dots = [(getattr(Token, _LIGHT_TOKENS[state]), "●") for state in _watched.values()]
    return [(Token.Blue, "Jobs [")] + dots + [(Token.Blue, "]")]


def install_job_lights() -> None:
    """Put this session's job lights on the right side of the REPL prompt."""
    from IPython import get_ipython

    set_rprompt(get_ipython(), job_lights)


def watch_subprocess(
    name: str,
    proc,
    kind: str = "Local job",
    log_path: str = None,
    on_finish=None,
    interval: int = SUBPROCESS_INTERVAL,
) -> threading.Thread:
    """Watch a subprocess until it exits, then report it.

    Args:
        name (str): The job name, as it should appear in reports
        proc (subprocess.Popen): The already-started child process
        kind (str): Label for the report banner (default: "Local job")
        log_path (str, optional): Child output log; its tail becomes the failure reason
        on_finish (callable, optional): Called with the exit code before reporting, so the
            job's owner can record its own outcome (durable status, artifacts)
        interval (int, optional): Seconds between checks. Defaults to SUBPROCESS_INTERVAL.

    Returns:
        threading.Thread: The started daemon thread.
    """
    register(name)
    thread = threading.Thread(
        target=_watch_subprocess,
        args=(name, proc, kind, log_path, on_finish, interval),
        daemon=True,
        name=f"job-watch-{name}",
    )
    thread.start()
    return thread


def _watch_subprocess(name: str, proc, kind: str, log_path: str, on_finish, interval: int) -> None:
    """Poll a child process until it exits, then report the outcome."""
    started = time.time()

    # Wait on the child, but stay responsive to interpreter shutdown
    while proc.poll() is None:
        if shutdown.wait(interval):
            return

    runtime = f"{time.time() - started:.0f}s"
    success = proc.returncode == 0

    if on_finish:
        try:
            on_finish(proc.returncode)
        except Exception as e:
            log.error(f"Job '{name}' finish handler failed: {e}")

    report(
        {
            "kind": kind,
            "name": name,
            "status": "COMPLETED" if success else "FAILED",
            "runtime": runtime,
            "reason": "" if success else _failure_reason(proc.returncode, log_path),
        },
        success=success,
    )


def _failure_reason(returncode: int, log_path: str) -> str:
    """Build a failure reason from the exit code plus the tail of the child's log.

    Args:
        returncode (int): The child's exit code
        log_path (str): Path to the child's output log (may be None or missing)

    Returns:
        str: A short reason string
    """
    reason = f"exit {returncode}"
    if not log_path:
        return reason
    try:
        with open(log_path, "r") as fp:
            # A long training run's log can be very large; keep a rolling tail rather
            # than reading the whole thing into memory to report three lines
            tail = deque((line.strip() for line in fp if line.strip()), maxlen=3)
    except OSError:
        return reason
    return f"{reason}: {' | '.join(tail)}" if tail else reason
