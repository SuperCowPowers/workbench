"""Subprocess-based wall-clock timeout utility.

Purpose:
    RDKit's conformer generation runs C++ code that holds the Python GIL, so
    signal-based timeouts (SIGALRM) don't work — the signal handler can't run
    until Python regains control, which it won't until the C++ call returns.
    Certain strained molecules can cause the distance geometry solver to loop
    for minutes or forever.

    This module runs the conformer function in a persistent worker subprocess
    and enforces a real wall-clock timeout. If the worker hangs, it's killed
    cleanly and a fresh one is spawned on the next call.

Usage:
    from workbench.utils.chem_utils.conformer_timeout import run_with_timeout

    mol = run_with_timeout(
        generate_conformers,
        args=(mol,),
        kwargs={"n_conformers": 10, "optimize": True},
        timeout=30.0,
    )
    if mol is None:
        # Timed out or failed — caller should handle as a NaN case

Notes:
    - The worker pool is lazily initialized and PID-scoped for fork safety
      (uvicorn workers, multiprocessing parents, etc.)
    - Typical overhead: ~5-20 ms per call for pickling/IPC of the mol object
    - When a worker is killed on timeout, the next call pays ~200ms for a
      fresh worker spawn
"""

import logging
import multiprocessing as mp
import multiprocessing.pool  # noqa: F401 — needed for the mp.pool.Pool type reference
import os
from typing import Any, Callable, Optional

logger = logging.getLogger("workbench")

# Module-level persistent worker pool (PID-scoped)
_POOL: Optional["mp.pool.Pool"] = None
_POOL_PID: Optional[int] = None


def _get_pool() -> "mp.pool.Pool":
    """Get or create the persistent worker pool (PID-scoped for fork safety)."""
    global _POOL, _POOL_PID
    current_pid = os.getpid()
    if _POOL is None or _POOL_PID != current_pid:
        _POOL = mp.Pool(1)
        _POOL_PID = current_pid
    return _POOL


def _reset_pool() -> None:
    """Terminate the current worker and clear the pool."""
    global _POOL
    if _POOL is not None:
        try:
            _POOL.terminate()
        except Exception:
            pass
        _POOL = None


def run_with_timeout(
    func: Callable,
    args: tuple = (),
    kwargs: Optional[dict] = None,
    timeout: float = 30.0,
) -> Any:
    """Run func(*args, **kwargs) in a worker subprocess with a wall-clock timeout.

    Args:
        func: Function to run (must be picklable — module-level functions work)
        args: Positional arguments
        kwargs: Keyword arguments
        timeout: Max seconds to wait before killing the worker

    Returns:
        The function's return value, or None if the call timed out or raised.
    """
    if kwargs is None:
        kwargs = {}

    pool = _get_pool()
    try:
        async_result = pool.apply_async(func, args, kwargs)
        return async_result.get(timeout=timeout)
    except mp.TimeoutError:
        logger.warning(f"{func.__name__} timed out after {timeout}s — killing worker")
        _reset_pool()
        return None
    except Exception as e:
        logger.warning(f"{func.__name__} raised in worker: {e}")
        return None
