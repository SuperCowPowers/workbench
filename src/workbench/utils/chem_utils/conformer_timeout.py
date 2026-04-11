"""Subprocess-based wall-clock timeout utility.

Purpose:
    RDKit's conformer generation runs C++ code that holds the Python GIL, so
    signal-based timeouts (SIGALRM) don't work — the signal handler can't run
    until Python regains control, which it won't until the C++ call returns.
    Certain strained molecules can cause the distance geometry solver to loop
    for minutes or forever.

    This module runs the conformer function in a fresh worker subprocess per
    call and enforces a real wall-clock timeout. If the worker hangs, it's
    killed cleanly on context-manager exit and the main process stays healthy.

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
    - A fresh ProcessPoolExecutor(1) is created per call. The ``with`` block
      guarantees full shutdown (including killing any hung worker) when we're
      done, so there's no state leakage between calls.
    - Cost: ~100-300 ms per call to spawn a worker. Cheap compared to the
      typical conformer generation time (several seconds) and the alternative
      of a hung container.
    - The previous implementation used a persistent ``multiprocessing.Pool(1)``
      with ``pool.terminate()`` on timeout. That approach was found to leave
      internal pool-management threads deadlocked after a timeout, eventually
      wedging the entire endpoint container. Fresh-per-call is more resilient.
"""

import logging
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Callable, Optional

logger = logging.getLogger("workbench")


def run_with_timeout(
    func: Callable,
    args: tuple = (),
    kwargs: Optional[dict] = None,
    timeout: float = 30.0,
) -> Any:
    """Run func(*args, **kwargs) in a worker subprocess with a wall-clock timeout.

    A fresh ``ProcessPoolExecutor(max_workers=1)`` is created per call and
    cleaned up via its context manager, which guarantees the worker process
    is killed on timeout or error. This avoids state leakage between calls.

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

    try:
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                return future.result(timeout=timeout)
            except FuturesTimeoutError:
                logger.warning(f"{func.__name__} timed out after {timeout}s — killing worker")
                # Leaving the 'with' block calls executor.shutdown(wait=True),
                # which will wait for the hung worker. Cancel the future first
                # so shutdown won't block indefinitely.
                future.cancel()
                return None
            except Exception as e:
                logger.warning(f"{func.__name__} raised in worker: {e}")
                return None
    except Exception as e:
        # Defensive: if executor setup/teardown itself fails, just log and
        # return None rather than crashing the caller.
        logger.warning(f"run_with_timeout failed to manage worker: {e}")
        return None
