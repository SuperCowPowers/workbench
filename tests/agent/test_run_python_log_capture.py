"""run_python surfaces WARNING+ log records emitted while Bosco's code runs.

Workbench code often logs an error and returns an empty/None result instead of
raising. Those records go to their handlers' original stdout, which
`redirect_stdout` never swaps, so without the capture the tool output is clean
and Bosco can't see why a result came back empty.
"""

import logging

from workbench.agent.tools import run_python

MARKER = "--- logged during execution (not stdout) ---"


def test_clean_run_has_no_log_section():
    out = run_python("print('hello')", {})
    assert out == "hello"
    assert MARKER not in out


def test_workbench_warning_and_error_are_captured():
    code = (
        "import logging\n"
        "log = logging.getLogger('workbench')\n"
        "log.info('routine chatter')\n"
        "log.warning('metrics came back empty')\n"
        "log.error('no inference results')\n"
        "print('0 rows')"
    )
    out = run_python(code, {})
    assert "0 rows" in out
    assert MARKER in out
    assert "WARNING workbench: metrics came back empty" in out
    assert "ERROR workbench: no inference results" in out
    assert "routine chatter" not in out  # INFO is below the WARNING threshold


def test_third_party_root_logger_is_captured():
    # botocore propagates to root; the workbench logger has propagate=False, so
    # capture must watch both — this guards the root half.
    code = "import logging; logging.getLogger('botocore.creds').warning('token expiring')"
    out = run_python(code, {})
    assert "WARNING botocore.creds: token expiring" in out


def test_repeated_records_collapse_with_count():
    code = (
        "import logging\n"
        "log = logging.getLogger('workbench')\n"
        "for _ in range(500):\n"
        "    log.warning('same line')"
    )
    out = run_python(code, {})
    assert "WARNING workbench: same line (x500)" in out
    assert out.count("same line") == 1


def test_exception_and_warning_both_surface():
    code = "import logging\n" "logging.getLogger('workbench').warning('about to fail')\n" "raise ValueError('boom')"
    out = run_python(code, {})
    assert "ValueError: boom" in out  # traceback in the stdout buffer
    assert "WARNING workbench: about to fail" in out  # log section


def test_handlers_are_removed_after_run():
    watched = [logging.getLogger(""), logging.getLogger("workbench")]
    before = [len(lg.handlers) for lg in watched]
    run_python("logging.getLogger('workbench').warning('x') if False else None", {})
    assert [len(lg.handlers) for lg in watched] == before
