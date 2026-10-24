"""Install smoke test: the import + Bosco paths a fresh `pip install workbench` user hits first.

Needs no AWS account or API key. Building Bosco's system prompt reads every bundled
guide, so a guide read that depends on the platform encoding fails here.

Invoked by the Windows-smoke workflow; runnable on any OS.
"""

import sys

import workbench
import workbench.api
import workbench.repl.workbench_shell  # noqa: F401
from workbench.agent import bosco


def main() -> int:
    print(f"=== install smoke: workbench {workbench.__version__}, Python {sys.version.split()[0]} ===")
    api = [name for name in dir(workbench.api) if not name.startswith("_")]
    print(f"  OK   workbench.api ({len(api)} names)")
    print("  OK   workbench.repl.workbench_shell")
    prompt = bosco._system_prompt()
    print(f"  OK   Bosco system prompt ({len(prompt)} chars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
