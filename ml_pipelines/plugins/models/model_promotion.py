"""Promotion arbiter as a client plugin (``plugin:models/model_promotion.py``).

Exercises the ``plugin:`` resolution path, which resolves against
``<discovery root>/plugins/`` rather than the packaged ``workbench:`` namespace. A real
client plugin replaces this body with its own promotion policy — Parameter Store gates,
notifications, a different ranking rule — rather than re-stating the packaged one. This
delegates instead, so the demo cannot drift from the behavior it is demonstrating.
"""

from workbench.batch.models.model_promotion import main

if __name__ == "__main__":
    main()
