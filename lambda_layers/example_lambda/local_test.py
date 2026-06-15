"""Offline smoke test for the example lambda + the built layer.

Loads workbench/networkx *from the built layer* (lambda_layers/build/python),
isolated from the editable dev install, so this exercises exactly the code the
layer ships -- then invokes the handler against a local sample with simulated
mtimes (no AWS, no creds). Run after `build_deploy.sh`:

    python lambda_layers/example_lambda/local_test.py
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LAYER = os.path.abspath(os.path.join(HERE, "..", "build", "python"))

if not os.path.isdir(LAYER):
    sys.exit(f"Layer not built: {LAYER} missing. Run lambda_layers/build_deploy.sh first.")

# Load the layer's copy first; drop the editable dev workbench so we test what the
# layer actually ships (this is how Lambda loads it -- layer on the path, nothing else).
sys.path = [HERE, LAYER] + [p for p in sys.path if "site-packages" not in p and "workbench/src" not in p]

import handler  # noqa: E402  (after sys.path surgery)

assert handler.workbench.__file__.startswith(LAYER), f"handler loaded dev workbench, not the layer: {handler.workbench.__file__}"

# ds:demo_raw "modified" -> demo_fs stale -> both demo_reg models flood downstream.
event = {"pipelines_path": os.path.join(HERE, "sample_pipelines"), "simulate_modified": ["ds:demo_raw"]}
result = handler.lambda_handler(event)
body = json.loads(result["body"])
print(json.dumps(body, indent=2))

assert result["statusCode"] == 200
assert body["num_pipelines"] == 1
assert set(body["will_run"]) == {"demo_fs", "demo_reg [dt]", "demo_reg [ts]"}, body["will_run"]
print("\nLOCAL LAYER TEST PASSED -- handler ran against the built layer with no AWS.")
