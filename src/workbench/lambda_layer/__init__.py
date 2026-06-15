"""workbench.lambda_layer: the curated, dependency-minimal subset of workbench.

Everything under this package is what ships in the published workbench Lambda
layer. Membership is the contract: a module belongs here only if it imports
cleanly with the layer's allowlisted dependencies (currently stdlib + networkx;
boto3/botocore come from the Lambda runtime). Keep it that way -- adding a module
that drags a new third-party dependency either grows the layer's allowlist on
purpose or doesn't belong here.

Importing this package is intentionally side-effect free; import the specific
module you need (e.g. ``workbench.lambda_layer.pipelines_manager``).
"""
