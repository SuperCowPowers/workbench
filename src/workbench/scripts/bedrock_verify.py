"""
Verify that Bedrock/Claude is reachable from this AWS account.

Usage:
    bedrock_verify [model_id]
    bedrock_verify --list

Example:
    bedrock_verify
    bedrock_verify --list
    bedrock_verify anthropic.claude-sonnet-5

Runs a tiny round-trip against Bedrock using the Workbench assumed role and
region. See docs/aws_setup/bedrock_setup.md for setup.
"""

import sys

# Workbench Imports
from workbench.utils.repl_utils import cprint
from workbench.utils.bedrock_utils import DEFAULT_MODEL, bedrock_client, ping_model
from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp

DOC_HINT = "See docs/aws_setup/bedrock_setup.md (Troubleshooting) for the usual causes."


def list_models():
    """Print the Claude models and inference profiles available in this region."""
    client = bedrock_client()

    summaries = client.list_foundation_models(byProvider="anthropic")["modelSummaries"]
    cprint("lightpurple", "Foundation models:")
    for model_id in dict.fromkeys(s["modelId"] for s in summaries):
        cprint("lightblue", f"  {model_id}")

    # Legacy (bedrock-runtime) models route through these; the Messages API
    # endpoint used by the agent takes bare anthropic.* ids instead
    profiles = client.list_inference_profiles()["inferenceProfileSummaries"]
    claude = [p["inferenceProfileId"] for p in profiles if "anthropic" in p["inferenceProfileId"]]
    if claude:
        cprint("lightpurple", "Inference profiles (legacy path):")
        for profile_id in claude:
            cprint("lightblue", f"  {profile_id}")


def main():
    try:
        import anthropic  # noqa: F401
    except ImportError:
        cprint("red", "The 'anthropic' package is not installed.")
        cprint("lightblue", "Install it with: pip install -U 'anthropic[bedrock]'")
        sys.exit(1)

    if "--list" in sys.argv:
        list_models()
        return

    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    model_id = args[0] if args else DEFAULT_MODEL

    clamp = AWSAccountClamp()
    cprint("lightpurple", f"Region: {clamp.region}")
    cprint("lightpurple", f"Model:  {model_id}")

    ok, detail = ping_model(model_id)
    if not ok:
        cprint("red", f"Bedrock call failed: {detail}")
        if "marketplace" in detail.lower():
            cprint("lightblue", "Run 'bedrock_subscribe' with admin credentials (one time per account).")
        else:
            cprint("lightblue", "Run 'bedrock_verify --list' to see available models.")
        cprint("lightblue", DOC_HINT)
        sys.exit(1)

    cprint("lightgreen", f"Success: {detail}")


if __name__ == "__main__":
    main()
