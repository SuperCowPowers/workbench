"""Bedrock/Claude helpers for Workbench.

Current-generation Claude models are reached through bedrock-runtime with an
inference profile id. The `us.` prefix keeps inference inside US regions,
which is what customer data agreements generally require; `global.` routes
anywhere for a 10% lower price.

AWS also offers a newer Messages API endpoint ("Mantle") taking bare
`anthropic.` ids, which returns 404 for every model as of 2026-07. Moving to
it is a one-line change: swap AnthropicBedrock for AnthropicBedrockMantle
below and drop the `us.` prefix from the ids above.
"""

from typing import List

# Workbench Imports
from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp

# Preference order: the agent uses the first one available in the account.
# Current-generation models only -- subscribe these in the Bedrock playground.
CLAUDE_MODELS: List[str] = [
    "us.anthropic.claude-opus-4-8",
    "us.anthropic.claude-sonnet-5",
]

DEFAULT_MODEL = CLAUDE_MODELS[0]


def bedrock_client():
    """Bedrock control-plane client on the Workbench assumed role."""
    clamp = AWSAccountClamp()
    return clamp.boto3_session.client("bedrock", region_name=clamp.region)


def claude_client():
    """Anthropic client bound to the Workbench assumed role.

    Credentials are passed explicitly: the Anthropic client otherwise resolves
    its own from the default chain, which would use the base identity rather
    than the Workbench role (where the Bedrock policies live).
    """
    from anthropic import AnthropicBedrock

    clamp = AWSAccountClamp()
    creds = clamp.boto3_session.get_credentials().get_frozen_credentials()
    return AnthropicBedrock(
        aws_access_key=creds.access_key,
        aws_secret_key=creds.secret_key,
        aws_session_token=creds.token,
        aws_region=clamp.region,
    )


def ping_model(model_id: str) -> tuple:
    """Send a minimal message to a model.

    Returns:
        tuple: (ok: bool, detail: str) - the reply text, or the error.
    """
    try:
        msg = claude_client().messages.create(
            model=model_id,
            max_tokens=16,
            messages=[{"role": "user", "content": "Reply with the word: ready"}],
        )
        return True, next((b.text for b in msg.content if b.type == "text"), "").strip()
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
