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
    "us.anthropic.claude-opus-5",
    "us.anthropic.claude-opus-4-8",
]

DEFAULT_MODEL = CLAUDE_MODELS[0]

# Shared across calls so the credential-freezing client is built once per session.
_client = None


def bedrock_client():
    """Bedrock control-plane client on the Workbench assumed role."""
    clamp = AWSAccountClamp()
    return clamp.boto3_session.client("bedrock", region_name=clamp.region)


def bedrock_available(model_id: str = DEFAULT_MODEL) -> bool:
    """Fast check that this session can reach Bedrock and see the model.

    Control-plane only (ListFoundationModels): verifies region + IAM + that the
    model exists in the account. Does not verify the marketplace subscription --
    that surfaces on the first invoke. Swallows all errors and returns False, so
    it is safe to call at REPL startup.
    """
    try:
        base = model_id.split(".", 1)[1] if model_id.startswith(("us.", "global.")) else model_id
        resp = bedrock_client().list_foundation_models(byProvider="anthropic")
        return any(m["modelId"].startswith(base) for m in resp.get("modelSummaries", []))
    except Exception:
        return False


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


def _call_with_retry(fn):
    """Run fn(client) against the shared client, rebuilding it once on an expired token.

    claude_client() freezes credentials into the Anthropic client, so an SSO token
    renewed elsewhere is only picked up when the client is rebuilt (Workbench's own
    boto3 session refreshes on its own). Retrying once on a 403 means a renewal takes
    effect without restarting the REPL.
    """
    global _client
    import anthropic

    if _client is None:
        _client = claude_client()
    try:
        return fn(_client)
    except anthropic.PermissionDeniedError:
        _client = claude_client()
        return fn(_client)


def message_create(**kwargs):
    """Send a message and return the complete reply.

    Args:
        **kwargs: Passed straight to the Messages API (model, messages, tools, ...).

    Returns:
        anthropic.types.Message: The completed message.
    """
    return _call_with_retry(lambda client: client.messages.create(**kwargs))


def message_stream(on_phase=None, **kwargs):
    """Send a message over a stream, reporting each content block as it starts.

    Streaming is what makes a long turn observable: the caller learns when the model
    moves from thinking to writing to requesting a tool, rather than waiting on one
    opaque call. Nothing is printed here -- `on_phase` decides what the user sees.

    Args:
        on_phase (callable, optional): Called `on_phase(kind, tool_name)` as each
            content block starts, where kind is "thinking", "text", or "tool_use",
            and tool_name is the tool for a "tool_use" block, otherwise None.
        **kwargs: Passed straight to the Messages API (model, messages, tools, ...).

    Returns:
        anthropic.types.Message: The assembled message, same shape as message_create().
    """

    def run(client):
        with client.messages.stream(**kwargs) as stream:
            for event in stream:
                if on_phase is not None and event.type == "content_block_start":
                    block = event.content_block
                    on_phase(block.type, getattr(block, "name", None))
            return stream.get_final_message()

    return _call_with_retry(run)


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
