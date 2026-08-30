"""Claude client for Workbench, provider-aware.

Bosco reaches Claude one of three ways, decided by `llm_provider()`:

    bedrock     the Workbench assumed role, inside the user's own AWS account
    anthropic   the user's own key in ANTHROPIC_API_KEY (local mode only)
    trial       a SuperCowPowers key, forwarded to Bedrock by the SCP proxy

Mode decides this, not precedence. A connected AWS account always goes to Bedrock,
so Bosco's traffic never leaves it and there is no setting that routes an AWS
session elsewhere. The key lanes serve local mode, where there is no account for
the traffic to stay inside.

Bedrock takes an inference profile id rather than a bare model name. The `us.`
prefix keeps inference inside US regions, which is what customer data agreements
generally require; `global.` routes anywhere for a 10% lower price.

AWS also offers a newer Messages API endpoint ("Mantle") taking bare `anthropic.`
ids, which returns 404 for every model as of 2026-07. Moving to it is a one-line
change: swap AnthropicBedrock for AnthropicBedrockMantle and drop the prefix.
"""

import os
import logging
from functools import lru_cache
from typing import List, Optional

# Workbench Imports
from workbench.utils.config_manager import ConfigManager

log = logging.getLogger("workbench")

# Preference order: the agent uses the first one available.
# Current-generation models only -- subscribe these in the Bedrock playground.
CLAUDE_MODELS: List[str] = [
    "claude-opus-5",
    "claude-opus-4-8",
]

# Bedrock's inference profile prefix; the direct API takes the bare name above.
BEDROCK_PREFIX = "us.anthropic."

# Config keys for the trial lane: the SuperCowPowers key and the proxy it talks to.
LLM_KEY = "WORKBENCH_LLM_KEY"
LLM_URL = "WORKBENCH_LLM_URL"

# The SCP proxy is not stood up yet, so there is no default endpoint to fall back
# to -- the trial lane stays dark until WORKBENCH_LLM_URL names one.
SCP_PROXY_URL = None

# Shared across calls so the credential-freezing client is built once per session.
_client = None


@lru_cache(maxsize=1)
def llm_provider() -> str:
    """Which path this session reaches Claude on.

    An AWS account takes precedence over any key, so a connected session cannot be
    diverted off Bedrock by an ANTHROPIC_API_KEY that happens to be in the user's
    environment for some other tool.

    Resolved once: the answer comes from the environment and the config file,
    neither of which changes mid-session, and `config_okay()` logs when a config
    is incomplete -- something a per-turn call would repeat endlessly.

    Returns:
        str: "bedrock", "anthropic", "trial", or "none" when no path is available.
    """
    if ConfigManager().config_okay():
        return "bedrock"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if ConfigManager().get_config(LLM_KEY) and proxy_url():
        return "trial"
    return "none"


def proxy_url() -> Optional[str]:
    """The SCP proxy endpoint for the trial lane, or None when none is configured."""
    return ConfigManager().get_config(LLM_URL) or SCP_PROXY_URL


def default_model(provider: str = None) -> str:
    """The preferred model id, in the form the given provider expects.

    Args:
        provider (str, optional): Provider to format for. Defaults to the active one.

    Returns:
        str: e.g. "us.anthropic.claude-opus-5" on Bedrock, "claude-opus-5" otherwise.
    """
    provider = provider or llm_provider()
    name = CLAUDE_MODELS[0]
    return f"{BEDROCK_PREFIX}{name}" if provider == "bedrock" else name


def bedrock_client():
    """Bedrock control-plane client on the Workbench assumed role."""
    from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp

    clamp = AWSAccountClamp()
    return clamp.boto3_session.client("bedrock", region_name=clamp.region)


def llm_available() -> bool:
    """Fast check that this session can actually reach a model.

    Safe to call at REPL startup: Bedrock gets a control-plane check, and the
    key-based providers are taken at their word rather than paying a network
    round trip on every launch -- a bad key surfaces on the first real call.

    Returns:
        bool: True if Bosco has a usable path to Claude.
    """
    provider = llm_provider()
    if provider == "none":
        return False
    if provider == "bedrock":
        return _bedrock_has_model(default_model("bedrock"))
    return True


def _bedrock_has_model(model_id: str) -> bool:
    """Verify region + IAM + that the model exists in the account.

    Control-plane only (ListFoundationModels). Does not verify the marketplace
    subscription -- that surfaces on the first invoke. Swallows all errors and
    returns False, so it is safe to call at REPL startup.
    """
    try:
        base = model_id.split(".", 1)[1] if model_id.startswith(("us.", "global.")) else model_id
        resp = bedrock_client().list_foundation_models(byProvider="anthropic")
        return any(m["modelId"].startswith(base) for m in resp.get("modelSummaries", []))
    except Exception:
        return False


def claude_client(provider: str = None):
    """Anthropic client for the active provider.

    On Bedrock the credentials are passed explicitly: the Anthropic client
    otherwise resolves its own from the default chain, which would use the base
    identity rather than the Workbench role (where the Bedrock policies live).

    Args:
        provider (str, optional): Pin a provider. Defaults to the active one.
    """
    from anthropic import Anthropic, AnthropicBedrock

    provider = provider or llm_provider()
    if provider == "bedrock":
        from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp

        clamp = AWSAccountClamp()
        creds = clamp.boto3_session.get_credentials().get_frozen_credentials()
        return AnthropicBedrock(
            aws_access_key=creds.access_key,
            aws_secret_key=creds.secret_key,
            aws_session_token=creds.token,
            aws_region=clamp.region,
        )
    if provider == "trial":
        return Anthropic(api_key=ConfigManager().get_config(LLM_KEY), base_url=proxy_url())
    if provider == "anthropic":
        return Anthropic()  # the SDK reads ANTHROPIC_API_KEY itself
    raise RuntimeError(f"No path to Claude: connect an AWS account, or set ANTHROPIC_API_KEY / {LLM_KEY}")


def _call_with_retry(fn):
    """Run fn(client) against the shared client, rebuilding it once on an expired token.

    Only the Bedrock path can go stale this way: claude_client() freezes credentials
    into the Anthropic client, so an SSO token renewed elsewhere is picked up only
    when the client is rebuilt (Workbench's own boto3 session refreshes on its own).
    Retrying once on a 403 means a renewal takes effect without restarting the REPL.
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
        anthropic.types.Message: The assembled message.
    """

    def run(client):
        with client.messages.stream(**kwargs) as stream:
            for event in stream:
                if on_phase is not None and event.type == "content_block_start":
                    block = event.content_block
                    on_phase(block.type, getattr(block, "name", None))
            return stream.get_final_message()

    return _call_with_retry(run)


def ping_model(model_id: str, provider: str = None) -> tuple:
    """Send a minimal message to a model.

    Args:
        model_id (str): The model to ping, in the provider's own id form.
        provider (str, optional): Pin a provider. Defaults to the active one.

    Returns:
        tuple: (ok: bool, detail: str) - the reply text, or the error.
    """
    try:
        msg = claude_client(provider).messages.create(
            model=model_id,
            max_tokens=16,
            messages=[{"role": "user", "content": "Reply with the word: ready"}],
        )
        return True, next((b.text for b in msg.content if b.type == "text"), "").strip()
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
