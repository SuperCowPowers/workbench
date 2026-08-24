"""Outbound HTTP for the REPL, checked against the configured egress mode."""

import re
import logging
from typing import NamedTuple
from urllib.parse import unquote

import requests
from rdkit import Chem, RDLogger

from workbench.utils.config_manager import ConfigManager

log = logging.getLogger("workbench")

# How far the REPL may reach: "off" (AWS only), "guarded" (public web, payload
# checked), "full" (unchecked). Read once at import -- a setting an agent could
# flip mid-session would be the accident it exists to prevent.
EGRESS_MODES = ("off", "guarded", "full")
DEFAULT_EGRESS = "off"

# A structure needs this many heavy atoms to count, and at least one of these
# characters. Ordinary URL words are drawn from the SMILES alphabet often enough
# to parse ("cocoons" is a valid molecule); none of them carry ring closures,
# branches, or explicit bonds.
MIN_HEAVY_ATOMS = 8
_STRUCTURAL = re.compile(r"[()\[\]=#]|\d")

# Split on URL punctuation only -- `=` is a SMILES double bond, so splitting on it
# shreds a structure into unparseable fragments.
_TOKENS = re.compile(r"[^/?&,\s]+")
_EXTENSION = re.compile(r"\.(?:json|xml|csv|tsv|txt|html?|sdf|smi)$", re.I)

# An InChIKey is a local hash of a structure -- not reversible, but a unique
# handle on a compound the user holds.
_INCHIKEY = re.compile(r"\b[A-Z]{14}-[A-Z]{10}-[A-Z]\b")

_SECRETS = {
    "AWS key id": re.compile(r"\b(?:AKIA|ASIA|AROA|AIDA)[0-9A-Z]{16}\b"),
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    "Slack token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    "Google API key": re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    "API key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "JWT": re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "credential parameter": re.compile(r"(?i)\b(?:api[_-]?key|secret|password|passwd|token)=[^&\s]{8,}"),
}


class EgressBlocked(RuntimeError):
    """Raised when a request is refused by the egress policy."""


class Finding(NamedTuple):
    """Something in a URL that shouldn't leave the machine."""

    label: str
    value: str

    def __str__(self) -> str:
        return f"{self.label}: {self.value}"


def egress_mode() -> str:
    """The configured egress mode, falling back to the default when unrecognized."""
    mode = ConfigManager().get_config("BOSCO_EGRESS", DEFAULT_EGRESS)
    if mode not in EGRESS_MODES:
        log.warning(f"Unknown BOSCO_EGRESS {mode!r}; using {DEFAULT_EGRESS!r} ({', '.join(EGRESS_MODES)})")
        return DEFAULT_EGRESS
    return mode


EGRESS_MODE = egress_mode()


def _candidates(text: str):
    """Substrings of a URL that could be a structure: path segments and param values."""
    for token in _TOKENS.findall(text):
        value = token.split("=", 1)[1] if "=" in token else None
        for candidate in (token, value):
            if candidate:
                yield _EXTENSION.sub("", candidate)


def _is_structure(token: str) -> bool:
    """True if the token is a chemical structure rather than a URL word."""
    if len(token) < MIN_HEAVY_ATOMS or not _STRUCTURAL.search(token):
        return False
    RDLogger.DisableLog("rdApp.*")
    try:
        mol = Chem.MolFromSmiles(token)
    finally:
        RDLogger.EnableLog("rdApp.*")
    return mol is not None and mol.GetNumHeavyAtoms() >= MIN_HEAVY_ATOMS


def scan(url: str) -> list:
    """Findings for anything in the URL that shouldn't leave the machine."""
    text = unquote(url)
    findings = [Finding(label, m.group()) for label, pat in _SECRETS.items() if (m := pat.search(text))]
    findings += [Finding("InChIKey", key) for key in dict.fromkeys(_INCHIKEY.findall(text))]
    findings += [Finding("structure", tok) for tok in dict.fromkeys(_candidates(text)) if _is_structure(tok)]
    return findings


def _unconfirmed(findings: list, confirm) -> list:
    """Findings the user hasn't approved.

    Consent is bound to the value, not to the assent -- one "yes" covers that
    structure wherever it appears, and nothing else. A finding is covered when the
    approved text contains it, so the `key=value` split of an already-approved
    structure doesn't re-prompt.
    """
    if not confirm:
        return findings
    if isinstance(confirm, str):
        confirm = [confirm]
    if not isinstance(confirm, (list, tuple, set)) or not all(isinstance(value, str) for value in confirm):
        raise ValueError("confirm= takes the value(s) the user approved, e.g. confirm=smiles -- not True")
    approved = " ".join(unquote(value) for value in confirm)
    return [f for f in findings if f.value not in approved]


def web_get(url: str, params: dict = None, confirm=None, timeout: int = 60, **kwargs):
    """Fetch a public URL, checked against the session's egress mode.

    Returns the `requests.Response`. In guarded mode the fully-resolved URL is
    scanned for structures, InChIKeys, and secrets; anything found raises
    EgressBlocked. Pass `confirm=` the exact value the user approved sending
    (a string, or a list of them) -- it covers that value and nothing else.
    """
    if EGRESS_MODE == "off":
        raise EgressBlocked(
            "Egress is off for this session. Use `pub_data` for external datasets, "
            "chem_utils/RDKit for computing on structures, and the installed source for API behavior."
        )

    prepared = requests.Request("GET", url, params=params, headers=kwargs.pop("headers", None)).prepare()

    if EGRESS_MODE == "guarded":
        host = prepared.url.split("/")[2]
        if findings := _unconfirmed(scan(prepared.url), confirm):
            detail = "; ".join(str(f) for f in findings)
            log.warning(f"Egress blocked to {host}: {detail}")
            raise EgressBlocked(
                f"This request would send {detail} to {host}. If the user supplied this and wants "
                "it sent, ask them first, then re-run passing that exact value as confirm=."
            )

    with requests.Session() as session:
        return session.send(prepared, timeout=timeout, **kwargs)
