"""Interactive configuration for ax-prover API keys."""

from __future__ import annotations

import importlib.resources as pkg_resources
import os
import re

from ..utils.config import USER_SECRETS_PATH

# Keys to prompt for, in order. Each is (env_var, label, required)
_API_KEYS = [
    ("ANTHROPIC_API_KEY", "Anthropic (Claude) — recommended", True),
    ("OPENAI_API_KEY", "OpenAI", False),
    ("GOOGLE_API_KEY", "Google (Gemini)", False),
    ("TAVILY_API_KEY", "Tavily (web search)", False),
    ("LANGSMITH_API_KEY", "LangSmith (tracing, optional)", False),
]


def _mask(value: str) -> str:
    """Mask a secret value, showing only the last 4 characters."""
    if len(value) <= 8:
        return "****"
    return "****" + value[-4:]


def _load_existing_secrets() -> dict[str, str]:
    """Load existing secrets from the user secrets file."""
    secrets: dict[str, str] = {}
    if not USER_SECRETS_PATH.exists():
        return secrets
    for line in USER_SECRETS_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([A-Z_]+)=(.+)$", line)
        if match:
            key, value = match.group(1), match.group(2)
            # Skip placeholder values
            if not value.startswith("your-"):
                secrets[key] = value
    return secrets


def _get_template() -> str:
    """Load the .env.secrets.example template from bundled configs."""
    template_path = pkg_resources.files("ax_prover.configs") / ".env.secrets.example"
    return template_path.read_text()


def configure() -> None:
    """Interactive setup for API keys."""
    print("ax-prover configure")
    print("=" * 40)
    print()
    print(f"Secrets will be saved to: {USER_SECRETS_PATH}")
    print()

    existing = _load_existing_secrets()
    new_secrets: dict[str, str] = {}

    for env_var, label, _required in _API_KEYS:
        # Check environment first, then existing file
        current = os.environ.get(env_var) or existing.get(env_var)

        if current:
            prompt = f"  {label} [{_mask(current)}]: "
        else:
            prompt = f"  {label}: "

        value = input(prompt).strip()

        if value:
            new_secrets[env_var] = value
        elif current:
            new_secrets[env_var] = current

    # Write to secrets file
    USER_SECRETS_PATH.parent.mkdir(parents=True, exist_ok=True)

    template = _get_template()

    # Replace placeholder values in the template with actual values
    lines = []
    for line in template.splitlines():
        match = re.match(r"^([A-Z_]+)=(.*)$", line)
        if match:
            key = match.group(1)
            if key in new_secrets:
                lines.append(f"{key}={new_secrets[key]}")
            else:
                lines.append(line)
        else:
            lines.append(line)

    USER_SECRETS_PATH.write_text("\n".join(lines) + "\n")

    print()
    print(f"Saved to {USER_SECRETS_PATH}")

    configured = [k for k in new_secrets if not new_secrets[k].startswith("your-")]
    if configured:
        print(f"Configured: {', '.join(configured)}")
    else:
        print("No keys configured. You can run 'ax-prover configure' again anytime.")
