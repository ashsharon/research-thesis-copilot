"""Single place that talks to the Claude API, so every node shares one client
and one retry policy."""
from __future__ import annotations

from anthropic import Anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings

_client: Anthropic | None = None


def get_client() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic(api_key=settings.anthropic_api_key)
    return _client


@retry(wait=wait_exponential(multiplier=1, min=2, max=20), stop=stop_after_attempt(4))
def complete(system: str, user: str, max_tokens: int = 1500, temperature: float = 0.2) -> str:
    client = get_client()
    response = client.messages.create(
        model=settings.claude_model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(block.text for block in response.content if block.type == "text")
