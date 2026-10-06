"""Central configuration, loaded from environment variables (.env)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    claude_model: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5")

    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", str(ROOT_DIR / "data" / "chroma"))
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    arxiv_category: str = os.getenv("ARXIV_CATEGORY", "cs.CL")
    arxiv_max_results: int = int(os.getenv("ARXIV_MAX_RESULTS", "15"))

    digest_recipient: str = os.getenv("DIGEST_RECIPIENT", "you@example.com")

    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8080"))

    papers_dir: Path = ROOT_DIR / "data" / "papers"


settings = Settings()
settings.papers_dir.mkdir(parents=True, exist_ok=True)
