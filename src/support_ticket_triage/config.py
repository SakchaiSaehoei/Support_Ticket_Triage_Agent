"""Environment settings and repository-relative paths."""

import os
from pathlib import Path

from dotenv import load_dotenv

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def repository_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


class Setting:
    """Load configuration once, before clients and the graph are constructed."""

    def __init__(self) -> None:
        load_dotenv(REPOSITORY_ROOT / ".env")
        self.apikey = os.getenv("OPENAI_API_KEY")
        self.response_model = os.getenv("OPENAI_RESPONSE_MODEL")
        self.decision_model = os.getenv("OPENAI_DECISION_MODEL")
        self.embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.database_path = repository_path(
            os.getenv("SUPPORT_DATABASE_PATH", ".state/support.sqlite3")
        )
        checkpoint_path = os.getenv("SUPPORT_CHECKPOINT_PATH", "").strip()
        self.checkpoint_path = repository_path(checkpoint_path) if checkpoint_path else None
        self.langsmith_tracing_status = os.getenv("LANGSMITH_TRACING", "false").strip().lower() in {
            "true", "1", "yes", "on",
        }
        self._validate_required_setting()

    def _validate_required_setting(self) -> None:
        values = {
            "OPENAI_API_KEY": self.apikey,
            "OPENAI_RESPONSE_MODEL": self.response_model,
            "OPENAI_DECISION_MODEL": self.decision_model,
        }
        missing = [name for name, value in values.items() if not value or not value.strip()]
        if missing:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing)}. "
                "Set them in the repository's .env file."
            )


if __name__ == "__main__":
    setting = Setting()
    print(setting.response_model)
    print(setting.decision_model)
