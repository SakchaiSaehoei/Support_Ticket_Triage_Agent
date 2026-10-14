"""Customer, knowledge, and cached status access, independent of triage policy."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from support_ticket_triage.config import REPOSITORY_ROOT, repository_path
from support_ticket_triage.database import SQLiteStore
from support_ticket_triage.retrieval import KNOWLEDGE_PATH, KnowledgeBaseRetriever
from support_ticket_triage.schemas import CustomerProfile, KnowledgeBaseArticle


class SupportTools:
    """Expose data access; model services decide urgency and ownership."""

    def __init__(
        self,
        store: SQLiteStore | None = None,
        retriever: KnowledgeBaseRetriever | None = None,
        knowledge_base_path: Path | None = None,
        status_path: Path | None = None,
    ) -> None:
        self._store = store
        self.retriever = retriever
        self.knowledge_base_path = knowledge_base_path or KNOWLEDGE_PATH
        self.status_path = status_path or REPOSITORY_ROOT / "data/fixtures/system_status.json"

    @property
    def store(self) -> SQLiteStore:
        if self._store is None:
            load_dotenv(REPOSITORY_ROOT / ".env")
            self._store = SQLiteStore(repository_path(
                os.getenv("SUPPORT_DATABASE_PATH", ".state/support.sqlite3")
            ))
        return self._store

    def check_customer_history(self, customer_id: str) -> CustomerProfile | None:
        """Read the stored profile; an unknown customer is distinct from a failed lookup."""
        return self.store.get_customer(customer_id)

    async def lookup_knowledge_base(self, query: str) -> list[KnowledgeBaseArticle]:
        """Return relevant internal diagnostic procedures, not verified product facts."""
        if self.retriever is None:
            raise ValueError("Configure the knowledge retriever before searching.")
        return await self.retriever.search(query)

    def check_system_status(self, region: str | None = None) -> dict:
        """Read a mock status source with a five-minute cache; unknown remains unknown."""
        key = (region or "default").strip().casefold()
        cached = self.store.get_cached_status(key)
        if cached is not None:
            return {**cached, "cached": True}
        statuses = json.loads(self.status_path.read_text(encoding="utf-8"))
        if not isinstance(statuses, dict):
            raise ValueError("The mock status fixture must be a JSON object.")
        entry = statuses.get(key, statuses.get("default"))
        if not isinstance(entry, dict):
            raise ValueError("The mock status fixture has no usable status record.")
        payload = {
            **entry,
            "region": region,
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "source": "data/fixtures/system_status.json",
            "is_mock": True,
        }
        self.store.cache_status(key, payload, ttl_seconds=300)
        return {**payload, "cached": False}
