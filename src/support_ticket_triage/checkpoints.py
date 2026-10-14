"""Manage asynchronous graph checkpoints separately from conversation records."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from support_ticket_triage import schemas


CHECKPOINT_TYPES = (
    schemas.TicketInput, schemas.TicketUpdate, schemas.TicketMessage,
    schemas.CustomerProfile, schemas.KnowledgeBaseArticle, schemas.ToolContext,
    schemas.ExtractedIssue, schemas.TicketAnalysis, schemas.IssueDecision,
    schemas.TriageDecision, schemas.HandoffReceipt, schemas.TicketResult,
    schemas.Urgency, schemas.NextAction, schemas.SpecialistTeam,
)


def checkpoint_path_for(database_path: Path, checkpoint_path: Path | None = None) -> Path:
    """Keep progress beside the selected support database unless explicitly overridden."""
    database_path = database_path.expanduser().resolve()
    return (checkpoint_path or database_path.with_name(
        f"{database_path.stem}.checkpoints.sqlite3"
    )).expanduser().resolve()


@asynccontextmanager
async def open_checkpointer(
    database_path: Path, checkpoint_path: Path | None = None,
) -> AsyncIterator[AsyncSqliteSaver]:
    """Keep the saver connection open for a run/batch and close it on every exit."""
    database_path = database_path.expanduser().resolve()
    path = checkpoint_path_for(database_path, checkpoint_path)
    if path == database_path:
        raise ValueError("Use separate files for support records and graph checkpoints.")
    path.parent.mkdir(parents=True, exist_ok=True)
    serializer = JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES)
    async with aiosqlite.connect(str(path)) as connection:
        yield AsyncSqliteSaver(connection, serde=serializer)
