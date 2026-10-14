"""Local persistence for customer records, ticket history, and triage results.

Storage records facts supplied by callers. It does not classify tickets or send replies.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import time
from typing import Any
from uuid import uuid4

from pydantic import TypeAdapter

from .schemas import (
    CustomerProfile,
    HandoffReceipt,
    TicketAnalysis,
    TicketInput,
    TicketMessage,
    TicketResult,
    TicketSummary,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE_PATH = REPOSITORY_ROOT / ".state" / "support.sqlite3"
DEFAULT_FIXTURE_DIRECTORY = REPOSITORY_ROOT / "data" / "fixtures"


def stable_message_id(ticket_id: str, sequence: int, message: TicketMessage) -> str:
    """Identify a full-transcript entry using its ticket, one-based order, and content."""
    content = json.dumps(
        {
            "ticket_id": ticket_id,
            "sequence": sequence,
            "text": message.text,
            "time_label": message.time_label,
            "role": message.role,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return "message-" + hashlib.sha256(content.encode("utf-8")).hexdigest()


class SQLiteStore:
    """Use one short-lived connection per operation, with atomic writes."""

    def __init__(
        self,
        path: str | Path = DEFAULT_DATABASE_PATH,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if str(path) == ":memory:":
            raise ValueError("SQLiteStore requires a file-backed database path.")
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        with self._connection(write=True) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS customers (
                    customer_id TEXT PRIMARY KEY,
                    profile_json TEXT NOT NULL,
                    sort_order INTEGER
                );
                CREATE TABLE IF NOT EXISTS tickets (
                    ticket_id TEXT PRIMARY KEY,
                    customer_id TEXT NOT NULL,
                    title TEXT,
                    status TEXT NOT NULL DEFAULT 'open'
                        CHECK(status IN ('open', 'pending', 'resolved')),
                    created_at TEXT,
                    updated_at TEXT
                );
                CREATE TABLE IF NOT EXISTS ticket_messages (
                    message_id TEXT PRIMARY KEY,
                    ticket_id TEXT NOT NULL REFERENCES tickets(ticket_id),
                    sequence INTEGER NOT NULL CHECK(sequence > 0),
                    text TEXT NOT NULL,
                    time_label TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('customer', 'support')),
                    observed_at TEXT,
                    UNIQUE(ticket_id, sequence)
                );
                CREATE TABLE IF NOT EXISTS triage_runs (
                    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket_id TEXT NOT NULL REFERENCES tickets(ticket_id),
                    result_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS ticket_memory (
                    ticket_id TEXT PRIMARY KEY REFERENCES tickets(ticket_id),
                    analysis_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS system_status_cache (
                    cache_key TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    expires_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS mock_handoffs (
                    dispatch_id TEXT PRIMARY KEY,
                    ticket_id TEXT NOT NULL REFERENCES tickets(ticket_id),
                    customer_id TEXT NOT NULL,
                    receipt_json TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
        # A separate write transaction makes the column check and migration atomic.
        # Existing message rows retain NULL because their observation time is unknown.
        with self._connection(write=True) as connection:
            columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(ticket_messages)")
            }
            if "observed_at" not in columns:
                connection.execute("ALTER TABLE ticket_messages ADD COLUMN observed_at TEXT")
            self._migrate_metadata(connection)

    @staticmethod
    def _ticket_title(text: str) -> str:
        title = " ".join(text.split())
        return title if len(title) <= 80 else title[:77].rstrip() + "..."

    def _migrate_metadata(self, connection: sqlite3.Connection) -> None:
        """Add selection metadata while retaining every legacy message and profile."""
        customer_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(customers)")
        }
        if "sort_order" not in customer_columns:
            connection.execute("ALTER TABLE customers ADD COLUMN sort_order INTEGER")
        next_order = connection.execute(
            "SELECT COALESCE(MAX(sort_order), 0) FROM customers"
        ).fetchone()[0]
        for row in connection.execute(
            "SELECT customer_id FROM customers WHERE sort_order IS NULL ORDER BY rowid"
        ).fetchall():
            next_order += 1
            connection.execute(
                "UPDATE customers SET sort_order = ? WHERE customer_id = ?",
                (next_order, row["customer_id"]),
            )

        ticket_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(tickets)")
        }
        additions = {
            "title": "TEXT",
            "status": "TEXT NOT NULL DEFAULT 'open' CHECK(status IN ('open', 'pending', 'resolved'))",
            "created_at": "TEXT",
            "updated_at": "TEXT",
        }
        for name, definition in additions.items():
            if name not in ticket_columns:
                connection.execute(f"ALTER TABLE tickets ADD COLUMN {name} {definition}")

        for ticket in connection.execute(
            "SELECT * FROM tickets WHERE title IS NULL OR created_at IS NULL OR updated_at IS NULL"
        ).fetchall():
            messages = connection.execute(
                "SELECT text, observed_at FROM ticket_messages WHERE ticket_id = ? ORDER BY sequence",
                (ticket["ticket_id"],),
            ).fetchall()
            observations = [
                datetime.fromisoformat(row["observed_at"]).astimezone(timezone.utc)
                for row in messages if row["observed_at"] is not None
            ]
            run_times = [
                datetime.fromtimestamp(row[0], tz=timezone.utc)
                for row in connection.execute(
                    "SELECT created_at FROM triage_runs WHERE ticket_id = ?",
                    (ticket["ticket_id"],),
                )
            ]
            recorded_times = observations + run_times
            created_at = ticket["created_at"] or (
                min(recorded_times) if recorded_times else self._now()
            ).isoformat()
            updated_at = ticket["updated_at"] or max(
                recorded_times + [datetime.fromisoformat(created_at)]
            ).isoformat()
            title = ticket["title"] or (
                self._ticket_title(messages[0]["text"]) if messages else "New support request"
            )
            connection.execute(
                "UPDATE tickets SET title = ?, created_at = ?, updated_at = ? WHERE ticket_id = ?",
                (title, created_at, updated_at, ticket["ticket_id"]),
            )

    def _now(self) -> datetime:
        return datetime.fromtimestamp(self._clock(), tz=timezone.utc)

    @contextmanager
    def _connection(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _ensure_owner(
        self, connection: sqlite3.Connection, ticket_id: str, customer_id: str
    ) -> bool:
        row = connection.execute(
            "SELECT customer_id FROM tickets WHERE ticket_id = ?", (ticket_id,)
        ).fetchone()
        if row is not None:
            if row["customer_id"] != customer_id:
                raise ValueError("Ticket belongs to a different customer.")
            return False
        now = self._now().isoformat()
        connection.execute(
            """INSERT INTO tickets(ticket_id, customer_id, title, status, created_at, updated_at)
               VALUES (?, ?, ?, 'open', ?, ?)""",
            (ticket_id, customer_id, "New support request", now, now),
        )
        return True

    @staticmethod
    def _same_content(row: sqlite3.Row, message: TicketMessage) -> bool:
        observation_matches = message.observed_at is None or (
            row["observed_at"] is not None
            and datetime.fromisoformat(row["observed_at"]) == message.observed_at
        )
        return (
            row["text"] == message.text
            and row["time_label"] == message.time_label
            and row["role"] == message.role
            and observation_matches
        )

    @staticmethod
    def _message_from_row(row: sqlite3.Row) -> TicketMessage:
        return TicketMessage(
            message_id=row["message_id"],
            text=row["text"],
            time_label=row["time_label"],
            role=row["role"],
            observed_at=row["observed_at"],
        )

    def get_customer(self, customer_id: str) -> CustomerProfile | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT profile_json FROM customers WHERE customer_id = ?",
                (customer_id,),
            ).fetchone()
        return None if row is None else CustomerProfile.model_validate_json(row[0])

    @staticmethod
    def _require_customer(connection: sqlite3.Connection, customer_id: str) -> None:
        if connection.execute(
            "SELECT 1 FROM customers WHERE customer_id = ?", (customer_id,)
        ).fetchone() is None:
            raise KeyError(f"Customer not found: {customer_id}")

    def list_customers(self) -> list[CustomerProfile]:
        """Retain a customer's menu position across restarts and repeated seeding."""
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT profile_json FROM customers ORDER BY sort_order, customer_id"
            ).fetchall()
        return [CustomerProfile.model_validate_json(row[0]) for row in rows]

    def create_customer(
        self,
        *,
        plan: str,
        display_name: str | None = None,
        tenure_months: int | None = None,
        region: str | None = None,
        seats: int | None = None,
        prior_support_tickets: int | None = None,
        notes: list[str] | None = None,
    ) -> CustomerProfile:
        """Create a durable customer, leaving skipped wizard facts unknown."""
        if not plan.strip():
            raise ValueError("Customer plan must not be blank.")
        with self._connection(write=True) as connection:
            count = connection.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
            sort_order = connection.execute(
                "SELECT COALESCE(MAX(sort_order), 0) + 1 FROM customers"
            ).fetchone()[0]
            customer = CustomerProfile(
                customer_id="customer-" + uuid4().hex,
                plan=plan.strip(),
                display_name=(display_name or "").strip() or f"Customer {count + 1}",
                tenure_months=tenure_months,
                region=region,
                seats=seats,
                prior_support_tickets=prior_support_tickets,
                notes=notes or [],
            )
            connection.execute(
                "INSERT INTO customers(customer_id, profile_json, sort_order) VALUES (?, ?, ?)",
                (customer.customer_id, customer.model_dump_json(), sort_order),
            )
        return customer

    @staticmethod
    def _summary_from_row(row: sqlite3.Row) -> TicketSummary:
        return TicketSummary.model_validate(dict(row))

    def list_tickets(self, customer_id: str) -> list[TicketSummary]:
        """List one known customer's tickets, most recently active first."""
        with self._connection() as connection:
            self._require_customer(connection, customer_id)
            rows = connection.execute(
                """SELECT t.*, COUNT(m.message_id) AS message_count FROM tickets t
                   LEFT JOIN ticket_messages m ON m.ticket_id = t.ticket_id
                   WHERE t.customer_id = ? GROUP BY t.ticket_id
                   ORDER BY t.updated_at DESC, t.created_at DESC, t.ticket_id""",
                (customer_id,),
            ).fetchall()
        return [self._summary_from_row(row) for row in rows]

    def get_ticket_summary(self, ticket_id: str, customer_id: str) -> TicketSummary:
        with self._connection() as connection:
            self._require_ticket_owner(connection, ticket_id, customer_id)
            row = connection.execute(
                """SELECT t.*, COUNT(m.message_id) AS message_count FROM tickets t
                   LEFT JOIN ticket_messages m ON m.ticket_id = t.ticket_id
                   WHERE t.ticket_id = ? GROUP BY t.ticket_id""",
                (ticket_id,),
            ).fetchone()
        assert row is not None
        return self._summary_from_row(row)

    def create_ticket(self, customer_id: str, message: TicketMessage) -> TicketInput:
        """Commit a ticket and its first message together; failed writes leave no chat."""
        if not message.text.strip():
            raise ValueError("The first ticket message must not be blank.")
        ticket_id = "ticket-" + uuid4().hex
        with self._connection(write=True) as connection:
            self._require_customer(connection, customer_id)
            persisted, _ = self._append_message(connection, ticket_id, customer_id, message)
        return TicketInput(
            ticket_id=ticket_id, customer_id=customer_id,
            messages=[persisted], as_of=self._now(),
        )

    def latest_result(self, ticket_id: str, customer_id: str) -> TicketResult | None:
        with self._connection() as connection:
            self._require_ticket_owner(connection, ticket_id, customer_id)
            row = connection.execute(
                "SELECT result_json FROM triage_runs WHERE ticket_id = ? ORDER BY run_id DESC LIMIT 1",
                (ticket_id,),
            ).fetchone()
        return None if row is None else TicketResult.model_validate_json(row[0])

    def contains_message(self, ticket_id: str, message_id: str, customer_id: str) -> bool:
        with self._connection() as connection:
            self._require_ticket_owner(connection, ticket_id, customer_id)
            return connection.execute(
                "SELECT 1 FROM ticket_messages WHERE ticket_id = ? AND message_id = ?",
                (ticket_id, message_id),
            ).fetchone() is not None

    def _append_message(
        self,
        connection: sqlite3.Connection,
        ticket_id: str,
        customer_id: str,
        message: TicketMessage,
        *,
        expected_sequence: int | None = None,
    ) -> tuple[TicketMessage, bool]:
        self._ensure_owner(connection, ticket_id, customer_id)
        message_id = message.message_id or "message-" + uuid4().hex
        existing = connection.execute(
            "SELECT * FROM ticket_messages WHERE message_id = ?", (message_id,)
        ).fetchone()
        if existing is not None:
            if existing["ticket_id"] != ticket_id or not self._same_content(
                existing, message
            ):
                raise ValueError("Message ID is already used for a different message.")
            if expected_sequence is not None and existing["sequence"] != expected_sequence:
                raise ValueError("Transcript message order conflicts with stored history.")
            return self._message_from_row(existing), False

        sequence = connection.execute(
            "SELECT COALESCE(MAX(sequence), 0) + 1 FROM ticket_messages WHERE ticket_id = ?",
            (ticket_id,),
        ).fetchone()[0]
        if expected_sequence is not None and sequence != expected_sequence:
            raise ValueError("Transcript message order conflicts with stored history.")
        observed_at = (
            message.observed_at.astimezone(timezone.utc)
            if message.observed_at is not None
            else self._now()
        )
        connection.execute(
            """INSERT INTO ticket_messages
               (message_id, ticket_id, sequence, text, time_label, role, observed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                message_id, ticket_id, sequence, message.text, message.time_label,
                message.role, observed_at.isoformat(),
            ),
        )
        now = self._now().isoformat()
        connection.execute(
            """UPDATE tickets SET updated_at = MAX(updated_at, ?),
               title = CASE WHEN ? = 1 THEN ? ELSE title END WHERE ticket_id = ?""",
            (now, sequence, self._ticket_title(message.text), ticket_id),
        )
        return message.model_copy(update={
            "message_id": message_id, "observed_at": observed_at,
        }), True

    def append_message(
        self, ticket_id: str, customer_id: str, message: TicketMessage
    ) -> TicketMessage:
        """Commit an incoming message before any potentially failing provider work.

        Explicit IDs make retries idempotent. Without an ID, each call is a new message.
        """
        with self._connection(write=True) as connection:
            persisted, _ = self._append_message(
                connection, ticket_id, customer_id, message
            )
        return persisted

    def _ingest_ticket(
        self, connection: sqlite3.Connection, ticket: TicketInput
    ) -> tuple[bool, int]:
        created_ticket = self._ensure_owner(
            connection, ticket.ticket_id, ticket.customer_id
        )
        existing_rows = connection.execute(
            "SELECT * FROM ticket_messages WHERE ticket_id = ? ORDER BY sequence",
            (ticket.ticket_id,),
        ).fetchall()
        added_messages = 0
        for sequence, message in enumerate(ticket.messages, start=1):
            if sequence <= len(existing_rows):
                stored = existing_rows[sequence - 1]
                if not self._same_content(stored, message) or (
                    message.message_id is not None
                    and message.message_id != stored["message_id"]
                ):
                    raise ValueError("Transcript conflicts with stored message history.")
                continue
            identified_message = message.model_copy(
                update={
                    "message_id": message.message_id
                    or stable_message_id(ticket.ticket_id, sequence, message)
                }
            )
            _, added = self._append_message(
                connection,
                ticket.ticket_id,
                ticket.customer_id,
                identified_message,
                expected_sequence=sequence,
            )
            added_messages += int(added)
        return created_ticket, added_messages

    def ingest_ticket(self, ticket: TicketInput) -> None:
        """Persist an ordered transcript, preserving any newer stored messages.

        Existing transcript entries must match; conflicting history is never overwritten.
        """
        with self._connection(write=True) as connection:
            self._ingest_ticket(connection, ticket)

    def validate_ticket_owner(self, ticket_id: str, customer_id: str) -> None:
        """Reject a different owner before restoring any saved workflow state."""
        with self._connection() as connection:
            row = connection.execute(
                "SELECT customer_id FROM tickets WHERE ticket_id = ?", (ticket_id,)
            ).fetchone()
        if row is not None and row["customer_id"] != customer_id:
            raise ValueError("Ticket belongs to a different customer.")

    def load_history(
        self, ticket_id: str, customer_id: str | None = None
    ) -> TicketInput:
        with self._connection() as connection:
            ticket = connection.execute(
                "SELECT customer_id FROM tickets WHERE ticket_id = ?", (ticket_id,)
            ).fetchone()
            if ticket is None:
                raise KeyError(f"Ticket not found: {ticket_id}")
            if customer_id is not None and customer_id != ticket["customer_id"]:
                raise ValueError("Ticket belongs to a different customer.")
            rows = connection.execute(
                "SELECT * FROM ticket_messages WHERE ticket_id = ? ORDER BY sequence",
                (ticket_id,),
            ).fetchall()
        if not rows:
            raise ValueError("Ticket has no stored messages yet.")
        return TicketInput(
            ticket_id=ticket_id,
            customer_id=ticket["customer_id"],
            messages=[self._message_from_row(row) for row in rows],
            as_of=self._now(),
        )

    def save_result(self, result: TicketResult) -> int:
        """Record a completed run and latest analysis; a draft reply is not a sent message."""
        created_at = self._clock()
        with self._connection(write=True) as connection:
            if connection.execute(
                "SELECT 1 FROM tickets WHERE ticket_id = ?", (result.ticket_id,)
            ).fetchone() is None:
                raise KeyError(f"Ticket not found: {result.ticket_id}")
            cursor = connection.execute(
                "INSERT INTO triage_runs(ticket_id, result_json, created_at) VALUES (?, ?, ?)",
                (result.ticket_id, result.model_dump_json(), created_at),
            )
            connection.execute(
                """INSERT INTO ticket_memory(ticket_id, analysis_json, updated_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(ticket_id) DO UPDATE SET
                       analysis_json = excluded.analysis_json,
                       updated_at = excluded.updated_at""",
                (result.ticket_id, result.analysis.model_dump_json(), created_at),
            )
            connection.execute(
                "UPDATE tickets SET updated_at = MAX(updated_at, ?) WHERE ticket_id = ?",
                (datetime.fromtimestamp(created_at, tz=timezone.utc).isoformat(), result.ticket_id),
            )
            run_id = cursor.lastrowid
        assert run_id is not None
        return run_id

    def get_memory(self, ticket_id: str) -> TicketAnalysis | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT analysis_json FROM ticket_memory WHERE ticket_id = ?", (ticket_id,)
            ).fetchone()
        return None if row is None else TicketAnalysis.model_validate_json(row[0])

    @staticmethod
    def _require_ticket_owner(
        connection: sqlite3.Connection, ticket_id: str, customer_id: str
    ) -> None:
        row = connection.execute(
            "SELECT customer_id FROM tickets WHERE ticket_id = ?", (ticket_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Ticket not found: {ticket_id}")
        if row["customer_id"] != customer_id:
            raise ValueError("Ticket belongs to a different customer.")

    def get_mock_handoff(
        self, dispatch_id: str, *, ticket_id: str, customer_id: str
    ) -> HandoffReceipt | None:
        """Read a local mock acceptance only for the known ticket owner."""
        with self._connection() as connection:
            self._require_ticket_owner(connection, ticket_id, customer_id)
            row = connection.execute(
                "SELECT * FROM mock_handoffs WHERE dispatch_id = ?", (dispatch_id,)
            ).fetchone()
        if row is None:
            return None
        if row["ticket_id"] != ticket_id or row["customer_id"] != customer_id:
            raise ValueError("Mock handoff belongs to a different ticket.")
        return HandoffReceipt.model_validate_json(row["receipt_json"])

    def record_mock_handoff(
        self, ticket_id: str, customer_id: str, receipt: HandoffReceipt, payload: dict
    ) -> HandoffReceipt:
        """Persist an accepted local mock dispatch without changing message history.

        Repeated dispatch IDs retain the first acceptance time and payload.
        """
        payload_json = TypeAdapter(dict[str, Any]).dump_json(payload).decode("utf-8")
        with self._connection(write=True) as connection:
            self._require_ticket_owner(connection, ticket_id, customer_id)
            existing = connection.execute(
                "SELECT * FROM mock_handoffs WHERE dispatch_id = ?", (receipt.dispatch_id,)
            ).fetchone()
            if existing is not None:
                previous = HandoffReceipt.model_validate_json(existing["receipt_json"])
                if (
                    existing["ticket_id"] != ticket_id
                    or existing["customer_id"] != customer_id
                    or previous.model_dump(exclude={"created_at"})
                    != receipt.model_dump(exclude={"created_at"})
                ):
                    raise ValueError("Mock handoff ID is already used for a different dispatch.")
                return previous
            receipt = receipt.model_copy(update={
                "created_at": receipt.created_at.astimezone(timezone.utc)
            })
            connection.execute(
                """INSERT INTO mock_handoffs
                   (dispatch_id, ticket_id, customer_id, receipt_json, payload_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    receipt.dispatch_id, ticket_id, customer_id,
                    receipt.model_dump_json(), payload_json, receipt.created_at.isoformat(),
                ),
            )
        return receipt

    def cache_status(
        self, key: str, payload: dict, *, ttl_seconds: float = 300
    ) -> None:
        if ttl_seconds < 0:
            raise ValueError("Status cache TTL cannot be negative.")
        encoded_payload = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        with self._connection(write=True) as connection:
            connection.execute(
                """INSERT INTO system_status_cache(cache_key, payload_json, expires_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(cache_key) DO UPDATE SET
                       payload_json = excluded.payload_json,
                       expires_at = excluded.expires_at""",
                (key, encoded_payload, self._clock() + ttl_seconds),
            )

    def get_cached_status(self, key: str) -> dict | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT payload_json, expires_at FROM system_status_cache WHERE cache_key = ?",
                (key,),
            ).fetchone()
        if row is None or self._clock() >= row["expires_at"]:
            return None
        return json.loads(row["payload_json"])

    def seed_samples(
        self, customer_path: str | Path, tickets_path: str | Path
    ) -> dict[str, int]:
        """Insert sample history up to each latest message without resetting live state."""
        raw_customers = json.loads(Path(customer_path).read_text(encoding="utf-8"))
        raw_tickets = json.loads(Path(tickets_path).read_text(encoding="utf-8"))
        if not isinstance(raw_customers, dict) or not isinstance(raw_tickets, list):
            raise ValueError("Customer fixtures must be an object and tickets an array.")
        customers = {
            key: CustomerProfile.model_validate(value)
            for key, value in raw_customers.items()
        }
        tickets = [TicketInput.model_validate(value) for value in raw_tickets]
        if any(key != profile.customer_id for key, profile in customers.items()):
            raise ValueError("Customer fixture keys must match customer IDs.")
        if len({ticket.ticket_id for ticket in tickets}) != len(tickets):
            raise ValueError("Ticket fixture IDs must be unique.")
        if any(ticket.customer_id not in customers for ticket in tickets):
            raise ValueError("Every sample ticket must reference a customer fixture.")
        message_ids: set[str] = set()
        for ticket in tickets:
            for sequence, message in enumerate(ticket.messages, start=1):
                message_id = message.message_id or stable_message_id(
                    ticket.ticket_id, sequence, message
                )
                if message_id in message_ids:
                    raise ValueError("Sample message IDs must be unique.")
                message_ids.add(message_id)

        counts = {"customers": 0, "tickets": 0, "messages": 0}
        with self._connection(write=True) as connection:
            for customer in customers.values():
                sort_order = connection.execute(
                    "SELECT COALESCE(MAX(sort_order), 0) + 1 FROM customers"
                ).fetchone()[0]
                cursor = connection.execute(
                    "INSERT OR IGNORE INTO customers(customer_id, profile_json, sort_order) VALUES (?, ?, ?)",
                    (customer.customer_id, customer.model_dump_json(), sort_order),
                )
                counts["customers"] += cursor.rowcount
            for ticket in tickets:
                if len(ticket.messages) == 1:
                    counts["tickets"] += int(
                        self._ensure_owner(connection, ticket.ticket_id, ticket.customer_id)
                    )
                    continue
                history = ticket.model_copy(update={"messages": ticket.messages[:-1]})
                added_ticket, added_messages = self._ingest_ticket(connection, history)
                counts["tickets"] += int(added_ticket)
                counts["messages"] += added_messages
        return counts
