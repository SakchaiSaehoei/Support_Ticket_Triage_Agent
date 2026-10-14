"""Exercise persistent state using temporary databases and no provider calls."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3

import pytest

from support_ticket_triage.database import SQLiteStore, stable_message_id
from support_ticket_triage.schemas import (
    CustomerProfile,
    ExtractedIssue,
    NextAction,
    TicketAnalysis,
    TicketInput,
    TicketMessage,
    TicketResult,
    TriageDecision,
    Urgency,
)


FIXTURES = Path(__file__).resolve().parents[1] / "data" / "fixtures"


def message(text: str, identifier: str | None = None, role: str = "customer") -> TicketMessage:
    return TicketMessage(text=text, time_label="recent", message_id=identifier, role=role)


def seed(store: SQLiteStore) -> dict[str, int]:
    return store.seed_samples(
        FIXTURES / "customer_accounts.json", FIXTURES / "sample_tickets.json"
    )


def test_order_and_roles_survive_restart(tmp_path: Path) -> None:
    path = tmp_path / "support.sqlite3"
    store = SQLiteStore(path)
    store.ingest_ticket(
        TicketInput(
            ticket_id="ticket", customer_id="customer",
            messages=[message("First"), message("Support update", role="support")],
        )
    )
    appended = store.append_message("ticket", "customer", message("Follow-up"))
    history = SQLiteStore(path).load_history("ticket", "customer")
    assert [item.text for item in history.messages] == ["First", "Support update", "Follow-up"]
    assert [item.role for item in history.messages] == ["customer", "support", "customer"]
    assert history.messages[-1].message_id == appended.message_id


def test_explicit_ids_are_idempotent_and_conflicts_are_rejected(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "support.sqlite3")
    first = store.append_message("ticket", "customer", message("Question", "request-id"))
    assert store.append_message("ticket", "customer", message("Question", "request-id")) == first
    assert len(store.load_history("ticket").messages) == 1
    with pytest.raises(ValueError, match="Message ID"):
        store.append_message("ticket", "customer", message("Changed question", "request-id"))
    with pytest.raises(ValueError, match="Message ID"):
        store.append_message("another-ticket", "customer", message("Question", "request-id"))
    with pytest.raises(KeyError):
        store.load_history("another-ticket")


def test_ticket_ownership_cannot_be_changed_or_read_as_another_customer(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "support.sqlite3")
    store.append_message("ticket", "owner", message("Private history"))
    with pytest.raises(ValueError, match="different customer"):
        store.append_message("ticket", "other", message("New message"))
    with pytest.raises(ValueError, match="different customer"):
        store.load_history("ticket", "other")
    with pytest.raises(ValueError, match="different customer"):
        store.ingest_ticket(
            TicketInput(ticket_id="ticket", customer_id="other", messages=[message("Private history")])
        )
    assert len(store.load_history("ticket", "owner").messages) == 1


def test_incoming_message_remains_persisted_when_later_work_fails(tmp_path: Path) -> None:
    path = tmp_path / "support.sqlite3"
    store = SQLiteStore(path)
    with pytest.raises(RuntimeError, match="provider unavailable"):
        store.append_message("ticket", "customer", message("Please investigate", "incoming"))
        raise RuntimeError("provider unavailable")
    assert SQLiteStore(path).load_history("ticket").messages[0].text == "Please investigate"
    assert store.get_memory("ticket") is None


def test_ingest_is_stable_rejects_reordering_and_preserves_newer_messages(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "support.sqlite3")
    original = TicketInput(
        ticket_id="ticket", customer_id="customer", messages=[message("First"), message("Second")]
    )
    store.ingest_ticket(original)
    first_ids = [item.message_id for item in store.load_history("ticket").messages]
    assert first_ids == [stable_message_id("ticket", index, item) for index, item in enumerate(original.messages, 1)]
    store.ingest_ticket(original)
    store.append_message("ticket", "customer", message("Newer", "newer"))
    store.ingest_ticket(original)
    assert [item.text for item in store.load_history("ticket").messages] == ["First", "Second", "Newer"]
    with pytest.raises(ValueError, match="conflicts"):
        store.ingest_ticket(original.model_copy(update={"messages": list(reversed(original.messages))}))
    assert [item.message_id for item in store.load_history("ticket").messages[:2]] == first_ids


def test_seed_omits_latest_and_repeated_seed_preserves_new_state(tmp_path: Path) -> None:
    path = tmp_path / "support.sqlite3"
    store = SQLiteStore(path)
    assert seed(store) == {"customers": 3, "tickets": 3, "messages": 9}
    raw_tickets = json.loads((FIXTURES / "sample_tickets.json").read_text(encoding="utf-8"))
    ticket = TicketInput.model_validate(raw_tickets[0])
    history = store.load_history(ticket.ticket_id)
    assert len(history.messages) == len(ticket.messages) - 1
    assert [item.text for item in history.messages] == [item.text for item in ticket.messages[:-1]]
    latest = ticket.messages[-1].model_copy(
        update={"message_id": stable_message_id(ticket.ticket_id, len(ticket.messages), ticket.messages[-1])}
    )
    store.append_message(ticket.ticket_id, ticket.customer_id, latest)
    customer = store.get_customer(ticket.customer_id)
    assert customer is not None
    changed_customer = customer.model_copy(update={"notes": ["Updated since seeding"]})
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE customers SET profile_json = ? WHERE customer_id = ?",
            (changed_customer.model_dump_json(), ticket.customer_id),
        )
    assert seed(SQLiteStore(path)) == {"customers": 0, "tickets": 0, "messages": 0}
    assert len(store.load_history(ticket.ticket_id).messages) == 4
    assert store.get_customer(ticket.customer_id) == changed_customer
    assert store.get_customer("unknown-customer") is None


def test_seed_validation_and_history_conflicts_do_not_partially_insert(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "support.sqlite3")
    store.append_message("ticket-001", "customer-001", message("Different stored history"))
    with pytest.raises(ValueError, match="conflicts"):
        seed(store)
    assert store.get_customer("customer-001") is None
    assert store.load_history("ticket-001").messages[0].text == "Different stored history"


def test_seed_validates_latest_message_ids_before_any_insert(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "support.sqlite3")
    raw_tickets = json.loads((FIXTURES / "sample_tickets.json").read_text(encoding="utf-8"))
    raw_tickets[0]["messages"][0]["message_id"] = "duplicated-id"
    raw_tickets[0]["messages"][-1]["message_id"] = "duplicated-id"
    fixture = tmp_path / "invalid-tickets.json"
    fixture.write_text(json.dumps(raw_tickets), encoding="utf-8")
    with pytest.raises(ValueError, match="message IDs"):
        store.seed_samples(FIXTURES / "customer_accounts.json", fixture)
    assert store.get_customer("customer-001") is None
    with pytest.raises(KeyError):
        store.load_history("ticket-001")


def test_completed_runs_and_latest_memory_persist_without_sending_reply(tmp_path: Path) -> None:
    path = tmp_path / "support.sqlite3"
    store = SQLiteStore(path)
    store.append_message("ticket", "customer", message("Question"))
    analysis = TicketAnalysis(
        product=None, issue_type="usage", customer_sentiment="neutral",
        issues=[ExtractedIssue(issue_type="usage", description="Needs documented guidance")],
    )
    result = TicketResult(
        ticket_id="ticket", analysis=analysis,
        decision=TriageDecision(urgency=Urgency.LOW, action=NextAction.AUTO_RESPOND),
        reply="A draft reply, not a sent support message.",
    )
    first_id = store.save_result(result)
    changed = analysis.model_copy(update={"customer_sentiment": "frustrated"})
    second_id = store.save_result(result.model_copy(update={"analysis": changed}))
    assert second_id > first_id
    assert SQLiteStore(path).get_memory("ticket") == changed
    assert len(store.load_history("ticket").messages) == 1
    assert store.get_memory("missing-ticket") is None
    with sqlite3.connect(path) as connection:
        rows = connection.execute("SELECT result_json FROM triage_runs ORDER BY run_id").fetchall()
    assert len(rows) == 2
    assert TicketResult.model_validate_json(rows[0][0]) == result


def test_status_cache_expires_and_survives_restart(tmp_path: Path) -> None:
    path = tmp_path / "support.sqlite3"
    now = [1000.0]
    store = SQLiteStore(path, clock=lambda: now[0])
    assert store.get_cached_status("service") is None
    store.cache_status("service", {"state": "reported", "verified": False})
    restarted = SQLiteStore(path, clock=lambda: now[0])
    now[0] = 1299.0
    assert restarted.get_cached_status("service") == {"state": "reported", "verified": False}
    now[0] = 1300.0
    assert restarted.get_cached_status("service") is None
    restarted.cache_status("empty", {}, ttl_seconds=10)
    assert restarted.get_cached_status("empty") == {}
    with pytest.raises(ValueError, match="TTL"):
        store.cache_status("service", {}, ttl_seconds=-1)


def test_observation_time_is_persisted_and_processing_time_advances(tmp_path: Path) -> None:
    path = tmp_path / "support.sqlite3"
    now = [1_700_000_000.0]
    store = SQLiteStore(path, clock=lambda: now[0])
    incoming = TicketMessage(text="Time-sensitive request", message_id="first", time_label="just now")
    inserted = store.append_message("ticket", "customer", incoming)
    first_observation = datetime.fromtimestamp(now[0], tz=timezone.utc)
    assert inserted.observed_at == first_observation
    assert inserted.observed_at.tzinfo is timezone.utc
    first_history = store.load_history("ticket")
    assert first_history.as_of == first_observation

    now[0] += 3600
    retried = store.append_message("ticket", "customer", incoming)
    assert retried.observed_at == first_observation
    later_history = SQLiteStore(path, clock=lambda: now[0]).load_history("ticket")
    assert later_history.as_of == first_observation + timedelta(hours=1)
    assert later_history.messages[0].observed_at == first_observation
    assert later_history.messages[0].time_label == "just now"
    assert len(later_history.messages) == 1
    newer = store.append_message("ticket", "customer", TicketMessage(text="Follow-up", message_id="second"))
    assert newer.observed_at == later_history.as_of
    assert store.load_history("ticket").messages[0].observed_at == first_observation


def test_supplied_observation_is_normalized_and_conflicting_retry_is_rejected(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "support.sqlite3", clock=lambda: 1_700_000_000.0)
    supplied = datetime(2026, 10, 14, 9, 0, tzinfo=timezone(timedelta(hours=7)))
    incoming = TicketMessage(text="Reported event", message_id="reported", observed_at=supplied)
    inserted = store.append_message("ticket", "customer", incoming)
    expected = datetime(2026, 10, 14, 2, 0, tzinfo=timezone.utc)
    assert inserted.observed_at == expected
    assert inserted.observed_at.tzinfo is timezone.utc
    equivalent = incoming.model_copy(update={"observed_at": expected})
    assert store.append_message("ticket", "customer", equivalent) == inserted
    with pytest.raises(ValueError, match="Message ID"):
        store.append_message(
            "ticket", "customer", incoming.model_copy(update={"observed_at": supplied + timedelta(seconds=1)})
        )
    history = store.load_history("ticket")
    assert len(history.messages) == 1
    assert history.messages[0].observed_at == expected
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT observed_at FROM ticket_messages").fetchone()[0] == expected.isoformat()


def test_prefix_ingestion_keeps_first_observation_and_validates_explicit_times(tmp_path: Path) -> None:
    now = [1_700_000_000.0]
    store = SQLiteStore(tmp_path / "support.sqlite3", clock=lambda: now[0])
    original = TicketInput(
        ticket_id="ticket", customer_id="customer", messages=[message("First"), message("Second")],
        as_of=datetime.fromtimestamp(now[0], tz=timezone.utc),
    )
    store.ingest_ticket(original)
    first_history = store.load_history("ticket")
    now[0] += 7200
    store.ingest_ticket(original)
    replay = store.load_history("ticket")
    assert [item.observed_at for item in replay.messages] == [item.observed_at for item in first_history.messages]
    assert replay.as_of == original.as_of + timedelta(hours=2)
    # A caller that supplies the original observations can also retry safely.
    store.ingest_ticket(first_history)
    conflicting_message = replay.messages[0].model_copy(update={
        "observed_at": replay.messages[0].observed_at + timedelta(seconds=1)
    })
    with pytest.raises(ValueError, match="conflicts"):
        store.ingest_ticket(replay.model_copy(update={"messages": [conflicting_message, replay.messages[1]]}))
    assert store.load_history("ticket").messages == first_history.messages


def test_legacy_migration_preserves_unknown_times_and_existing_sample_ids(tmp_path: Path) -> None:
    path = tmp_path / "legacy.sqlite3"
    raw_samples = json.loads((FIXTURES / "sample_tickets.json").read_text(encoding="utf-8"))
    sample = TicketInput.model_validate(raw_samples[0])
    customer = CustomerProfile(
        customer_id=sample.customer_id, plan="Free", tenure_months=4,
        notes=["Profile already maintained in the legacy database"],
    )
    previous = sample.messages[:-1]
    previous_ids = [stable_message_id(sample.ticket_id, index, item) for index, item in enumerate(previous, 1)]
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE customers (customer_id TEXT PRIMARY KEY, profile_json TEXT NOT NULL);
            CREATE TABLE tickets (ticket_id TEXT PRIMARY KEY, customer_id TEXT NOT NULL);
            CREATE TABLE ticket_messages (
                message_id TEXT PRIMARY KEY,
                ticket_id TEXT NOT NULL REFERENCES tickets(ticket_id),
                sequence INTEGER NOT NULL CHECK(sequence > 0),
                text TEXT NOT NULL,
                time_label TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('customer', 'support')),
                UNIQUE(ticket_id, sequence)
            );
        """)
        connection.execute("INSERT INTO customers VALUES (?, ?)", (customer.customer_id, customer.model_dump_json()))
        connection.execute("INSERT INTO tickets VALUES (?, ?)", (sample.ticket_id, sample.customer_id))
        for index, (identifier, item) in enumerate(zip(previous_ids, previous), 1):
            connection.execute(
                "INSERT INTO ticket_messages VALUES (?, ?, ?, ?, ?, ?)",
                (identifier, sample.ticket_id, index, item.text, item.time_label, item.role),
            )

    now = [1_700_000_000.0]
    store = SQLiteStore(path, clock=lambda: now[0])
    history = store.load_history(sample.ticket_id)
    assert [item.message_id for item in history.messages] == previous_ids
    assert [item.text for item in history.messages] == [item.text for item in previous]
    assert all(item.observed_at is None for item in history.messages)
    assert store.get_customer(sample.customer_id) == customer
    assert store.append_message(sample.ticket_id, sample.customer_id, history.messages[0]).observed_at is None
    assert seed(store) == {"customers": 2, "tickets": 2, "messages": 6}
    assert all(item.observed_at is None for item in store.load_history(sample.ticket_id).messages)
    assert all(item.observed_at is not None for item in store.load_history("ticket-002").messages)

    now[0] += 3600
    restarted = SQLiteStore(path, clock=lambda: now[0])
    assert seed(restarted) == {"customers": 0, "tickets": 0, "messages": 0}
    assert restarted.get_customer(sample.customer_id) == customer
    legacy_history = restarted.load_history(sample.ticket_id)
    assert legacy_history.messages == history.messages
    assert legacy_history.as_of == datetime.fromtimestamp(now[0], tz=timezone.utc)
    with sqlite3.connect(path) as connection:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(ticket_messages)")]
        assert columns.count("observed_at") == 1
        assert connection.execute(
            "SELECT observed_at FROM ticket_messages WHERE ticket_id = ?", (sample.ticket_id,)
        ).fetchall() == [(None,), (None,), (None,)]


def test_stable_sample_id_does_not_change_when_observation_time_is_added() -> None:
    raw_samples = json.loads((FIXTURES / "sample_tickets.json").read_text(encoding="utf-8"))
    sample = TicketInput.model_validate(raw_samples[0])
    legacy_identifier = "message-5fd46611e578754108e734acf0022f42f87c73618ab1a81bad48dce2fab51d66"
    assert stable_message_id(sample.ticket_id, 1, sample.messages[0]) == legacy_identifier
    observed = sample.messages[0].model_copy(update={
        "observed_at": datetime(2026, 10, 14, tzinfo=timezone.utc)
    })
    assert stable_message_id(sample.ticket_id, 1, observed) == legacy_identifier
