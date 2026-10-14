"""Verify accepted local mock deliveries without models, messaging, or network calls."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3

import pytest

import support_ticket_triage.handoffs as handoffs_module
from support_ticket_triage.database import DEFAULT_FIXTURE_DIRECTORY, SQLiteStore
from support_ticket_triage.handoffs import MockHandoffService, STAFF_DESTINATION
from support_ticket_triage.schemas import (
    ExtractedIssue, HandoffReceipt, NextAction, SpecialistTeam, TicketAnalysis,
    TicketInput, TicketMessage, TicketResult, TriageDecision, Urgency,
)


def ticket(store: SQLiteStore | None = None) -> TicketInput:
    value = TicketInput(
        ticket_id="handoff-ticket", customer_id="handoff-customer",
        messages=[TicketMessage(text="An unresolved request", message_id="event-one")],
    )
    if store is not None:
        store.ingest_ticket(value)
        return store.load_history(value.ticket_id)
    return value


def decision(
    action: NextAction = NextAction.ROUTE_TO_SPECIALIST,
    owner: SpecialistTeam = SpecialistTeam.GENERAL_SUPPORT,
) -> TriageDecision:
    return TriageDecision(urgency=Urgency.LOW, action=action, owner=owner)


def dispatch_count(store: SQLiteStore) -> int:
    with sqlite3.connect(store.path) as connection:
        return connection.execute("SELECT COUNT(*) FROM mock_handoffs").fetchone()[0]


@pytest.mark.parametrize("owner", list(SpecialistTeam))
def test_selected_mock_specialist_actually_accepts_its_packet(tmp_path: Path, owner: SpecialistTeam) -> None:
    store = SQLiteStore(tmp_path / "support.sqlite3")
    incoming = ticket(store)
    history_before = store.load_history(incoming.ticket_id).messages
    payload = {"reason": "Needs specialist attention", "messages": [item.model_dump() for item in incoming.messages]}
    service = MockHandoffService(store)
    receipt = service.dispatch(incoming, decision(owner=owner), payload)

    assert isinstance(receipt, HandoffReceipt)
    assert receipt.action == NextAction.ROUTE_TO_SPECIALIST
    assert receipt.owner == owner
    assert receipt.destination == f"mock-specialist:{owner.value}"
    assert receipt.status == "accepted"
    assert receipt.is_mock is True
    assert receipt.follow_up_requested is False
    assert receipt.created_at.utcoffset() == timedelta(0)
    accepted = service.specialists[owner].inbox[receipt.dispatch_id]
    assert accepted["payload"] == payload
    assert accepted["receipt"] == receipt
    assert all(not agent.inbox for label, agent in service.specialists.items() if label != owner)
    assert not service.staff_notifier.inbox
    assert store.get_mock_handoff(
        receipt.dispatch_id, ticket_id=incoming.ticket_id, customer_id=incoming.customer_id
    ) == receipt
    assert dispatch_count(store) == 1
    assert store.load_history(incoming.ticket_id).messages == history_before


def test_mock_staff_notification_accepts_a_follow_up_request(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "support.sqlite3")
    incoming = ticket(store)
    service = MockHandoffService(store)
    selected = decision(NextAction.ESCALATE_TO_HUMAN, SpecialistTeam.BILLING)
    payload = {"unresolved_issues": ["Requires staff authority"]}
    receipt = service.dispatch(incoming, selected, payload)

    assert receipt.action == NextAction.ESCALATE_TO_HUMAN
    assert receipt.owner == SpecialistTeam.BILLING
    assert receipt.destination == STAFF_DESTINATION
    assert receipt.status == "accepted"
    assert receipt.is_mock is True
    assert receipt.follow_up_requested is True
    assert service.staff_notifier.inbox[receipt.dispatch_id]["payload"] == payload
    assert all(not agent.inbox for agent in service.specialists.values())
    assert len(store.load_history(incoming.ticket_id).messages) == 1


def test_snapshot_retry_and_restart_preserve_first_receipt_and_payload(tmp_path: Path) -> None:
    path = tmp_path / "support.sqlite3"
    now = [1_700_000_000.0]
    store = SQLiteStore(path, clock=lambda: now[0])
    incoming = ticket(store)
    service = MockHandoffService(store)
    payload = {"reason": "Original accepted packet", "as_of": incoming.as_of}
    first = service.dispatch(incoming, decision(), payload)
    assert service.dispatch(incoming, decision(), {"reason": "Retry packet"}) == first
    assert len(service.specialists[SpecialistTeam.GENERAL_SUPPORT].inbox) == 1

    now[0] += 3600
    restarted_store = SQLiteStore(path, clock=lambda: now[0])
    refreshed = restarted_store.load_history(incoming.ticket_id)
    assert refreshed.as_of != incoming.as_of
    # Observation references may differ in a submitted packet; they are not dispatch identity.
    refreshed.messages[0].observed_at += timedelta(hours=1)
    restarted = MockHandoffService(restarted_store)
    retried = restarted.dispatch(refreshed, decision(), {"reason": "Another attempt"})
    assert retried == first
    assert retried.created_at == first.created_at
    assert all(not agent.inbox for agent in restarted.specialists.values())
    assert not restarted.staff_notifier.inbox
    assert dispatch_count(restarted_store) == 1
    with sqlite3.connect(path) as connection:
        persisted = json.loads(connection.execute("SELECT payload_json FROM mock_handoffs").fetchone()[0])
    assert persisted["reason"] == "Original accepted packet"
    assert persisted["as_of"] is not None


def test_new_follow_up_produces_a_new_receipt_without_changing_prior_delivery(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "support.sqlite3")
    incoming = ticket(store)
    service = MockHandoffService(store)
    first = service.dispatch(incoming, decision(), {"reason": "Initial request"})
    store.append_message(incoming.ticket_id, incoming.customer_id, TicketMessage(text="Additional evidence", message_id="event-two"))
    follow_up = store.load_history(incoming.ticket_id)
    second = service.dispatch(follow_up, decision(), {"reason": "New evidence"})
    assert second.dispatch_id != first.dispatch_id
    assert dispatch_count(store) == 2
    assert len(service.specialists[SpecialistTeam.GENERAL_SUPPORT].inbox) == 2
    assert len(store.load_history(incoming.ticket_id).messages) == 2


def test_action_and_owner_changes_select_new_mock_destinations(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "support.sqlite3")
    incoming = ticket(store)
    service = MockHandoffService(store)
    billing = service.dispatch(incoming, decision(owner=SpecialistTeam.BILLING), {})
    technical = service.dispatch(incoming, decision(owner=SpecialistTeam.TECHNICAL_SUPPORT), {})
    staff = service.dispatch(incoming, decision(NextAction.ESCALATE_TO_HUMAN, SpecialistTeam.BILLING), {})
    assert len({billing.dispatch_id, technical.dispatch_id, staff.dispatch_id}) == 3
    assert len({billing.destination, technical.destination, staff.destination}) == 3
    assert dispatch_count(store) == 3


def test_auto_response_does_not_dispatch_or_create_a_receipt(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "support.sqlite3")
    service = MockHandoffService(store)
    with pytest.raises(ValueError, match="Automatic responses"):
        service.dispatch(ticket(store), decision(NextAction.AUTO_RESPOND), {})
    assert dispatch_count(store) == 0
    assert not service.staff_notifier.inbox
    assert all(not agent.inbox for agent in service.specialists.values())


def test_unknown_or_wrong_ticket_owner_is_rejected_before_inbox_acceptance(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "support.sqlite3")
    incoming = ticket(store)
    service = MockHandoffService(store)
    with pytest.raises(ValueError, match="different customer"):
        service.dispatch(incoming.model_copy(update={"customer_id": "other-customer"}), decision(), {})
    with pytest.raises(KeyError, match="Ticket not found"):
        service.dispatch(incoming.model_copy(update={"ticket_id": "unknown-ticket"}), decision(), {})
    assert dispatch_count(store) == 0
    assert all(not agent.inbox for agent in service.specialists.values())
    assert not service.staff_notifier.inbox
    assert len(store.load_history(incoming.ticket_id, incoming.customer_id).messages) == 1
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM tickets").fetchone()[0] == 1


def test_receipt_storage_rejects_conflicting_ids_and_wrong_ownership_atomically(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "support.sqlite3")
    incoming = ticket(store)
    first = MockHandoffService(store).dispatch(incoming, decision(owner=SpecialistTeam.BILLING), {"reason": "Original"})
    with pytest.raises(ValueError, match="different customer"):
        store.record_mock_handoff(incoming.ticket_id, "other-customer", first, {})
    with pytest.raises(ValueError, match="different dispatch"):
        store.record_mock_handoff(incoming.ticket_id, incoming.customer_id, first.model_copy(update={
            "owner": SpecialistTeam.TECHNICAL_SUPPORT, "destination": "mock-specialist:technical_support"
        }), {})
    assert store.get_mock_handoff(first.dispatch_id, ticket_id=incoming.ticket_id, customer_id=incoming.customer_id) == first
    assert dispatch_count(store) == 1
    with sqlite3.connect(store.path) as connection:
        assert json.loads(connection.execute("SELECT payload_json FROM mock_handoffs").fetchone()[0]) == {"reason": "Original"}


def test_without_store_uses_only_memory_and_preserves_idempotency(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden_store(*args, **kwargs):
        pytest.fail("An in-memory handoff service must not create a default database.")

    monkeypatch.setattr(handoffs_module.SQLiteStore, "__init__", forbidden_store)
    service = MockHandoffService()
    incoming = ticket()
    first = service.dispatch(incoming, decision(), {"reason": "Standalone"})
    assert service.dispatch(incoming, decision(), {"reason": "Retry"}) is first
    assert len(service.specialists[SpecialistTeam.GENERAL_SUPPORT].inbox) == 1
    with pytest.raises(ValueError, match="different customer"):
        service.dispatch(incoming.model_copy(update={"customer_id": "other"}), decision(), {})
    follow_up = incoming.model_copy(update={"messages": [*incoming.messages, TicketMessage(text="New turn")]})
    second = service.dispatch(follow_up, decision(), {})
    assert second.dispatch_id != first.dispatch_id


def test_new_handoff_table_migration_preserves_existing_records(tmp_path: Path) -> None:
    path = tmp_path / "existing.sqlite3"
    store = SQLiteStore(path, clock=lambda: 1000.0)
    store.seed_samples(
        DEFAULT_FIXTURE_DIRECTORY / "customer_accounts.json",
        DEFAULT_FIXTURE_DIRECTORY / "sample_tickets.json",
    )
    history = store.load_history("ticket-001")
    analysis = TicketAnalysis(
        product=None, issue_type="test_issue", customer_sentiment="neutral",
        issues=[ExtractedIssue(issue_type="test_issue", description="Preserved previous analysis")],
    )
    store.save_result(TicketResult(ticket_id=history.ticket_id, analysis=analysis, decision=decision()))
    store.cache_status("default", {"status": "unknown"})
    tables = ["customers", "tickets", "ticket_messages", "triage_runs", "ticket_memory", "system_status_cache"]
    with sqlite3.connect(path) as connection:
        before = {name: connection.execute(f"SELECT * FROM {name}").fetchall() for name in tables}
        connection.execute("DROP TABLE mock_handoffs")
    migrated = SQLiteStore(path, clock=lambda: 1000.0)
    with sqlite3.connect(path) as connection:
        after = {name: connection.execute(f"SELECT * FROM {name}").fetchall() for name in tables}
    assert after == before
    assert migrated.get_memory(history.ticket_id) == analysis
    assert migrated.get_cached_status("default") == {"status": "unknown"}
    receipt = MockHandoffService(migrated).dispatch(history, decision(), {"reason": "After migration"})
    restarted = SQLiteStore(path, clock=lambda: 1000.0)
    assert MockHandoffService(restarted).dispatch(history, decision(), {}) == receipt
    assert dispatch_count(restarted) == 1
    with sqlite3.connect(path) as connection:
        final = {name: connection.execute(f"SELECT * FROM {name}").fetchall() for name in tables}
    assert final == before
