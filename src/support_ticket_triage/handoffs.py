"""Execute handoffs into local mock inboxes; no model or external messaging calls."""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from typing import Any

from pydantic import TypeAdapter

from .database import SQLiteStore, stable_message_id
from .schemas import (
    HandoffReceipt, NextAction, SpecialistTeam, TicketInput, TriageDecision,
)


STAFF_DESTINATION = "mock-staff:review-queue"


class MockSpecialistAgent:
    """Accept a context packet into one selected specialist's mock inbox."""

    def __init__(self, owner: SpecialistTeam) -> None:
        self.owner = owner
        self.destination = f"mock-specialist:{owner.value}"
        self.inbox: dict[str, dict] = {}

    def receive_handoff(
        self, ticket: TicketInput, decision: TriageDecision, payload: dict,
        *, dispatch_id: str, created_at: datetime,
    ) -> HandoffReceipt:
        if decision.action != NextAction.ROUTE_TO_SPECIALIST or decision.owner != self.owner:
            raise ValueError("The specialist does not match the requested routing action.")
        receipt = HandoffReceipt(
            dispatch_id=dispatch_id, action=decision.action, owner=self.owner,
            destination=self.destination, created_at=created_at,
        )
        self.inbox[dispatch_id] = {
            "ticket_id": ticket.ticket_id, "customer_id": ticket.customer_id,
            "payload": deepcopy(payload), "receipt": receipt,
        }
        return receipt


class MockStaffNotifier:
    """Accept a staff-review notification and explicitly request follow-up."""

    def __init__(self) -> None:
        self.destination = STAFF_DESTINATION
        self.inbox: dict[str, dict] = {}

    def notify(
        self, ticket: TicketInput, decision: TriageDecision, payload: dict,
        *, dispatch_id: str, created_at: datetime,
    ) -> HandoffReceipt:
        if decision.action != NextAction.ESCALATE_TO_HUMAN:
            raise ValueError("A staff notification requires a human escalation action.")
        receipt = HandoffReceipt(
            dispatch_id=dispatch_id, action=decision.action, owner=decision.owner,
            destination=self.destination, follow_up_requested=True, created_at=created_at,
        )
        self.inbox[dispatch_id] = {
            "ticket_id": ticket.ticket_id, "customer_id": ticket.customer_id,
            "payload": deepcopy(payload), "receipt": receipt,
        }
        return receipt


class MockHandoffService:
    """Dispatch each ticket snapshot once, optionally retaining receipts in SQLite."""

    def __init__(self, store: SQLiteStore | None = None) -> None:
        self.store = store
        self.specialists = {owner: MockSpecialistAgent(owner) for owner in SpecialistTeam}
        self.staff_notifier = MockStaffNotifier()
        self._receipts: dict[str, HandoffReceipt] = {}
        self._ticket_owners: dict[str, str] = {}

    @staticmethod
    def _dispatch_id(
        ticket: TicketInput, decision: TriageDecision, destination: str
    ) -> str:
        # Time references advance between attempts; source identities and content do not.
        identity = {
            "ticket_id": ticket.ticket_id, "customer_id": ticket.customer_id,
            "messages": [{
                "message_id": message.message_id
                or stable_message_id(ticket.ticket_id, sequence, message),
                "text": message.text, "time_label": message.time_label, "role": message.role,
            } for sequence, message in enumerate(ticket.messages, start=1)],
            "action": decision.action.value, "owner": decision.owner.value,
            "destination": destination,
        }
        serialized = json.dumps(identity, ensure_ascii=False, sort_keys=True)
        return "mock-handoff-" + hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def dispatch(
        self, ticket: TicketInput, decision: TriageDecision, payload: dict
    ) -> HandoffReceipt:
        if decision.action == NextAction.ROUTE_TO_SPECIALIST:
            receiver = self.specialists[decision.owner]
            destination = receiver.destination
        elif decision.action == NextAction.ESCALATE_TO_HUMAN:
            destination = self.staff_notifier.destination
        else:
            raise ValueError("Automatic responses do not dispatch a handoff.")

        dispatch_id = self._dispatch_id(ticket, decision, destination)
        if self.store is not None:
            existing = self.store.get_mock_handoff(
                dispatch_id, ticket_id=ticket.ticket_id, customer_id=ticket.customer_id
            )
        else:
            owner = self._ticket_owners.get(ticket.ticket_id)
            if owner is not None and owner != ticket.customer_id:
                raise ValueError("Ticket belongs to a different customer.")
            existing = self._receipts.get(dispatch_id)
        if existing is not None:
            return existing

        # Check that the packet can be persisted before accepting it into an inbox.
        TypeAdapter(dict[str, Any]).dump_json(payload)
        created_at = datetime.now(timezone.utc)
        if decision.action == NextAction.ROUTE_TO_SPECIALIST:
            receipt = receiver.receive_handoff(
                ticket, decision, payload, dispatch_id=dispatch_id, created_at=created_at,
            )
        else:
            receipt = self.staff_notifier.notify(
                ticket, decision, payload, dispatch_id=dispatch_id, created_at=created_at,
            )
        if self.store is not None:
            receipt = self.store.record_mock_handoff(
                ticket.ticket_id, ticket.customer_id, receipt, payload
            )
        else:
            self._receipts[dispatch_id] = receipt
            self._ticket_owners[ticket.ticket_id] = ticket.customer_id
        return receipt
