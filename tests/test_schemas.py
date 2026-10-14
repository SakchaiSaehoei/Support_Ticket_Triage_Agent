"""Validate incoming message identity and the meaning of temporal references."""

from datetime import datetime, timedelta, timezone

from pydantic import ValidationError
import pytest

from support_ticket_triage.schemas import TicketInput, TicketMessage, TicketUpdate


@pytest.mark.parametrize("identifier", ["", " ", "\t\n", "\u2003"])
def test_explicit_blank_message_ids_are_rejected(identifier: str) -> None:
    with pytest.raises(ValidationError, match="message_id must not be blank"):
        TicketMessage(text="A valid message", message_id=identifier)
    with pytest.raises(ValidationError, match="message_id must not be blank"):
        TicketUpdate.model_validate({
            "ticket_id": "ticket", "customer_id": "customer",
            "message": {"text": "A valid message", "message_id": identifier},
        })


@pytest.mark.parametrize("identifier", [None, "event-001", " event-001 "])
def test_valid_message_identifiers_are_preserved_without_trimming(identifier: str | None) -> None:
    assert TicketMessage(text="A valid message", message_id=identifier).message_id == identifier


@pytest.mark.parametrize("field", ["observed_at", "as_of"])
@pytest.mark.parametrize("value", [datetime(2026, 10, 14, 9), "2026-10-14T09:00:00"])
def test_temporal_references_require_timezone_awareness(field: str, value) -> None:
    with pytest.raises(ValidationError, match="timezone"):
        if field == "observed_at":
            TicketMessage(text="A valid message", observed_at=value)
        else:
            TicketInput(
                ticket_id="ticket", customer_id="customer",
                messages=[TicketMessage(text="A valid message")], as_of=value,
            )


def test_aware_temporal_references_preserve_the_supplied_instant() -> None:
    supplied = datetime(2026, 10, 14, 9, tzinfo=timezone(timedelta(hours=7)))
    message = TicketMessage(text="A valid message", observed_at=supplied)
    ticket = TicketInput(ticket_id="ticket", customer_id="customer", messages=[message], as_of=supplied)
    assert message.observed_at == datetime(2026, 10, 14, 2, tzinfo=timezone.utc)
    assert ticket.as_of == message.observed_at
    assert TicketMessage(text="Timing not yet observed").observed_at is None
    assert TicketInput(ticket_id="ticket", customer_id="customer", messages=[message]).as_of is None
