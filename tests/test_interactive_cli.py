"""Script the Rich terminal workflow with SQLite and an offline runner."""

import asyncio
from datetime import datetime, timezone
from io import StringIO
import json
from pathlib import Path
import sqlite3

import pytest
from rich.console import Console

from support_ticket_triage.database import SQLiteStore, stable_message_id
from support_ticket_triage.interactive import InteractiveCLI
from support_ticket_triage.schemas import (
    CustomerProfile,
    ExtractedIssue,
    NextAction,
    SpecialistTeam,
    TicketAnalysis,
    TicketInput,
    TicketMessage,
    TicketResult,
    TicketUpdate,
    TriageDecision,
    Urgency,
)


FIXTURES = Path(__file__).resolve().parents[1] / "data" / "fixtures"
DRAFT = "[bold]This is a literal draft[/bold], awaiting review."


def offline_result(ticket_id: str) -> TicketResult:
    return TicketResult(
        ticket_id=ticket_id,
        analysis=TicketAnalysis(
            product=None,
            issue_type="billing",
            customer_sentiment="frustrated",
            issues=[ExtractedIssue(issue_type="billing", description="An unverified billing issue")],
        ),
        decision=TriageDecision(
            urgency=Urgency.HIGH,
            action=NextAction.ROUTE_TO_SPECIALIST,
            owner=SpecialistTeam.BILLING,
            reason="A billing specialist must investigate the payment record.",
        ),
        reply=DRAFT,
        errors=["Offline warning shown to the operator."],
    )


class FakeRunner:
    """Require the UI to persist inputs before any provider operation."""

    def __init__(self, store: SQLiteStore, *, fail: bool = False) -> None:
        self.store = store
        self.fail = fail
        self.calls: list[TicketUpdate] = []
        self.history_at_run: list[TicketInput] = []
        self.resume_calls: list[tuple[str, str]] = []
        self.pending_tickets: set[str] = set()

    async def run(self, update: TicketUpdate) -> TicketResult:
        assert isinstance(update, TicketUpdate)
        history = self.store.load_history(update.ticket_id, update.customer_id)
        assert history.messages[-1].text == update.message.text
        assert history.messages[-1].message_id == update.message.message_id
        self.calls.append(update.model_copy(deep=True))
        self.history_at_run.append(history.model_copy(deep=True))
        if self.fail:
            raise RuntimeError("Offline runner failed after input persistence.")
        result = offline_result(update.ticket_id)
        self.store.save_result(result)
        return result

    async def resume(self, ticket_id: str, customer_id: str) -> TicketResult:
        self.store.validate_ticket_owner(ticket_id, customer_id)
        self.resume_calls.append((ticket_id, customer_id))
        self.pending_tickets.discard(ticket_id)
        result = offline_result(ticket_id)
        self.store.save_result(result)
        return result

    async def pending(self, ticket_id: str, customer_id: str) -> bool:
        self.store.validate_ticket_owner(ticket_id, customer_id)
        return ticket_id in self.pending_tickets


def scripted_session(
    monkeypatch: pytest.MonkeyPatch,
    store: SQLiteStore,
    runner: FakeRunner,
    script: str,
) -> str:
    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None, width=140)
    monkeypatch.setattr("sys.stdin", StringIO(script))
    asyncio.run(InteractiveCLI(store, runner, console=console).run())
    return output.getvalue()


def profiles(store: SQLiteStore) -> list[CustomerProfile]:
    with sqlite3.connect(store.path) as connection:
        rows = connection.execute("SELECT profile_json FROM customers ORDER BY customer_id").fetchall()
    return [CustomerProfile.model_validate_json(row[0]) for row in rows]


def seed_one_customer(store: SQLiteStore) -> CustomerProfile:
    profile = CustomerProfile(customer_id="customer", plan="Pro", tenure_months=2)
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "INSERT INTO customers(customer_id, profile_json) VALUES (?, ?)",
            (profile.customer_id, profile.model_dump_json()),
        )
    return profile


def tickets(store: SQLiteStore) -> list[tuple[str, str]]:
    with sqlite3.connect(store.path) as connection:
        return connection.execute("SELECT ticket_id, customer_id FROM tickets ORDER BY ticket_id").fetchall()


def test_invalid_customer_menu_recovers_and_exits_without_running(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    store = SQLiteStore(tmp_path / "interactive.sqlite3")
    seed_one_customer(store)
    runner = FakeRunner(store)
    output = scripted_session(monkeypatch, store, runner, "wrong\n99\n1\nB\n0\n")
    assert "customer" in output.lower()
    assert "valid" in output.lower() or "choose" in output.lower()
    assert not runner.calls and not runner.resume_calls
    assert tickets(store) == []


def test_new_customer_two_step_wizard_persists_plan_and_generated_label(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    store = SQLiteStore(tmp_path / "interactive.sqlite3")
    runner = FakeRunner(store)
    output = scripted_session(monkeypatch, store, runner, "N\n2\n1\n0\n")
    customers = profiles(SQLiteStore(store.path))
    assert len(customers) == 1
    customer = customers[0]
    assert customer.customer_id
    assert customer.display_name == "Customer 1"
    assert customer.plan == "Pro"
    assert customer.tenure_months is None
    assert customer.region is None and customer.seats is None
    assert customer.prior_support_tickets is None
    assert "Pro" in output
    assert not runner.calls


def test_optional_customer_details_validate_numbers_and_preserve_literal_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    store = SQLiteStore(tmp_path / "interactive.sqlite3")
    runner = FakeRunner(store)
    output = scripted_session(monkeypatch, store, runner, "\n".join([
        "N", "3", "2", "[bold]ACME customer[/bold]",
        "negative", "-1", "6",
        "9", "2", "Japan",
        "wrong", "0", "-1", "25",
        "wrong", "-1", "4",
        "[red]A literal account note[/red]", "0", "",
    ]))
    customers = profiles(store)
    assert len(customers) == 1
    customer = customers[0]
    assert customer.plan == "Enterprise"
    assert customer.display_name == "[bold]ACME customer[/bold]"
    assert customer.tenure_months == 6
    assert customer.region == "Japan"
    assert customer.seats == 25
    assert customer.prior_support_tickets == 4
    assert customer.notes == ["[red]A literal account note[/red]"]
    assert "[bold]ACME customer[/bold]" in output
    assert not runner.calls


def test_optional_customer_details_leave_unknown_values_unknown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    store = SQLiteStore(tmp_path / "interactive.sqlite3")
    runner = FakeRunner(store)
    scripted_session(monkeypatch, store, runner, "N\n1\n2\n\n\n3\n\n\n\n0\n")
    customer, = profiles(store)
    assert customer.plan == "Free"
    assert customer.display_name == "Customer 1"
    assert customer.tenure_months is None
    assert customer.region is None and customer.seats is None
    assert customer.prior_support_tickets is None and customer.notes == []


def test_ticket_screen_displays_history_roles_times_and_literal_markup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    store = SQLiteStore(tmp_path / "interactive.sqlite3")
    seed_one_customer(store)
    store.append_message("ticket", "customer", TicketMessage(
        text="[bold]Customer content remains literal[/bold]",
        message_id="customer-message", time_label="yesterday",
        observed_at=datetime(2026, 10, 14, 9, tzinfo=timezone.utc),
    ))
    store.append_message("ticket", "customer", TicketMessage(
        text="[red]Prior support reply remains literal[/red]",
        message_id="support-message", role="support", time_label="today",
        observed_at=datetime(2026, 10, 14, 10, tzinfo=timezone.utc),
    ))
    runner = FakeRunner(store)
    output = scripted_session(monkeypatch, store, runner, "1\n1\n0\n")
    assert "[bold]Customer content remains literal[/bold]" in output
    assert "[red]Prior support reply remains literal[/red]" in output
    assert "customer" in output.lower() and "support" in output.lower()
    assert "yesterday" in output and "today" in output
    assert "2026-10-14" in output
    assert not runner.calls
    assert len(store.load_history("ticket").messages) == 2


def test_new_ticket_is_persisted_before_runner_failure_and_failure_is_graceful(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    store = SQLiteStore(tmp_path / "interactive.sqlite3")
    seed_one_customer(store)
    runner = FakeRunner(store, fail=True)
    output = scripted_session(monkeypatch, store, runner, "1\nN\nPlease investigate my invoice\n0\n")
    assert len(runner.calls) == 1
    update = runner.calls[0]
    assert update.customer_id == "customer"
    assert update.ticket_id and update.message.message_id
    assert tickets(SQLiteStore(store.path)) == [(update.ticket_id, "customer")]
    history = store.load_history(update.ticket_id)
    assert len(history.messages) == 1
    assert history.messages[0].text == "Please investigate my invoice"
    assert history.messages[0].observed_at is not None
    assert "fail" in output.lower()
    assert "Traceback" not in output


def test_multiple_new_tickets_have_distinct_ids_and_drafts_are_not_sent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    store = SQLiteStore(tmp_path / "interactive.sqlite3")
    seed_one_customer(store)
    runner = FakeRunner(store)
    output = scripted_session(
        monkeypatch, store, runner,
        "1\nN\nFirst support question\nB\nN\nSecond support question\n0\n",
    )
    assert len(runner.calls) == 2
    assert len({update.ticket_id for update in runner.calls}) == 2
    assert len({update.message.message_id for update in runner.calls}) == 2
    assert len(tickets(store)) == 2
    for update in runner.calls:
        history = store.load_history(update.ticket_id, update.customer_id)
        assert [message.text for message in history.messages] == [update.message.text]
        assert history.messages[0].role == "customer"
        assert store.get_ticket_summary(update.ticket_id, update.customer_id).status != "resolved"
    assert DRAFT in output


def test_follow_up_is_persisted_before_failure_without_losing_prior_history(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    store = SQLiteStore(tmp_path / "interactive.sqlite3")
    seed_one_customer(store)
    store.append_message("ticket", "customer", TicketMessage(text="Original question", message_id="original"))
    runner = FakeRunner(store, fail=True)
    output = scripted_session(monkeypatch, store, runner, "1\n1\n1\nMore details about my question\n0\n")
    assert len(runner.calls) == 1
    assert runner.calls[0].ticket_id == "ticket"
    assert runner.calls[0].message.message_id != "original"
    assert [message.text for message in runner.history_at_run[0].messages] == [
        "Original question", "More details about my question",
    ]
    assert len(store.load_history("ticket").messages) == 2
    assert "fail" in output.lower()


def test_blank_ticket_and_follow_up_messages_cancel_without_running(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    store = SQLiteStore(tmp_path / "interactive.sqlite3")
    seed_one_customer(store)
    store.append_message("ticket", "customer", TicketMessage(text="Original question", message_id="original"))
    runner = FakeRunner(store)
    scripted_session(monkeypatch, store, runner, "1\nN\n\n1\n1\n\n0\n")
    assert not runner.calls
    assert tickets(store) == [("ticket", "customer")]
    assert len(store.load_history("ticket").messages) == 1


def test_pending_sample_latest_message_is_processed_once_and_survives_restart(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    store = SQLiteStore(tmp_path / "interactive.sqlite3")
    store.seed_samples(FIXTURES / "customer_accounts.json", FIXTURES / "sample_tickets.json")
    sample = TicketInput.model_validate(json.loads(
        (FIXTURES / "sample_tickets.json").read_text(encoding="utf-8")
    )[0])
    runner = FakeRunner(store)
    output = scripted_session(monkeypatch, store, runner, "1\n1\n2\n0\n")
    assert len(runner.calls) == 1
    update = runner.calls[0]
    assert update.ticket_id == sample.ticket_id
    assert update.customer_id == sample.customer_id
    assert update.message.text == sample.messages[-1].text
    assert update.message.message_id == stable_message_id(
        sample.ticket_id, len(sample.messages), sample.messages[-1],
    )
    assert len(store.load_history(sample.ticket_id).messages) == len(sample.messages)
    assert DRAFT in output
    restarted_store = SQLiteStore(store.path)
    restarted_runner = FakeRunner(restarted_store)
    scripted_session(monkeypatch, restarted_store, restarted_runner, "1\n1\n2\n0\n")
    assert not restarted_runner.calls
    assert len(restarted_store.load_history(sample.ticket_id).messages) == len(sample.messages)


def test_last_saved_result_is_visible_after_restart_without_sending_draft(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    store = SQLiteStore(tmp_path / "interactive.sqlite3")
    seed_one_customer(store)
    store.append_message("ticket", "customer", TicketMessage(text="An open question", message_id="original"))
    result = offline_result("ticket")
    store.save_result(result)
    restarted_store = SQLiteStore(store.path)
    runner = FakeRunner(restarted_store)
    output = scripted_session(monkeypatch, restarted_store, runner, "1\n1\n0\n")
    assert "high" in output.lower()
    assert "specialist" in output.lower() and "billing" in output.lower()
    assert result.decision.reason in output
    assert DRAFT in output
    assert result.errors[0] in output
    assert not runner.calls and not runner.resume_calls
    history = restarted_store.load_history("ticket")
    assert [message.text for message in history.messages] == ["An open question"]
    assert all(message.role == "customer" for message in history.messages)


def test_pending_checkpoint_resume_does_not_append_a_new_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    store = SQLiteStore(tmp_path / "interactive.sqlite3")
    seed_one_customer(store)
    store.append_message("ticket", "customer", TicketMessage(text="An open question", message_id="original"))
    runner = FakeRunner(store)
    runner.pending_tickets.add("ticket")
    output = scripted_session(monkeypatch, store, runner, "1\n1\nR\n0\n")
    assert runner.resume_calls == [("ticket", "customer")]
    assert not runner.calls
    assert len(store.load_history("ticket").messages) == 1
    assert DRAFT in output


def test_customer_switch_selects_only_that_customers_ticket(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    store = SQLiteStore(tmp_path / "interactive.sqlite3")
    seed_one_customer(store)
    second_customer = CustomerProfile(customer_id="customer-z", plan="Free", tenure_months=1)
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "INSERT INTO customers(customer_id, profile_json) VALUES (?, ?)",
            (second_customer.customer_id, second_customer.model_dump_json()),
        )
    store.append_message("first-ticket", "customer", TicketMessage(
        text="First customer's question", message_id="first-original",
    ))
    store.append_message("second-ticket", "customer-z", TicketMessage(
        text="Second customer's question", message_id="second-original",
    ))
    runner = FakeRunner(store)
    scripted_session(
        monkeypatch, store, runner,
        "1\n1\nC\n2\n1\n1\nA follow-up for the second customer\n0\n",
    )
    assert len(runner.calls) == 1
    update = runner.calls[0]
    assert update.ticket_id == "second-ticket" and update.customer_id == "customer-z"
    assert len(store.load_history("first-ticket").messages) == 1
    assert [message.text for message in store.load_history("second-ticket").messages] == [
        "Second customer's question", "A follow-up for the second customer",
    ]


@pytest.mark.parametrize("script", ["", "1\n", "1\n1\n", "N\n"])
def test_end_of_input_exits_cleanly_without_provider_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, script: str,
) -> None:
    store = SQLiteStore(tmp_path / "interactive.sqlite3")
    seed_one_customer(store)
    store.append_message("ticket", "customer", TicketMessage(text="An open question", message_id="original"))
    runner = FakeRunner(store)
    output = scripted_session(monkeypatch, store, runner, script)
    assert "Traceback" not in output
    assert not runner.calls and not runner.resume_calls
    assert len(profiles(store)) == 1
    assert len(store.load_history("ticket").messages) == 1
