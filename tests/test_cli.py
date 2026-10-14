"""Verify terminal inputs and shared-client wiring without provider calls."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import support_ticket_triage.cli as cli
from support_ticket_triage.database import SQLiteStore
from support_ticket_triage.schemas import TicketInput, TicketMessage, TicketUpdate


def test_load_ticket_accepts_latest_message_without_requiring_history(tmp_path: Path):
    path = tmp_path / "latest.json"
    path.write_text(json.dumps({
        "ticket_id": "ticket", "customer_id": "customer",
        "message": {"text": "The earlier problem is still unresolved.", "message_id": "retry-id"},
    }))
    incoming = cli.load_ticket(path)
    assert isinstance(incoming, TicketUpdate)
    assert incoming.message.message_id == "retry-id"
    assert incoming.message.time_label == "just now"


@pytest.mark.parametrize("data", [[], {
    "ticket_id": "ticket", "customer_id": "customer", "message": {}, "messages": [],
}])
def test_load_ticket_rejects_ambiguous_or_nonobject_input(tmp_path: Path, data):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        cli.load_ticket(path)


def test_seed_cli_does_not_construct_model_settings_or_duplicate_history(monkeypatch, tmp_path, capsys):
    path = tmp_path / "history.sqlite3"
    monkeypatch.setattr(cli, "load_dotenv", lambda *_: None)

    def forbidden_settings():
        pytest.fail("Seeding must not require model settings or an API key.")

    monkeypatch.setattr(cli, "Setting", forbidden_settings)
    monkeypatch.setattr("sys.argv", ["triage", "--seed-samples", "--database", str(path)])
    cli.main()
    assert json.loads(capsys.readouterr().out) == {"customers": 3, "tickets": 3, "messages": 9}
    cli.main()
    assert json.loads(capsys.readouterr().out) == {"customers": 0, "tickets": 0, "messages": 0}
    assert len(SQLiteStore(path).load_history("ticket-003").messages) == 3


def test_message_cli_requires_identity_before_opening_store(monkeypatch):
    monkeypatch.setattr("sys.argv", ["triage", "--message", "Please help"])
    monkeypatch.setattr(cli, "SQLiteStore", lambda *_: pytest.fail("Invalid input opened storage."))
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2


def test_batch_uses_one_shared_client_and_closes_it(monkeypatch, tmp_path, capsys):
    events = []
    settings = SimpleNamespace(checkpoint_path=None)

    class Client:
        async def __aenter__(self):
            events.append("open")
            return self

        async def __aexit__(self, *_):
            events.append("close")

    client = Client()
    store = SQLiteStore(tmp_path / "history.sqlite3")

    class Agent:
        async def run(self, incoming):
            events.append(incoming.ticket_id)
            return SimpleNamespace(model_dump_json=lambda **_: '{"result":"offline"}')

    def configured_agent(actual_settings, actual_client, actual_store, *, checkpointer):
        assert (actual_settings, actual_client, actual_store) == (settings, client, store)
        events.append("configure")
        assert checkpointer is not None
        return Agent()

    monkeypatch.setattr(cli, "Setting", lambda: settings)
    monkeypatch.setattr(cli, "openai_client", lambda actual: client if actual is settings else None)
    monkeypatch.setattr(cli, "configured_agent", configured_agent)
    tickets = [TicketInput(
        ticket_id=name, customer_id="customer", messages=[TicketMessage(text="Question")],
    ) for name in ("one", "two")]
    asyncio.run(cli._run_tickets(tickets, store))
    assert events == ["open", "configure", "one", "two", "close"]
    assert capsys.readouterr().out.count('"result"') == 2


def test_cli_accepts_user_message_and_preserves_identity(monkeypatch, tmp_path, capsys):
    received = []
    monkeypatch.setattr(cli, "load_dotenv", lambda *_: None)

    async def run_tickets(tickets, store):
        received.extend(tickets)
        assert store.path == (tmp_path / "user.sqlite3").resolve()
        print('{"result":"offline-user-input"}')

    monkeypatch.setattr(cli, "_run_tickets", run_tickets)
    monkeypatch.setattr("sys.argv", [
        "triage", "--message", "I cannot export my report.",
        "--ticket-id", "user-ticket", "--customer-id", "customer-001",
        "--message-id", "user-message-001", "--database", str(tmp_path / "user.sqlite3"),
    ])
    cli.main()
    assert len(received) == 1
    assert isinstance(received[0], TicketUpdate)
    assert received[0].ticket_id == "user-ticket"
    assert received[0].customer_id == "customer-001"
    assert received[0].message.text == "I cannot export my report."
    assert received[0].message.message_id == "user-message-001"
    assert json.loads(capsys.readouterr().out) == {"result": "offline-user-input"}


@pytest.mark.parametrize("message_id", ["", "   "])
def test_cli_rejects_blank_explicit_message_id_before_running_models(monkeypatch, tmp_path, message_id):
    monkeypatch.setattr(cli, "load_dotenv", lambda *_: None)
    monkeypatch.setattr(cli, "_run_tickets", lambda *_: pytest.fail("Invalid input reached model execution."))
    monkeypatch.setattr("sys.argv", [
        "triage", "--message", "A user question", "--ticket-id", "ticket",
        "--customer-id", "customer", "--message-id", message_id,
        "--database", str(tmp_path / "user.sqlite3"),
    ])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


@pytest.mark.parametrize("arguments", [[], ["--interactive"]])
def test_terminal_defaults_to_interactive_without_constructing_model_settings(
    monkeypatch, tmp_path, arguments,
):
    opened = []
    monkeypatch.setattr(cli, "load_dotenv", lambda *_: None)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr(cli, "Setting", lambda: pytest.fail("Browsing constructed model settings."))

    async def interactive(store, *, no_color=False):
        opened.append((store.path, no_color))

    monkeypatch.setattr(cli, "run_interactive", interactive)
    database = tmp_path / "interactive.sqlite3"
    monkeypatch.setattr("sys.argv", ["triage", *arguments, "--database", str(database), "--no-color"])
    cli.main()
    assert opened == [(database.resolve(), True)]


@pytest.mark.parametrize("arguments", [[], ["--interactive"]])
def test_interactive_without_terminal_fails_before_opening_database(monkeypatch, arguments, capsys):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("sys.argv", ["triage", *arguments])
    monkeypatch.setattr(cli, "SQLiteStore", lambda *_: pytest.fail("Nonterminal input opened storage."))
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "requires a terminal" in capsys.readouterr().err
