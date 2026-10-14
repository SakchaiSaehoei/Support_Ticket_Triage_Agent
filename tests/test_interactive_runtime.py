"""Verify lazy interactive resources and durable resume without provider calls."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

import support_ticket_triage.cli as cli
import support_ticket_triage.graph as graph_module
from support_ticket_triage.checkpoints import checkpoint_path_for, open_checkpointer
from support_ticket_triage.database import SQLiteStore
from support_ticket_triage.schemas import TicketMessage
from test_checkpoints import checkpoint_agent, incoming, thread_config


@pytest.fixture(autouse=True)
def isolate_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SUPPORT_CHECKPOINT_PATH", raising=False)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setattr(cli, "load_dotenv", lambda *_: None)
    monkeypatch.setattr(cli, "flush_traces", lambda: None)
    monkeypatch.setattr(
        graph_module, "load_handbook", lambda: "Fictional offline interactive runtime policy."
    )


def forbid_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args, **kwargs):
        pytest.fail("Browsing or rejected resume must not initialize provider resources.")

    monkeypatch.setattr(cli, "Setting", forbidden)
    monkeypatch.setattr(cli, "openai_client", forbidden)
    monkeypatch.setattr(cli, "configured_agent", forbidden)


class OfflineClient:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.closed = False

    async def __aenter__(self):
        self.events.append("client opened")
        return self

    async def __aexit__(self, *_):
        self.closed = True
        self.events.append("client closed")


def configured_factories(
    monkeypatch: pytest.MonkeyPatch, store: SQLiteStore, *, fail_first: bool = False,
):
    events: list[str] = []
    settings = SimpleNamespace(checkpoint_path=None)
    clients: list[OfflineClient] = []
    savers = []
    agents = []

    def make_settings():
        events.append("settings loaded")
        return settings

    def make_client(actual_settings):
        assert actual_settings is settings
        client = OfflineClient(events)
        clients.append(client)
        return client

    def make_agent(actual_settings, actual_client, actual_store, *, checkpointer):
        assert actual_settings is settings
        assert actual_client is clients[-1]
        assert actual_store is store
        events.append("agent configured")
        savers.append(checkpointer)
        if fail_first and len(savers) == 1:
            raise RuntimeError("Offline agent configuration failed.")
        configured = checkpoint_agent(store, checkpointer)
        agents.append(configured)
        return configured[0]

    monkeypatch.setattr(cli, "Setting", make_settings)
    monkeypatch.setattr(cli, "openai_client", make_client)
    monkeypatch.setattr(cli, "configured_agent", make_agent)
    return events, clients, savers, agents


async def assert_saver_closed(saver) -> None:
    with pytest.raises(ValueError, match="no active connection"):
        await saver.conn.execute("SELECT 1")


async def interrupted_checkpoint(
    store: SQLiteStore,
    monkeypatch: pytest.MonkeyPatch,
    *,
    metadata_change: dict[str, str] | None = None,
):
    """Crash only after model work, leaving the finish node ready for durable resume."""
    event = incoming()

    def crash_finish(_result):
        raise RuntimeError("Offline persistence interruption.")

    with monkeypatch.context() as failed_finish:
        failed_finish.setattr(store, "save_result", crash_finish)
        async with open_checkpointer(store.path) as saver:
            agent, response, decision, client = checkpoint_agent(store, saver)
            with pytest.raises(RuntimeError, match="persistence interruption"):
                await agent.run(event)
            config = thread_config(event.ticket_id)
            snapshot = await agent.graph.aget_state(config)
            assert snapshot.next == ("finish",)
            assert len(response.extraction_calls) == len(response.reply_calls) == len(decision.calls) == 1
            if metadata_change:
                await agent.graph.aupdate_state(
                    config,
                    {"incoming": event.model_copy(update=metadata_change)},
                    as_node="draft_reply",
                )
                assert (await agent.graph.aget_state(config)).next == ("finish",)
    return event, response, decision, client


def test_browsing_and_pending_checks_do_not_initialize_provider_resources(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    store = SQLiteStore(tmp_path / "support.sqlite3")
    customer = store.create_customer(plan="Pro")
    ticket = store.create_ticket(customer.customer_id, TicketMessage(text="A saved question"))
    forbid_providers(monkeypatch)

    async def exercise() -> None:
        async with cli.InteractiveRunner(store) as runner:
            assert store.list_customers() == [customer]
            assert store.list_tickets(customer.customer_id)[0].ticket_id == ticket.ticket_id
            assert await runner.pending(ticket.ticket_id, customer.customer_id) is False
            assert await runner.pending("absent-ticket", customer.customer_id) is False
        assert not checkpoint_path_for(store.path).exists()

    asyncio.run(exercise())


def test_two_turns_share_one_client_and_saver_then_close_on_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    store = SQLiteStore(tmp_path / "support.sqlite3")
    events, clients, savers, agents = configured_factories(monkeypatch, store)
    first = incoming()
    second = incoming(message_id="second-message", text="Additional context for the same ticket")

    async def exercise() -> None:
        async with cli.InteractiveRunner(store) as runner:
            first_result = await runner.run(first)
            second_result = await runner.run(second)
            assert first_result.ticket_id == second_result.ticket_id == first.ticket_id
            assert len(clients) == len(savers) == len(agents) == 1
            assert not clients[0].closed
            assert await runner.pending(first.ticket_id, first.customer_id) is False
            assert len(agents[0][1].extraction_calls) == len(agents[0][2].calls) == 2
            assert len(store.load_history(first.ticket_id).messages) == 2
        assert clients[0].closed
        await assert_saver_closed(savers[0])

    asyncio.run(exercise())
    assert events == ["settings loaded", "client opened", "agent configured", "client closed"]


def test_configuration_failure_closes_partial_resources_and_allows_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    store = SQLiteStore(tmp_path / "support.sqlite3")
    events, clients, savers, agents = configured_factories(monkeypatch, store, fail_first=True)
    event = incoming()

    async def exercise() -> None:
        async with cli.InteractiveRunner(store) as runner:
            with pytest.raises(RuntimeError, match="configuration failed"):
                await runner.run(event)
            assert len(clients) == len(savers) == 1
            assert clients[0].closed
            await assert_saver_closed(savers[0])
            result = await runner.run(event)
            assert result.ticket_id == event.ticket_id
            assert len(clients) == len(savers) == 2
            assert not clients[1].closed
            assert len(agents) == 1
            assert len(store.load_history(event.ticket_id).messages) == 1
        assert all(client.closed for client in clients)
        await assert_saver_closed(savers[1])

    asyncio.run(exercise())
    assert events == [
        "settings loaded", "client opened", "agent configured", "client closed",
        "settings loaded", "client opened", "agent configured", "client closed",
    ]


def test_reopened_interrupted_progress_is_inspected_without_model_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    store = SQLiteStore(tmp_path / "support.sqlite3")

    async def exercise() -> None:
        event, _, _, _ = await interrupted_checkpoint(store, monkeypatch)
        forbid_providers(monkeypatch)
        assert checkpoint_path_for(store.path).is_file()
        async with cli.InteractiveRunner(SQLiteStore(store.path)) as runner:
            assert await runner.pending(event.ticket_id, event.customer_id) is True
            assert await runner.pending(event.ticket_id, event.customer_id) is True
        assert len(store.load_history(event.ticket_id).messages) == 1
        assert store.latest_result(event.ticket_id, event.customer_id) is None

    asyncio.run(exercise())


def test_resume_finishes_saved_work_without_new_messages_or_repeated_model_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    store = SQLiteStore(tmp_path / "support.sqlite3")

    async def exercise() -> None:
        event, previous_response, previous_decision, _ = await interrupted_checkpoint(store, monkeypatch)
        history_before = store.load_history(event.ticket_id)
        events, clients, savers, agents = configured_factories(monkeypatch, store)
        async with cli.InteractiveRunner(store) as runner:
            result = await runner.resume(event.ticket_id, event.customer_id)
            assert result.analysis == previous_response.analysis
            assert result.decision == previous_decision.decision
            assert result.reply == "A fake draft for workflow verification."
            assert result.errors == []
            assert store.latest_result(event.ticket_id, event.customer_id) == result
            assert await runner.pending(event.ticket_id, event.customer_id) is False
            assert len(agents) == 1
            _, response, decision, mcp = agents[0]
            assert not response.extraction_calls
            assert not response.reply_calls
            assert not response.handoff_reply_calls
            assert not decision.calls
            assert mcp.discovery_calls == 0
            assert store.load_history(event.ticket_id).messages == history_before.messages
        assert clients[0].closed
        await assert_saver_closed(savers[0])
        assert events == ["settings loaded", "client opened", "agent configured", "client closed"]

    asyncio.run(exercise())


def test_resume_rejects_another_customer_before_provider_initialization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    store = SQLiteStore(tmp_path / "support.sqlite3")

    async def exercise() -> None:
        event, _, _, _ = await interrupted_checkpoint(store, monkeypatch)
        forbid_providers(monkeypatch)
        async with cli.InteractiveRunner(store) as runner:
            with pytest.raises(ValueError, match="different customer"):
                await runner.pending(event.ticket_id, "other-customer")
            with pytest.raises(ValueError, match="different customer"):
                await runner.resume(event.ticket_id, "other-customer")
        assert len(store.load_history(event.ticket_id).messages) == 1
        assert store.latest_result(event.ticket_id, event.customer_id) is None

    asyncio.run(exercise())


@pytest.mark.parametrize("metadata_change", [
    {"customer_id": "other-customer"}, {"ticket_id": "another-ticket"},
])
def test_saved_ticket_metadata_mismatch_rejects_resume_without_provider_resources(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, metadata_change: dict[str, str],
) -> None:
    store = SQLiteStore(tmp_path / "support.sqlite3")

    async def exercise() -> None:
        event, _, _, _ = await interrupted_checkpoint(
            store, monkeypatch, metadata_change=metadata_change,
        )
        forbid_providers(monkeypatch)
        async with cli.InteractiveRunner(store) as runner:
            with pytest.raises(ValueError, match="Saved progress belongs"):
                await runner.pending(event.ticket_id, event.customer_id)
            with pytest.raises(ValueError, match="Saved progress belongs"):
                await runner.resume(event.ticket_id, event.customer_id)
        assert len(store.load_history(event.ticket_id).messages) == 1
        assert store.latest_result(event.ticket_id, event.customer_id) is None

    asyncio.run(exercise())


@pytest.mark.parametrize("existing_ticket", [False, True])
def test_resume_without_interrupted_work_rejects_without_provider_initialization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, existing_ticket: bool,
) -> None:
    store = SQLiteStore(tmp_path / "support.sqlite3")
    event = incoming()
    if existing_ticket:
        store.append_message(event.ticket_id, event.customer_id, event.message)
    forbid_providers(monkeypatch)

    async def exercise() -> None:
        async with cli.InteractiveRunner(store) as runner:
            with pytest.raises(ValueError, match="no interrupted processing"):
                await runner.resume(event.ticket_id, event.customer_id)
        assert not checkpoint_path_for(store.path).exists()
        if existing_ticket:
            assert len(store.load_history(event.ticket_id).messages) == 1

    asyncio.run(exercise())
