"""Exercise durable LangGraph state with real SQLite and offline provider doubles."""

import asyncio
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import pytest

import support_ticket_triage.graph as graph_module
from support_ticket_triage.checkpoints import open_checkpointer
from support_ticket_triage.database import SQLiteStore
from support_ticket_triage.graph import TicketTriageAgent
from support_ticket_triage.schemas import (
    CustomerProfile,
    ExtractedIssue,
    HandoffReceipt,
    IssueDecision,
    KnowledgeBaseArticle,
    NextAction,
    SpecialistTeam,
    TicketAnalysis,
    TicketInput,
    TicketMessage,
    TicketResult,
    TicketUpdate,
    ToolContext,
    TriageDecision,
    Urgency,
)
from test_graph import (
    FakeDecisionService,
    FakeMCPClient,
    FakeMCPTool,
    FakeResponseService,
    make_analysis,
    make_decision,
)


@pytest.fixture(autouse=True)
def isolate_checkpoint_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setattr(
        graph_module, "load_handbook", lambda: "Fictional offline checkpoint test policy."
    )


def incoming(
    ticket_id: str = "checkpoint-ticket",
    *,
    customer_id: str = "customer",
    message_id: str = "first-message",
    text: str = "An unresolved support question",
) -> TicketUpdate:
    return TicketUpdate(
        ticket_id=ticket_id,
        customer_id=customer_id,
        message=TicketMessage(text=text, message_id=f"{ticket_id}:{message_id}"),
    )


def thread_config(ticket_id: str) -> dict:
    return {"configurable": {"thread_id": ticket_id}}


def checkpoint_agent(
    store: SQLiteStore,
    saver: AsyncSqliteSaver,
    *,
    action: NextAction = NextAction.AUTO_RESPOND,
    fail_at: str | None = None,
) -> tuple[TicketTriageAgent, FakeResponseService, FakeDecisionService, FakeMCPClient]:
    response = FakeResponseService(make_analysis(), fail_at=fail_at)
    chosen = make_decision(action)
    chosen.issue_decisions = [IssueDecision(
        issue_index=0,
        urgency=chosen.urgency,
        action=chosen.action,
        owner=chosen.owner,
    )]
    decision = FakeDecisionService(chosen, fail=fail_at == "decision")
    profile = CustomerProfile(customer_id="customer", plan="Pro", tenure_months=2)
    article = KnowledgeBaseArticle(
        article_id="checkpoint-guide",
        title="Fictional support guidance",
        content="Guidance used only by an offline test.",
        source_url="fixture://checkpoint-guide",
        audience="internal",
        is_mock=True,
    )
    client = FakeMCPClient([
        FakeMCPTool(name="check_customer_history", payload=profile.model_dump(mode="json")),
        FakeMCPTool(name="lookup_knowledge_base", payload=[article.model_dump(mode="json")]),
    ])
    return (
        TicketTriageAgent(
            decision, response, mcp_client=client, store=store, checkpointer=saver,
        ),
        response,
        decision,
        client,
    )


@pytest.mark.parametrize("explicit_path", [False, True])
def test_checkpoints_survive_restart_with_application_types(
    tmp_path: Path, explicit_path: bool,
) -> None:
    database_path = tmp_path / "support.custom.sqlite3"
    checkpoint_path = (
        tmp_path / "separate" / "graph.sqlite3" if explicit_path
        else tmp_path / "support.custom.checkpoints.sqlite3"
    )
    override = checkpoint_path if explicit_path else None
    event = incoming()

    async def exercise() -> None:
        store = SQLiteStore(database_path)
        async with open_checkpointer(database_path, override) as saver:
            assert isinstance(saver, AsyncSqliteSaver)
            agent, _, _, _ = checkpoint_agent(
                store, saver, action=NextAction.ROUTE_TO_SPECIALIST,
            )
            result = await agent.run(event)
            assert result.handoff is not None
        assert checkpoint_path.is_file()

        # A new saver and graph must deserialize durable state without a rerun.
        async with open_checkpointer(database_path, override) as reopened:
            agent, response, decision, client = checkpoint_agent(
                SQLiteStore(database_path), reopened,
            )
            state = await agent.graph.aget_state(thread_config(event.ticket_id))
            values = state.values
            assert values["result"] == result
            assert isinstance(values["incoming"], TicketUpdate)
            assert isinstance(values["incoming"].message, TicketMessage)
            assert isinstance(values["ticket"], TicketInput)
            assert isinstance(values["ticket"].messages[0], TicketMessage)
            assert isinstance(values["ticket"].messages[0].observed_at, datetime)
            assert isinstance(values["context"], ToolContext)
            assert isinstance(values["context"].customer, CustomerProfile)
            assert isinstance(values["context"].articles[0], KnowledgeBaseArticle)
            assert isinstance(values["analysis"], TicketAnalysis)
            assert isinstance(values["analysis"].issues[0], ExtractedIssue)
            assert isinstance(values["decision"], TriageDecision)
            assert values["decision"].urgency is Urgency.LOW
            assert values["decision"].action is NextAction.ROUTE_TO_SPECIALIST
            assert values["decision"].owner is SpecialistTeam.GENERAL_SUPPORT
            assert isinstance(values["decision"].issue_decisions[0], IssueDecision)
            receipt = values["dispatch_receipt"]
            assert isinstance(receipt, HandoffReceipt)
            assert receipt.action is NextAction.ROUTE_TO_SPECIALIST
            assert receipt.owner is SpecialistTeam.GENERAL_SUPPORT
            assert isinstance(receipt.created_at, datetime)
            assert isinstance(values["result"], TicketResult)
            assert isinstance(values["result"].analysis, TicketAnalysis)
            assert isinstance(values["result"].decision, TriageDecision)
            assert not response.extraction_calls
            assert not decision.calls
            assert client.discovery_calls == 0

    asyncio.run(exercise())


def test_ticket_threads_keep_independent_graph_state(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "support.sqlite3")
    first_event = incoming("ticket-a", text="First ticket needs specialist investigation")
    second_event = incoming("ticket-b", text="Second ticket has a documented question")

    async def exercise() -> None:
        async with open_checkpointer(store.path) as saver:
            agent, response, decision, _ = checkpoint_agent(
                store, saver, action=NextAction.ROUTE_TO_SPECIALIST,
            )
            first = await agent.run(first_event)
            first_state = await agent.graph.aget_state(thread_config(first_event.ticket_id))
            response.analysis = make_analysis(ExtractedIssue(
                issue_type="documented_question",
                description="A distinct issue for the second ticket",
                retrieval_query="documented question",
            ))
            decision.decision = make_decision(NextAction.AUTO_RESPOND)
            second = await agent.run(second_event)
            state_a = await agent.graph.aget_state(thread_config(first_event.ticket_id))
            state_b = await agent.graph.aget_state(thread_config(second_event.ticket_id))

            assert state_a.values == first_state.values
            assert state_a.values["result"] == first
            assert state_b.values["result"] == second
            assert first.ticket_id == "ticket-a" and second.ticket_id == "ticket-b"
            assert first.handoff is not None and second.handoff is None
            assert state_a.values["dispatch_receipt"] is not None
            assert state_b.values.get("dispatch_receipt") is None
            assert state_a.values["analysis"].issues[0].issue_type == "test_issue"
            assert state_b.values["analysis"].issues[0].issue_type == "documented_question"
            for state, event in ((state_a, first_event), (state_b, second_event)):
                assert state.values["incoming"] == event
                assert [message.text for message in state.values["ticket"].messages] == [event.message.text]
            assert response.extraction_calls[1][1].previous_analysis is None

    asyncio.run(exercise())


def test_interrupted_finish_resumes_after_restart_without_repeating_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    store = SQLiteStore(tmp_path / "support.sqlite3")
    event = incoming()

    def unavailable_storage(_result):
        raise OSError("Temporary result-storage failure")

    monkeypatch.setattr(store, "save_result", unavailable_storage)

    async def exercise() -> None:
        async with open_checkpointer(store.path) as saver:
            agent, _, _, _ = checkpoint_agent(store, saver)
            with pytest.raises(OSError, match="result-storage"):
                await agent.run(event)
            interrupted = await agent.graph.aget_state(thread_config(event.ticket_id))
            assert interrupted.next == ("finish",)
            assert interrupted.values["reply"] == "A fake draft for workflow verification."
        with pytest.raises(ValueError):
            await saver.conn.execute("SELECT 1")

        reopened_store = SQLiteStore(store.path)
        async with open_checkpointer(store.path) as saver:
            agent, response, decision, client = checkpoint_agent(reopened_store, saver)
            state = await agent.graph.ainvoke(None, config=thread_config(event.ticket_id))
            assert state["result"].decision.action is NextAction.AUTO_RESPOND
            assert state["result"].reply == "A fake draft for workflow verification."
            assert not response.extraction_calls
            assert not response.reply_calls
            assert not decision.calls
            assert client.discovery_calls == 0
            completed = await agent.graph.aget_state(thread_config(event.ticket_id))
            assert completed.next == ()
            assert completed.values["result"] == state["result"]
            assert len(reopened_store.load_history(event.ticket_id).messages) == 1

    asyncio.run(exercise())


@pytest.mark.parametrize("failure", ["extract", "decision", "reply"])
def test_new_turn_clears_failed_checkpoint_state_after_restart(
    tmp_path: Path, failure: str,
) -> None:
    store = SQLiteStore(tmp_path / "support.sqlite3")
    first_event = incoming()
    follow_up = incoming(message_id="follow-up", text="Here is another detail for the question")

    async def exercise() -> None:
        async with open_checkpointer(store.path) as saver:
            agent, _, _, _ = checkpoint_agent(store, saver, fail_at=failure)
            failed = await agent.run(first_event)
            failed_state = await agent.graph.aget_state(thread_config(first_event.ticket_id))
            assert failed.decision.action is NextAction.ESCALATE_TO_HUMAN
            assert failed.errors and failed.reply and failed.handoff
            assert isinstance(failed_state.values["dispatch_receipt"], HandoffReceipt)
            if failure == "extract":
                assert failed_state.values["analysis_failed"] is True
            if failure == "reply":
                assert failed_state.values["reply_failed"] is True

        async with open_checkpointer(store.path) as saver:
            agent, response, decision, _ = checkpoint_agent(store, saver)
            healthy = await agent.run(follow_up)
            state = await agent.graph.aget_state(thread_config(follow_up.ticket_id))

            assert healthy.decision.action is NextAction.AUTO_RESPOND
            assert healthy.errors == []
            assert healthy.reply == "A fake draft for workflow verification."
            assert healthy.reply != failed.reply
            assert healthy.handoff is None
            assert state.values["analysis_failed"] is False
            assert state.values["reply_failed"] is False
            assert state.values["errors"] == []
            assert state.values.get("handoff") is None
            assert state.values.get("dispatch_receipt") is None
            assert state.values["result"] == healthy
            assert state.values["incoming"] == follow_up
            assert state.values["analysis"] == response.analysis
            assert state.values["decision"] == decision.decision
            assert len(response.extraction_calls) == len(decision.calls) == len(response.reply_calls) == 1
            assert not response.handoff_reply_calls
            assert [message.text for message in response.extraction_calls[0][0].messages] == [
                first_event.message.text, follow_up.message.text,
            ]
            # Summary memory is retained as input; workflow failure flags are not.
            assert response.extraction_calls[0][1].previous_analysis == failed.analysis

    asyncio.run(exercise())


def test_ticket_thread_id_preserves_graph_tracing_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    store = SQLiteStore(tmp_path / "support.sqlite3")
    event = incoming()
    callback = BaseCallbackHandler()
    trace_config = {
        "callbacks": [callback],
        "tags": ["offline-checkpoint-test"],
        "run_name": "OfflineTriageTrace",
        "metadata": {"ticket_id": event.ticket_id, "custom_trace": "preserved"},
        "configurable": {"thread_id": "old-tracing-value", "trace_setting": 7},
    }

    @contextmanager
    def fake_tracing(ticket_id: str):
        assert ticket_id == event.ticket_id
        yield trace_config

    monkeypatch.setattr(graph_module, "graph_tracing", fake_tracing)

    async def exercise() -> None:
        async with open_checkpointer(store.path) as saver:
            agent, _, _, _ = checkpoint_agent(store, saver)
            original_invoke = agent.graph.ainvoke
            captured = []

            async def capture_invoke(inputs, config=None, **kwargs):
                captured.append(config)
                return await original_invoke(inputs, config=config, **kwargs)

            monkeypatch.setattr(agent.graph, "ainvoke", capture_invoke)
            result = await agent.run(event)
            assert result.errors == []
            assert len(captured) == 1
            config = captured[0]
            assert config["configurable"] == {"thread_id": event.ticket_id, "trace_setting": 7}
            assert config["callbacks"] == [callback]
            assert config["tags"] == trace_config["tags"]
            assert config["run_name"] == trace_config["run_name"]
            assert config["metadata"] == trace_config["metadata"]
            assert trace_config["configurable"]["thread_id"] == "old-tracing-value"
            assert (await agent.graph.aget_state(thread_config(event.ticket_id))).values["result"] == result

    asyncio.run(exercise())


def test_ownership_mismatch_does_not_read_or_add_checkpoints(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    store = SQLiteStore(tmp_path / "support.sqlite3")
    valid_event = incoming()
    wrong_owner = incoming(customer_id="other-customer", message_id="unauthorized")

    async def exercise() -> None:
        async with open_checkpointer(store.path) as saver:
            agent, response, decision, client = checkpoint_agent(store, saver)
            await agent.run(valid_event)
            before = [entry.config async for entry in saver.alist(None)]
            assert before
            reads = []
            writes = []
            original_read = saver.aget_tuple
            original_put = saver.aput
            original_writes = saver.aput_writes

            async def record_read(*args, **kwargs):
                reads.append(args)
                return await original_read(*args, **kwargs)

            async def record_put(*args, **kwargs):
                writes.append(args)
                return await original_put(*args, **kwargs)

            async def record_writes(*args, **kwargs):
                writes.append(args)
                return await original_writes(*args, **kwargs)

            monkeypatch.setattr(saver, "aget_tuple", record_read)
            monkeypatch.setattr(saver, "aput", record_put)
            monkeypatch.setattr(saver, "aput_writes", record_writes)
            with pytest.raises(ValueError, match="different customer"):
                await agent.run(wrong_owner)

            assert reads == []
            assert writes == []
            assert [entry.config async for entry in saver.alist(None)] == before
            assert len(response.extraction_calls) == len(decision.calls) == len(response.reply_calls) == 1
            assert client.discovery_calls == 1
            history = store.load_history(valid_event.ticket_id, valid_event.customer_id)
            assert len(history.messages) == 1
            assert history.messages[0].text == valid_event.message.text

    asyncio.run(exercise())
