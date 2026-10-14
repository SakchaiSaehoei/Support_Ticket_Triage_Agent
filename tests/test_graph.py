"""Test orchestration with fake services; these are not expected sample classifications."""

import asyncio
from collections.abc import Callable
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Literal

from langchain_core.tools import BaseTool, ToolException
from pydantic import Field
import pytest

import support_ticket_triage.graph as graph_module
from support_ticket_triage.database import SQLiteStore, stable_message_id
from support_ticket_triage.graph import TicketTriageAgent
from support_ticket_triage.schemas import (
    ExtractedIssue,
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
from support_ticket_triage.tools import SupportTools


FIXTURES = Path(__file__).resolve().parents[1] / "data" / "fixtures"
HANDBOOK = "Fictional internal policy for this isolated graph test."


@pytest.fixture(autouse=True)
def isolate_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setattr(graph_module, "load_handbook", lambda: HANDBOOK)


class FakeMCPTool(BaseTool):
    """Use BaseTool's real ToolMessage formatting without an MCP process."""

    name: str
    description: str = "An isolated test tool."
    response_format: Literal["content_and_artifact"] = "content_and_artifact"
    handle_tool_error: bool = True
    payload: Any = None
    factory: Callable[..., Any] | None = None
    wrapped: bool = True
    structured: bool = True
    fail: bool = False
    calls: list[dict[str, Any]] = Field(default_factory=list)

    def _run(self, **arguments: Any) -> tuple[str, dict]:
        self.calls.append(arguments)
        if self.fail:
            raise ToolException("The fake MCP tool failed.")
        payload = self.factory(**arguments) if self.factory else self.payload
        artifact = {}
        if self.structured:
            artifact["structured_content"] = {"result": payload} if self.wrapped else payload
        return "Display content must not replace structured results.", artifact

    async def _arun(self, **arguments: Any) -> tuple[str, dict]:
        return self._run(**arguments)


class FakeMCPClient:
    def __init__(self, tools: list[BaseTool]) -> None:
        self.tools = tools
        self.discovery_calls = 0

    async def get_tools(self) -> list[BaseTool]:
        self.discovery_calls += 1
        return self.tools


class FakeResponseService:
    def __init__(self, analysis: TicketAnalysis, *, fail_at: str | None = None) -> None:
        self.analysis = analysis
        self.fail_at = fail_at
        self.extraction_calls: list[tuple[TicketInput, ToolContext, str]] = []
        self.reply_calls: list[tuple[TicketInput, TicketAnalysis, TriageDecision, ToolContext, str]] = []
        self.handoff_reply_calls: list[tuple] = []

    async def extract_information(
        self, ticket: TicketInput, context: ToolContext, *, handbook: str
    ) -> TicketAnalysis:
        self.extraction_calls.append((ticket.model_copy(deep=True), context.model_copy(deep=True), handbook))
        if self.fail_at == "extract":
            raise RuntimeError("Fake extraction failure with private detail.")
        return self.analysis.model_copy(deep=True)

    async def draft_reply(
        self, ticket: TicketInput, analysis: TicketAnalysis, decision: TriageDecision,
        context: ToolContext, *, handbook: str,
    ) -> str:
        self.reply_calls.append((ticket, analysis, decision, context, handbook))
        if self.fail_at == "reply":
            raise RuntimeError("Fake reply failure with private detail.")
        return "A fake draft for workflow verification."

    async def draft_handoff_reply(
        self, ticket, analysis, decision, context, receipt, *, handbook: str,
    ) -> str:
        self.handoff_reply_calls.append((ticket, analysis, decision, context, receipt, handbook))
        if self.fail_at == "handoff_reply":
            raise RuntimeError("Fake acknowledgment failure with private detail.")
        return graph_module.fallback_handoff_reply(analysis.language, decision, receipt)


class FakeDecisionService:
    def __init__(self, decision: TriageDecision, *, fail: bool = False) -> None:
        self.decision = decision
        self.fail = fail
        self.calls: list[tuple[TicketInput, TicketAnalysis, ToolContext, str]] = []

    async def decide(
        self, ticket: TicketInput, analysis: TicketAnalysis, context: ToolContext,
        *, handbook: str,
    ) -> TriageDecision:
        self.calls.append((ticket.model_copy(deep=True), analysis.model_copy(deep=True), context.model_copy(deep=True), handbook))
        if self.fail:
            raise RuntimeError("Fake decision failure with private detail.")
        return self.decision.model_copy(deep=True)


def make_analysis(*issues: ExtractedIssue, needs_status: bool = False) -> TicketAnalysis:
    return TicketAnalysis(
        product=None, issue_type="test_issue", customer_sentiment="neutral",
        issues=list(issues) or [ExtractedIssue(
            issue_type="test_issue", description="An unresolved test issue",
            retrieval_query="general support investigation",
        )],
        needs_system_status=needs_status,
    )


def make_decision(action: NextAction = NextAction.AUTO_RESPOND) -> TriageDecision:
    return TriageDecision(
        urgency=Urgency.LOW, action=action, owner=SpecialistTeam.GENERAL_SUPPORT,
        reason="Fake service choice for testing the graph edge.",
    )


def make_agent(
    store: SQLiteStore, analysis: TicketAnalysis | None = None,
    *, action: NextAction = NextAction.AUTO_RESPOND, fail_at: str | None = None,
    customer_payload: Any = None, knowledge_tool: FakeMCPTool | None = None,
    status_tool: FakeMCPTool | None = None,
) -> tuple[TicketTriageAgent, FakeResponseService, FakeDecisionService, FakeMCPClient]:
    response = FakeResponseService(analysis or make_analysis(), fail_at=fail_at)
    decision = FakeDecisionService(make_decision(action), fail=fail_at == "decision")
    customer = FakeMCPTool(name="check_customer_history", payload=customer_payload)
    article = KnowledgeBaseArticle(
        article_id="internal-test-guide", title="Test investigation procedure",
        content="Fictional investigation guidance.", audience="internal", is_mock=True,
    )
    knowledge = knowledge_tool or FakeMCPTool(
        name="lookup_knowledge_base", payload=[article.model_dump(mode="json")]
    )
    tools = [customer, knowledge]
    if status_tool is not None:
        tools.append(status_tool)
    client = FakeMCPClient(tools)
    agent = TicketTriageAgent(decision, response, mcp_client=client, store=store)
    return agent, response, decision, client


def latest_update(ticket_id: str = "ticket", customer_id: str = "customer") -> TicketUpdate:
    return TicketUpdate(
        ticket_id=ticket_id, customer_id=customer_id,
        message=TicketMessage(text="A new question", message_id="incoming-id"),
    )


def test_latest_sample_message_loads_prior_order_calls_both_tools_and_persists(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "support.sqlite3")
    store.seed_samples(FIXTURES / "customer_accounts.json", FIXTURES / "sample_tickets.json")
    samples = json.loads((FIXTURES / "sample_tickets.json").read_text(encoding="utf-8"))
    sample = TicketInput.model_validate(samples[0])
    latest = sample.messages[-1].model_copy(update={
        "message_id": stable_message_id(sample.ticket_id, len(sample.messages), sample.messages[-1])
    })
    customer = store.get_customer(sample.customer_id)
    assert customer is not None
    agent, response, decision, client = make_agent(store, customer_payload=customer.model_dump(mode="json"))
    update = TicketUpdate(ticket_id=sample.ticket_id, customer_id=sample.customer_id, message=latest)
    result = asyncio.run(agent.run(update))

    assert [item.text for item in response.extraction_calls[0][0].messages] == [item.text for item in sample.messages]
    assert response.extraction_calls[0][2] == HANDBOOK
    assert decision.calls[0][3] == HANDBOOK
    assert response.reply_calls[0][4] == HANDBOOK
    assert client.discovery_calls == 1
    assert client.tools[0].calls == [{"customer_id": sample.customer_id}]
    assert client.tools[1].calls == [{"query": "general support investigation"}]
    assert decision.calls[0][2].customer == customer
    assert len(decision.calls[0][2].articles) == 1
    assert result.reply == "A fake draft for workflow verification."
    assert result.handoff is None
    assert store.get_memory(sample.ticket_id) == result.analysis
    assert len(store.load_history(sample.ticket_id).messages) == 4
    with sqlite3.connect(store.path) as connection:
        runs = connection.execute("SELECT result_json FROM triage_runs").fetchall()
    assert len(runs) == 1
    assert TicketResult.model_validate_json(runs[0][0]) == result

    asyncio.run(agent.run(update))
    assert len(store.load_history(sample.ticket_id).messages) == 4
    assert client.discovery_calls == 1


def test_per_issue_normalized_queries_preserve_old_issue_and_skip_resolved(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "support.sqlite3")
    store.ingest_ticket(TicketInput(
        ticket_id="ticket", customer_id="customer", messages=[
            TicketMessage(text="ปัญหาเดิมยังไม่ได้รับการแก้ไข", message_id="old-issue"),
            TicketMessage(text="ขั้นตอนหนึ่งใช้งานได้แล้ว", message_id="resolved-step"),
        ],
    ))
    analysis = make_analysis(
        ExtractedIssue(issue_type="old", description="Earlier problem remains open", retrieval_query="  unresolved account access  "),
        ExtractedIssue(issue_type="new", description="Additional capability question", retrieval_query="capability documentation"),
        ExtractedIssue(issue_type="related", description="Related unresolved detail", retrieval_query="unresolved account access"),
        ExtractedIssue(issue_type="done", description="Confirmed resolved step", retrieval_query="do not search resolved issue", resolved=True),
    )
    agent, response, decision, client = make_agent(store, analysis, action=NextAction.ROUTE_TO_SPECIALIST)
    update = latest_update()
    update.message.text = "มีคำถามเรื่องความสามารถเพิ่มเติม"
    result = asyncio.run(agent.run(update))

    assert [item.text for item in response.extraction_calls[0][0].messages] == [
        "ปัญหาเดิมยังไม่ได้รับการแก้ไข", "ขั้นตอนหนึ่งใช้งานได้แล้ว", "มีคำถามเรื่องความสามารถเพิ่มเติม"
    ]
    assert {item["query"] for item in client.tools[1].calls} == {
        "unresolved account access", "capability documentation"
    }
    assert len(client.tools[1].calls) == 2
    assert [item.issue_type for item in result.analysis.issues] == ["old", "new", "related", "done"]
    assert [item.issue_type for item in store.get_memory("ticket").issues] == ["old", "new", "related", "done"]
    assert len(decision.calls[0][2].articles) == 1
    assert result.reply
    assert not response.reply_calls
    assert len(response.handoff_reply_calls) == 1
    assert result.handoff["assignment_performed"] is False
    assert len(result.handoff["unresolved_issues"]) == 3
    assert len(result.handoff["observations_and_attempts"]) == 3


def test_stale_summary_does_not_replace_authoritative_full_history(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "support.sqlite3")
    store.append_message("ticket", "customer", TicketMessage(text="Still blocked", message_id="earlier"))
    stale = make_analysis(ExtractedIssue(issue_type="old", description="Incorrect summary says resolved", resolved=True))
    store.save_result(TicketResult(ticket_id="ticket", analysis=stale, decision=make_decision()))
    current = make_analysis(ExtractedIssue(issue_type="old", description="Still unresolved", retrieval_query="account investigation"))
    agent, response, decision, _ = make_agent(store, current)
    result = asyncio.run(agent.run(latest_update()))

    forwarded_ticket, context, _ = response.extraction_calls[0]
    assert [item.text for item in forwarded_ticket.messages] == ["Still blocked", "A new question"]
    assert context.previous_analysis == stale
    assert result.analysis == current
    assert not decision.calls[0][1].issues[0].resolved
    assert store.get_memory("ticket") == current


@pytest.mark.parametrize("failure", ["extract", "decision", "reply"])
def test_provider_failure_persists_incoming_and_emits_human_review(tmp_path: Path, failure: str) -> None:
    store = SQLiteStore(tmp_path / "support.sqlite3")
    agent, response, decision, client = make_agent(store, fail_at=failure)
    result = asyncio.run(agent.run(latest_update()))

    assert result.decision.action == NextAction.ESCALATE_TO_HUMAN
    assert result.reply
    assert result.errors
    assert all("private detail" not in error for error in result.errors)
    assert result.handoff["assignment_performed"] is False
    assert len(store.load_history("ticket").messages) == 1
    assert store.load_history("ticket").messages[0].message_id == "incoming-id"
    assert store.get_memory("ticket") == result.analysis
    if failure == "extract":
        assert result.decision.urgency is None
        assert not decision.calls
        assert not response.reply_calls
        assert not response.handoff_reply_calls
        assert not client.tools[1].calls
    elif failure == "decision":
        assert result.decision.urgency is None
        assert not response.reply_calls
        assert len(response.handoff_reply_calls) == 1
    else:
        assert len(response.reply_calls) == 1
        assert result.decision.urgency == Urgency.LOW
        assert not response.handoff_reply_calls
    assert result.handoff["mock_dispatch_performed"] is True


@pytest.mark.parametrize("failure", ["invalid_summary", "read_error"])
def test_optional_summary_failure_uses_full_history_without_blocking_reply(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str,
) -> None:
    store = SQLiteStore(tmp_path / "support.sqlite3")
    store.append_message("ticket", "customer", TicketMessage(text="Earlier context", message_id="earlier"))
    if failure == "invalid_summary":
        with sqlite3.connect(store.path) as connection:
            connection.execute(
                "INSERT INTO ticket_memory(ticket_id, analysis_json, updated_at) VALUES (?, ?, ?)",
                ("ticket", '{"private-summary-detail":', 0),
            )
    else:
        def failed_memory(_):
            raise RuntimeError("private-summary-detail")
        monkeypatch.setattr(store, "get_memory", failed_memory)

    agent, response, decision, _ = make_agent(store, customer_payload={
        "customer_id": "customer", "plan": "Pro", "tenure_months": 1,
    })
    result = asyncio.run(agent.run(latest_update()))

    assert result.decision.action == NextAction.AUTO_RESPOND
    assert result.reply is not None
    assert len(response.extraction_calls) == len(decision.calls) == len(response.reply_calls) == 1
    ticket, context, _ = response.extraction_calls[0]
    assert [message.text for message in ticket.messages] == ["Earlier context", "A new question"]
    assert context.previous_analysis is None
    assert context.tool_errors == []
    assert len(result.errors) == 1
    assert "Previous analysis unavailable" in result.errors[0]
    assert "private-summary-detail" not in result.errors[0]


def test_reply_failure_preserves_valid_classifications_for_the_handoff(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "support.sqlite3")
    analysis = make_analysis(
        ExtractedIssue(issue_type="billing", description="First issue", retrieval_query="billing"),
        ExtractedIssue(issue_type="access", description="Second issue", retrieval_query="access"),
    )
    agent, _, decision, _ = make_agent(store, analysis, fail_at="reply")
    classified = TriageDecision(
        urgency=Urgency.HIGH, action=NextAction.AUTO_RESPOND, owner=SpecialistTeam.BILLING,
        confidence=0.97, reason="Existing classification supported by the conversation.",
        issue_decisions=[
            IssueDecision(issue_index=0, urgency=Urgency.HIGH, action=NextAction.AUTO_RESPOND,
                          owner=SpecialistTeam.BILLING),
            IssueDecision(issue_index=1, urgency=Urgency.MEDIUM, action=NextAction.AUTO_RESPOND,
                          owner=SpecialistTeam.TECHNICAL_SUPPORT),
        ],
    )
    decision.decision = classified
    result = asyncio.run(agent.run(latest_update()))

    assert result.decision.action == NextAction.ESCALATE_TO_HUMAN
    assert result.decision.urgency == classified.urgency
    assert result.decision.owner == classified.owner
    assert result.decision.confidence == classified.confidence
    assert result.decision.issue_decisions == classified.issue_decisions
    assert classified.reason in result.decision.reason
    assert "Reply generation failed" in result.decision.reason
    assert result.handoff["recommended_owner"] == "billing"
    assert result.reply
    assert result.handoff["mock_dispatch"]["follow_up_requested"] is True
    assert classified.action == NextAction.AUTO_RESPOND
    with sqlite3.connect(store.path) as connection:
        persisted = json.loads(connection.execute("SELECT result_json FROM triage_runs").fetchone()[0])
    assert persisted["decision"] == result.decision.model_dump(mode="json")


@pytest.mark.parametrize("action", [NextAction.ROUTE_TO_SPECIALIST, NextAction.ESCALATE_TO_HUMAN])
def test_handoff_dispatch_precedes_acknowledgment_and_retries_reuse_receipt(tmp_path: Path, action) -> None:
    store = SQLiteStore(tmp_path / "support.sqlite3")
    agent, response, decision, _ = make_agent(store, action=action)
    decision.decision.owner = SpecialistTeam.BILLING
    incoming = latest_update()
    first = asyncio.run(agent.run(incoming))

    assert first.decision.action == action
    assert first.reply
    assert not response.reply_calls
    assert len(response.handoff_reply_calls) == 1
    receipt = response.handoff_reply_calls[0][4]
    assert receipt is not None and receipt.status == "accepted"
    assert receipt.owner == SpecialistTeam.BILLING and receipt.action == action
    assert receipt.is_mock is True
    assert receipt.follow_up_requested is (action == NextAction.ESCALATE_TO_HUMAN)
    assert "mock" not in first.reply.lower()
    assert "simulat" not in first.reply.lower()
    assert "no real staff" not in first.reply.lower()
    assert ("will contact you" in first.reply) is receipt.follow_up_requested
    assert first.handoff["mock_dispatch"] == receipt.model_dump(mode="json")
    assert first.handoff["mock_dispatch_performed"] is True
    assert first.handoff["assignment_performed"] is False
    second = asyncio.run(agent.run(incoming))
    assert second.handoff["mock_dispatch"]["dispatch_id"] == receipt.dispatch_id
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM mock_handoffs").fetchone()[0] == 1
    assert len(store.load_history("ticket").messages) == 1


@pytest.mark.parametrize("failure", ["dispatch_error", "mismatched_owner", "non_mock_receipt"])
def test_failed_or_invalid_dispatch_cannot_be_confirmed_to_customer(tmp_path: Path, failure: str) -> None:
    store = SQLiteStore(tmp_path / "support.sqlite3")
    agent, response, decision, _ = make_agent(store, action=NextAction.ROUTE_TO_SPECIALIST)
    decision.decision.owner = SpecialistTeam.BILLING
    calls = []

    class FailedDispatch:
        def dispatch(self, ticket, chosen, payload):
            calls.append((ticket, chosen, payload))
            if failure == "dispatch_error":
                raise RuntimeError("private-dispatch-detail")
            return {
                "dispatch_id": "invalid-receipt", "action": chosen.action,
                "owner": SpecialistTeam.PRODUCT if failure == "mismatched_owner" else chosen.owner,
                "destination": "mock-specialist:billing", "status": "accepted",
                "is_mock": failure != "non_mock_receipt", "follow_up_requested": False,
                "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
            }

    agent.handoff_service = FailedDispatch()
    result = asyncio.run(agent.run(latest_update()))
    assert len(calls) == 1
    assert result.decision.action == NextAction.ESCALATE_TO_HUMAN
    assert result.decision.owner == SpecialistTeam.BILLING
    assert result.reply
    assert "dispatch and staff notification have not been confirmed" in result.reply
    assert "has been routed" not in result.reply
    assert "has been escalated" not in result.reply
    assert "will contact" not in result.reply
    assert "mock" not in result.reply.lower()
    assert "simulat" not in result.reply.lower()
    assert result.handoff["mock_dispatch_performed"] is False
    assert result.handoff["mock_dispatch"] is None
    assert response.handoff_reply_calls[0][4] is None
    assert any("Mock handoff dispatch failed" in error for error in result.errors)
    assert all("private-dispatch-detail" not in error for error in result.errors)


@pytest.mark.parametrize("action", [NextAction.ROUTE_TO_SPECIALIST, NextAction.ESCALATE_TO_HUMAN])
def test_handoff_reply_failure_uses_receipt_grounded_fallback_without_changing_action(tmp_path: Path, action) -> None:
    store = SQLiteStore(tmp_path / "support.sqlite3")
    agent, response, _, _ = make_agent(store, action=action, fail_at="handoff_reply")
    result = asyncio.run(agent.run(latest_update()))
    assert result.decision.action == action
    assert result.reply
    assert "mock" not in result.reply.lower()
    assert "simulat" not in result.reply.lower()
    expected_status = "routed" if action == NextAction.ROUTE_TO_SPECIALIST else "escalated"
    assert expected_status in result.reply
    assert len(response.handoff_reply_calls) == 1
    assert result.handoff["mock_dispatch"]["status"] == "accepted"
    assert any("Handoff acknowledgment used fallback" in error for error in result.errors)
    assert all("private detail" not in error for error in result.errors)


def test_failed_thai_extraction_still_returns_localized_staff_acknowledgment(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "support.sqlite3")
    agent, response, _, _ = make_agent(store, fail_at="extract")
    incoming = latest_update()
    incoming.message.text = "ระบบยังใช้งานไม่ได้ครับ"
    result = asyncio.run(agent.run(incoming))
    assert result.analysis.language == "th"
    assert result.reply and any("\u0e00" <= char <= "\u0e7f" for char in result.reply)
    assert "จำลอง" not in result.reply
    assert "ไม่มีการติดต่อเจ้าหน้าที่จริง" not in result.reply
    assert "จะติดต่อ" in result.reply
    assert not response.handoff_reply_calls
    assert result.handoff["mock_dispatch"]["follow_up_requested"] is True


def test_later_follow_up_passes_original_observation_and_current_time_to_services(tmp_path: Path) -> None:
    observed = datetime(2026, 1, 1, 9, tzinfo=timezone.utc)
    now = [observed.timestamp()]
    store = SQLiteStore(tmp_path / "support.sqlite3", clock=lambda: now[0])
    store.append_message("ticket", "customer", TicketMessage(
        text="I have a presentation in two hours.", time_label="just now", message_id="old-deadline",
    ))
    now[0] += 2 * 24 * 60 * 60
    agent, response, decision, _ = make_agent(store)
    asyncio.run(agent.run(latest_update()))

    for ticket in (response.extraction_calls[0][0], decision.calls[0][0], response.reply_calls[0][0]):
        assert ticket.messages[0].observed_at == observed
        assert ticket.messages[0].time_label == "just now"
        assert ticket.as_of == datetime.fromtimestamp(now[0], timezone.utc)
        assert ticket.messages[1].observed_at == ticket.as_of


@pytest.mark.parametrize("wrapped,payload", [(True, None), (False, None), (True, {"example": 1}), (False, [{"example": 1}])])
def test_mcp_structured_wrapped_direct_and_null_results(tmp_path: Path, wrapped: bool, payload: Any) -> None:
    store = SQLiteStore(tmp_path / "support.sqlite3")
    agent, _, _, _ = make_agent(store)
    tool = FakeMCPTool(name="shape_test", payload=payload, wrapped=wrapped)
    agent.tools = {tool.name: tool}
    assert asyncio.run(agent.call_tool(tool.name, {"query": "example"})) == payload
    assert tool.calls == [{"query": "example"}]


@pytest.mark.parametrize("failure,expected", [("error", RuntimeError), ("missing_structured", ValueError)])
def test_mcp_error_and_missing_structured_results_are_not_parsed_as_success(tmp_path: Path, failure: str, expected: type[Exception]) -> None:
    store = SQLiteStore(tmp_path / "support.sqlite3")
    agent, _, _, _ = make_agent(store)
    tool = FakeMCPTool(name="shape_test", fail=failure == "error", structured=failure != "missing_structured")
    agent.tools = {tool.name: tool}
    with pytest.raises(expected, match="MCP tool"):
        asyncio.run(agent.call_tool(tool.name, {}))


@pytest.mark.parametrize("shape", ["null", "error", "malformed"])
def test_bad_knowledge_results_are_forwarded_as_tool_errors(tmp_path: Path, shape: str) -> None:
    store = SQLiteStore(tmp_path / "support.sqlite3")
    knowledge = FakeMCPTool(
        name="lookup_knowledge_base", payload=None if shape == "null" else {"wrong": "shape"},
        fail=shape == "error",
    )
    agent, _, decision, _ = make_agent(store, action=NextAction.ROUTE_TO_SPECIALIST, knowledge_tool=knowledge)
    result = asyncio.run(agent.run(latest_update()))
    assert any("Knowledge lookup failed" in item for item in result.errors)
    assert not decision.calls[0][2].articles
    assert decision.calls[0][2].tool_errors
    assert result.reply


def test_ownership_mismatch_rejects_before_tool_or_provider_calls(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "support.sqlite3")
    store.append_message("ticket", "owner", TicketMessage(text="Original customer message"))
    agent, response, decision, client = make_agent(store)
    with pytest.raises(ValueError, match="different customer"):
        asyncio.run(agent.run(latest_update(customer_id="other-customer")))
    assert not response.extraction_calls
    assert not response.reply_calls
    assert not decision.calls
    assert client.discovery_calls == 0
    assert len(store.load_history("ticket", "owner").messages) == 1


def test_status_tool_preserves_mock_marker_and_uses_expiring_cache(tmp_path: Path) -> None:
    now = [1000.0]
    store = SQLiteStore(tmp_path / "support.sqlite3", clock=lambda: now[0])
    fixture = tmp_path / "status.json"
    fixture.write_text(json.dumps({"default": {"status": "unknown", "details": "Test source"}}), encoding="utf-8")
    support_tools = SupportTools(store=store, status_path=fixture)
    status_tool = FakeMCPTool(name="check_system_status", factory=support_tools.check_system_status)
    agent, _, decision, _ = make_agent(store, make_analysis(needs_status=True), status_tool=status_tool)
    update = latest_update()
    asyncio.run(agent.run(update))
    first_status = decision.calls[0][2].system_status
    assert first_status["is_mock"] is True
    assert first_status["cached"] is False
    fixture.write_text(json.dumps({"default": {"status": "different"}}), encoding="utf-8")
    asyncio.run(agent.run(update))
    assert decision.calls[1][2].system_status["cached"] is True
    assert decision.calls[1][2].system_status["status"] == "unknown"
    assert decision.calls[1][2].system_status["observed_at"] == first_status["observed_at"]
    now[0] += 300
    asyncio.run(agent.run(update))
    assert decision.calls[2][2].system_status["cached"] is False
    assert decision.calls[2][2].system_status["status"] == "different"
    assert status_tool.calls == [{"region": None}] * 3
    assert len(store.load_history("ticket").messages) == 1
