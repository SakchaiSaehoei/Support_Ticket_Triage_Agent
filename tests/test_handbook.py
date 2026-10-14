"""Keep mandatory policy independent of embeddings, tool discovery, and models."""

import asyncio
from pathlib import Path

import pytest

import support_ticket_triage.graph as graph_module
from support_ticket_triage.graph import TicketTriageAgent
from support_ticket_triage.retrieval import load_handbook
from support_ticket_triage.schemas import (
    ExtractedIssue,
    NextAction,
    TicketAnalysis,
    TicketInput,
    TicketMessage,
    ToolContext,
    TriageDecision,
    Urgency,
)


@pytest.fixture(autouse=True)
def disable_external_tracing(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")


@pytest.fixture
def ticket() -> TicketInput:
    return TicketInput(
        ticket_id="offline-policy-test",
        customer_id="test-customer",
        messages=[TicketMessage(text="Please help with theme settings.", time_label="now")],
    )


class FakeResponse:
    def __init__(self, calls: list[tuple[str, str]]) -> None:
        self.calls = calls

    async def extract_information(self, ticket, context, *, handbook: str):
        self.calls.append(("extract", handbook))
        return TicketAnalysis(
            product=None,
            issue_type="appearance",
            customer_sentiment="neutral",
            issues=[ExtractedIssue(issue_type="appearance", description="Theme settings")],
        )

    async def draft_reply(self, ticket, analysis, decision, context, *, handbook: str):
        self.calls.append(("reply", handbook))
        return "Offline test reply"


class FakeDecision:
    def __init__(self, calls: list[tuple[str, str]]) -> None:
        self.calls = calls

    async def decide(self, ticket, analysis, context, *, handbook: str):
        self.calls.append(("decide", handbook))
        return TriageDecision(urgency=Urgency.LOW, action=NextAction.AUTO_RESPOND)


class OfflineAgent(TicketTriageAgent):
    def __init__(self, calls: list[tuple[str, str]], events: list[str]) -> None:
        self.events = events
        # A sentinel avoids even constructing a client connected to an MCP server.
        super().__init__(FakeDecision(calls), FakeResponse(calls), mcp_client=object())

    async def load_tools(self) -> None:
        self.events.append("discover")
        self.tools = {"offline": object()}

    async def _gather_context(self, state):
        if not self.tools:
            await self.load_tools()
        self.events.append("gather")
        return {"context": ToolContext()}


def test_handbook_load_reads_complete_current_file(tmp_path: Path):
    path = tmp_path / "agent_handbook.md"
    original = "# Policy\n\n## Rules\n\nAlways check evidence."
    path.write_text(f"\n{original}\n", encoding="utf-8")
    assert load_handbook(tmp_path) == original

    updated = "# Policy\n\n## Rules\n\nEscalate unverified account changes."
    path.write_text(updated, encoding="utf-8")
    assert load_handbook(tmp_path) == updated


@pytest.mark.parametrize("contents", ["", " \n\t "])
def test_empty_handbook_is_an_error(tmp_path: Path, contents: str):
    (tmp_path / "agent_handbook.md").write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match="(?i)handbook.*empty"):
        load_handbook(tmp_path)


def test_missing_handbook_is_an_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_handbook(tmp_path)


@pytest.mark.parametrize("failure", [FileNotFoundError, ValueError])
def test_failed_policy_load_stops_before_tool_discovery(
    monkeypatch, ticket: TicketInput, failure
):
    calls: list[tuple[str, str]] = []
    events: list[str] = []

    def unavailable_handbook():
        events.append("handbook")
        raise failure("Mandatory policy is unavailable")

    monkeypatch.setattr(graph_module, "load_handbook", unavailable_handbook)
    agent = OfflineAgent(calls, events)
    with pytest.raises(failure):
        asyncio.run(agent.run(ticket))
    assert events == ["handbook"]
    assert calls == []


def test_policy_is_loaded_before_successful_tool_discovery(monkeypatch, ticket):
    calls: list[tuple[str, str]] = []
    events: list[str] = []

    def handbook():
        events.append("handbook")
        return "Mandatory test policy"

    monkeypatch.setattr(graph_module, "load_handbook", handbook)
    result = asyncio.run(OfflineAgent(calls, events).run(ticket))

    assert events[:2] == ["handbook", "discover"]
    assert result.reply == "Offline test reply"
    assert calls == [
        ("extract", "Mandatory test policy"),
        ("decide", "Mandatory test policy"),
        ("reply", "Mandatory test policy"),
    ]


def test_each_run_passes_fresh_policy_to_every_model_step(monkeypatch, ticket):
    calls: list[tuple[str, str]] = []
    events: list[str] = []
    versions = iter(["Policy version one", "Policy version two"])

    def handbook():
        events.append("handbook")
        return next(versions)

    monkeypatch.setattr(graph_module, "load_handbook", handbook)
    agent = OfflineAgent(calls, events)
    agent.tools = {"offline": object()}
    asyncio.run(agent.run(ticket))
    asyncio.run(agent.run(ticket))

    assert events == ["handbook", "gather", "handbook", "gather"]
    assert calls == [
        ("extract", "Policy version one"),
        ("decide", "Policy version one"),
        ("reply", "Policy version one"),
        ("extract", "Policy version two"),
        ("decide", "Policy version two"),
        ("reply", "Policy version two"),
    ]
