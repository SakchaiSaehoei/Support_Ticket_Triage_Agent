"""Verify native graph trace addressing and flushing without a network client."""

import asyncio
import logging
from types import SimpleNamespace
from typing import TypedDict

import pytest
from langchain_core.tracers.langchain import LangChainTracer
from langgraph.graph import END, START, StateGraph
from langsmith import get_tracing_context, utils
from langsmith import run_trees
from openai.types.decision import Decision

import support_ticket_triage.observability as observability


class FakeClient:
    def __init__(self) -> None:
        self.created: list[dict] = []
        self.updated: list[dict] = []
        self.flush_timeouts: list[float | None] = []

    def create_run(self, **kwargs) -> None:
        self.created.append(kwargs)

    def update_run(self, **kwargs) -> None:
        self.updated.append(kwargs)

    def flush(self, timeout: float | None = None) -> None:
        self.flush_timeouts.append(timeout)


class CounterState(TypedDict):
    value: int


def native_graph():
    graph = StateGraph(CounterState)
    graph.add_node("increment", lambda state: {"value": state["value"] + 1})
    graph.add_edge(START, "increment")
    graph.add_edge("increment", END)
    return graph.compile()


@pytest.fixture(autouse=True)
def isolated_trace_environment(monkeypatch):
    for name in (
        "LANGSMITH_TRACING", "LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING",
        "LANGCHAIN_TRACING_V2", "LANGSMITH_PROJECT", "LANGCHAIN_PROJECT",
        "LANGCHAIN_SESSION", "LANGSMITH_SESSION", "HOSTED_LANGSERVE_PROJECT_NAME",
        "LANGSMITH_AGENT_ID", "LANGSMITH_AGENT_ENVIRONMENT",
    ):
        monkeypatch.delenv(name, raising=False)
    utils.get_env_var.cache_clear()
    utils.get_tracer_project.cache_clear()
    yield
    utils.get_env_var.cache_clear()
    utils.get_tracer_project.cache_clear()


def test_disabled_tracing_does_not_create_a_client(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")

    def unexpected_client():
        pytest.fail("Disabled tracing must not create a LangSmith client.")

    monkeypatch.setattr(observability, "get_cached_client", unexpected_client)
    assert observability.tracing_config("ticket-disabled") == {}
    observability.flush_traces()


def test_native_graph_preserves_agent_address_without_project(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_AGENT_ID", "offline-test-agent")
    monkeypatch.setenv("LANGSMITH_AGENT_ENVIRONMENT", "local")
    client = FakeClient()
    monkeypatch.setattr(observability, "get_cached_client", lambda: client)

    config = observability.tracing_config("ticket-agent-mode")
    assert len(config["callbacks"]) == 1
    assert isinstance(config["callbacks"][0], LangChainTracer)
    result = asyncio.run(native_graph().ainvoke({"value": 1}, config=config))

    assert result == {"value": 2}
    assert {run["name"] for run in client.created} == {"SupportTicketTriage", "increment"}
    assert len(client.created) == 2
    for run in client.created:
        assert run.get("session_name") is None
        assert run.get("session_id") is None
        assert run["address"].id == "offline-test-agent"
        assert run["address"].environment == "local"
        assert run["extra"]["metadata"]["ticket_id"] == "ticket-agent-mode"
    assert client.updated


def test_native_graph_preserves_configured_project(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_PROJECT", "offline-test-project")
    client = FakeClient()
    monkeypatch.setattr(observability, "get_cached_client", lambda: client)

    config = observability.tracing_config("ticket-project-mode")
    asyncio.run(native_graph().ainvoke({"value": 1}, config=config))

    assert len(client.created) == 2
    assert all(run.get("session_name") == "offline-test-project" for run in client.created)
    assert all(run.get("address") is None for run in client.created)


def test_copied_agent_callback_does_not_restore_default_project(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_AGENT_ID", "offline-test-agent")
    monkeypatch.setenv("LANGSMITH_AGENT_ENVIRONMENT", "local")
    client = FakeClient()
    monkeypatch.setattr(observability, "get_cached_client", lambda: client)

    callback = observability.tracing_config("ticket-copy")["callbacks"][0]
    copied = callback.copy_with_metadata_defaults(metadata={"environment": "test"})
    assert copied.project_name is None
    assert copied.client is client


def test_enabled_flush_uses_bounded_wait_on_cached_client(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    client = FakeClient()
    monkeypatch.setattr(observability, "get_cached_client", lambda: client)

    observability.flush_traces()

    assert client.flush_timeouts == [10]


@pytest.mark.parametrize("failure", [TimeoutError, RuntimeError])
def test_flush_failure_is_nonfatal_and_does_not_log_error_payload(
    monkeypatch, caplog, failure
):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")

    class FailingClient(FakeClient):
        def flush(self, timeout=None):
            assert timeout == 10
            raise failure("synthetic-sensitive-detail-do-not-log")

    monkeypatch.setattr(observability, "get_cached_client", lambda: FailingClient())
    with caplog.at_level(logging.WARNING):
        observability.flush_traces()

    assert failure.__name__ in caplog.text
    assert "synthetic-sensitive-detail-do-not-log" not in caplog.text


@pytest.mark.parametrize("failure", ["partial_address", "client_startup", "tracer_startup"])
def test_failed_trace_setup_disables_only_one_native_graph_invocation(
    monkeypatch, caplog, failure
):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_AGENT_ID", "offline-startup-test-agent")
    if failure != "partial_address":
        monkeypatch.setenv("LANGSMITH_AGENT_ENVIRONMENT", "local")
    parent_enabled = get_tracing_context()["enabled"]
    constructor_calls = []
    original_init = LangChainTracer.__init__

    def unavailable_client():
        if failure == "client_startup":
            raise RuntimeError("synthetic-sensitive-startup-detail-do-not-log")
        if failure == "tracer_startup":
            return FakeClient()
        pytest.fail("An incomplete address must fail before constructing a client.")

    def unavailable_tracer(self, *args, **kwargs):
        constructor_calls.append("tracer")
        if failure == "tracer_startup" and len(constructor_calls) == 1:
            raise RuntimeError("synthetic-sensitive-startup-detail-do-not-log")
        pytest.fail("A failed setup must not initialize an automatic graph tracer.")

    def unexpected_sdk_client(*args, **kwargs):
        pytest.fail("The native graph must not create a default LangSmith client.")

    monkeypatch.setattr(observability, "get_cached_client", unavailable_client)
    monkeypatch.setattr(run_trees, "get_cached_client", unexpected_sdk_client)
    monkeypatch.setattr(LangChainTracer, "__init__", unavailable_tracer)
    response = Decision.model_validate({
        "model": "offline-model", "answers": [],
        "usage": {
            "input_tokens": 1, "output_tokens": 1, "total_tokens": 2,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    })
    provider_calls = []
    enabled_during_node = []

    async def create(**request):
        provider_calls.append(request)
        return response

    sdk = SimpleNamespace(decisions=SimpleNamespace(create=create))

    async def increment(state):
        enabled_during_node.append(utils.tracing_is_enabled())
        result = await observability.request_decision(
            sdk, model="offline-model", input="Synthetic public evidence", questions=[],
        )
        assert result is response
        return {"value": state["value"] + 1}

    builder = StateGraph(CounterState)
    builder.add_node("increment", increment)
    builder.add_edge(START, "increment")
    builder.add_edge("increment", END)
    graph = builder.compile()

    async def invoke():
        with observability.graph_tracing("offline-startup-ticket") as config:
            return await graph.ainvoke({"value": 1}, config=config)

    with caplog.at_level(logging.WARNING):
        assert asyncio.run(invoke()) == {"value": 2}

    assert len(provider_calls) == 1
    assert enabled_during_node == [False]
    assert constructor_calls == (["tracer"] if failure == "tracer_startup" else [])
    assert get_tracing_context()["enabled"] is parent_enabled
    assert utils.tracing_is_enabled() is True
    assert "synthetic-sensitive-startup-detail-do-not-log" not in caplog.text
    assert "LangSmith graph trace did not start" in caplog.text

    # Repair configuration without restarting or globally disabling telemetry.
    monkeypatch.setenv("LANGSMITH_AGENT_ENVIRONMENT", "local")
    utils.get_env_var.cache_clear()
    utils.get_tracer_project.cache_clear()
    traces = FakeClient()
    monkeypatch.setattr(observability, "get_cached_client", lambda: traces)
    monkeypatch.setattr(LangChainTracer, "__init__", original_init)

    assert asyncio.run(invoke()) == {"value": 2}
    assert len(provider_calls) == 2
    assert enabled_during_node == [False, True]
    assert {run["name"] for run in traces.created} == {
        "SupportTicketTriage", "increment", "OpenAI Decisions",
    }
    assert all(run.get("session_name") is None for run in traces.created)
    assert get_tracing_context()["enabled"] is parent_enabled


def test_graph_body_error_is_not_treated_as_trace_startup_failure(monkeypatch, caplog):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    invocations = []

    with pytest.raises(RuntimeError, match="provider failure"):
        with observability.graph_tracing("offline-body-failure"):
            invocations.append("body")
            raise RuntimeError("provider failure")

    assert invocations == ["body"]
    assert "LangSmith graph trace did not start" not in caplog.text
