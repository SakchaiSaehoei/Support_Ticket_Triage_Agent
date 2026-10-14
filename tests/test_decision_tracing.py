"""Inspect Decisions LLM spans using native graph callbacks and offline clients."""

import asyncio
import json
import logging
from types import SimpleNamespace
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langsmith import get_tracing_context, utils
from langsmith.run_trees import RunTree
from openai import OpenAIError
from openai.types.decision import Decision
import pytest

import support_ticket_triage.observability as observability
from support_ticket_triage.decision_service import DecisionService
from support_ticket_triage.schemas import (
    ExtractedIssue, TicketAnalysis, TicketInput, TicketMessage, ToolContext,
)
from test_observability import FakeClient


SDK_SECRET = "synthetic-sdk-secret-must-not-be-traced"
PROVIDER_SECRET = "synthetic-provider-body-must-not-be-traced"
TELEMETRY_SECRET = "synthetic-telemetry-body-must-not-be-logged"
REQUEST_MODEL = "offline-requested-model"
ACTUAL_MODEL = "offline-resolved-model"
INPUT = "Public evidence for an isolated Decisions tracing test."
QUESTIONS = [{
    "type": "choice", "name": "action_0",
    "instructions": "Choose the next action for this isolated fixture.",
    "choices": [{"value": "route_to_specialist"}, {"value": "auto_respond"}],
}]


@pytest.fixture(autouse=True)
def isolated_trace_environment(monkeypatch: pytest.MonkeyPatch):
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


def sdk_decision() -> Decision:
    """Use the SDK's native schema, including the full choice distributions."""
    return Decision.model_validate({
        "model": ACTUAL_MODEL,
        "answers": [
            {
                "type": "choice", "name": "urgency_0", "choice": "low",
                "confidence": 0.61,
                "probabilities": [
                    {"value": "low", "probability": 0.61},
                    {"value": "high", "probability": 0.39},
                ],
            },
            {
                "type": "choice", "name": "action_0", "choice": "route_to_specialist",
                "confidence": 0.91,
                "probabilities": [
                    {"value": "route_to_specialist", "probability": 0.91},
                    {"value": "auto_respond", "probability": 0.09},
                ],
            },
            {
                "type": "choice", "name": "owner_0", "choice": "technical_support",
                "confidence": 0.72,
                "probabilities": [
                    {"value": "technical_support", "probability": 0.72},
                    {"value": "general_support", "probability": 0.28},
                ],
            },
        ],
        "usage": {
            "input_tokens": 120,
            "input_tokens_details": {"cached_tokens": 37, "cache_write_tokens": 0},
            "output_tokens": 18,
            "output_tokens_details": {"reasoning_tokens": 4},
            "total_tokens": 138,
        },
    })


class FakeEndpoint:
    def __init__(self, response: Decision, error: OpenAIError | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict] = []

    async def create(self, **payload: Any) -> Decision:
        self.calls.append(payload)
        if self.error is not None:
            raise self.error
        return self.response


class FakeSDKClient:
    def __init__(self, response: Decision, error: OpenAIError | None = None) -> None:
        self.api_key = SDK_SECRET
        self.decisions = FakeEndpoint(response, error)

    def __repr__(self) -> str:
        return f"FakeSDKClient(api_key={self.api_key!r})"


class DecisionState(TypedDict, total=False):
    input: str
    questions: list[dict]
    response: Decision
    failure: str
    triage: Any


def native_decide_graph(node):
    builder = StateGraph(DecisionState)
    builder.add_node("decide", node)
    builder.add_edge(START, "decide")
    builder.add_edge("decide", END)
    return builder.compile()


def enable_tracing(monkeypatch: pytest.MonkeyPatch, client: FakeClient, destination: str) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    if destination == "agent":
        monkeypatch.setenv("LANGSMITH_AGENT_ID", "offline-decisions-agent")
        monkeypatch.setenv("LANGSMITH_AGENT_ENVIRONMENT", "local")
    else:
        monkeypatch.setenv("LANGSMITH_PROJECT", "offline-decisions-project")
    monkeypatch.setattr(observability, "get_cached_client", lambda: client)


def completed_span(client: FakeClient, span: dict) -> dict:
    updates = [update for update in client.updated if str(update["run_id"]) == str(span["id"])]
    assert len(updates) == 1
    return updates[0]


def assert_no_client_or_secret_capture(client: FakeClient) -> None:
    serialized = json.dumps({"created": client.created, "updated": client.updated}, default=str)
    assert SDK_SECRET not in serialized
    assert PROVIDER_SECRET not in serialized
    assert "FakeSDKClient" not in serialized
    for span in client.created:
        assert "self" not in span.get("inputs", {})
        assert "client" not in span.get("inputs", {})
        assert "api_key" not in span.get("inputs", {})


@pytest.mark.parametrize("destination", ["agent", "project"])
def test_native_decisions_span_preserves_destination_hierarchy_and_raw_sdk_data(
    monkeypatch: pytest.MonkeyPatch, destination: str,
) -> None:
    traces = FakeClient()
    response = sdk_decision()
    sdk = FakeSDKClient(response)
    enable_tracing(monkeypatch, traces, destination)

    async def decide(state: DecisionState) -> DecisionState:
        result = await observability.request_decision(
            sdk, model=REQUEST_MODEL, input=state["input"], questions=state["questions"]
        )
        return {"response": result}

    output = asyncio.run(native_decide_graph(decide).ainvoke(
        {"input": INPUT, "questions": QUESTIONS},
        config=observability.tracing_config("offline-ticket"),
    ))

    assert output["response"] is response
    assert sdk.decisions.calls == [{"model": REQUEST_MODEL, "input": INPUT, "questions": QUESTIONS}]
    assert len(traces.created) == 3
    assert {span["name"] for span in traces.created} == {"SupportTicketTriage", "decide", "OpenAI Decisions"}
    llms = [span for span in traces.created if span["run_type"] == "llm"]
    assert len(llms) == 1
    llm = llms[0]
    parent = next(span for span in traces.created if span["name"] == "decide")
    root = next(span for span in traces.created if span["name"] == "SupportTicketTriage")
    assert str(llm["parent_run_id"]) == str(parent["id"])
    assert str(parent["parent_run_id"]) == str(root["id"])
    assert str(llm["trace_id"]) == str(root["id"])

    for span in traces.created:
        assert span.get("session_id") is None
        if destination == "agent":
            assert span.get("session_name") is None
            assert span["address"].id == "offline-decisions-agent"
            assert span["address"].environment == "local"
        else:
            assert span.get("address") is None
            assert span.get("session_name") == "offline-decisions-project"

    assert set(llm["inputs"]) == {"model", "input", "questions", "messages"}
    assert llm["inputs"]["model"] == REQUEST_MODEL
    assert llm["inputs"]["input"] == INPUT
    assert llm["inputs"]["questions"] == QUESTIONS
    input_message = llm["inputs"]["messages"][0]
    assert input_message["role"] == "user"
    assert json.loads(input_message["content"]) == {"input": INPUT, "questions": QUESTIONS}

    update = completed_span(traces, llm)
    raw = response.model_dump(mode="json")
    assert update["error"] is None
    assert {key: update["outputs"][key] for key in raw} == raw
    assert update["outputs"]["answers"] == raw["answers"]
    assert [answer["confidence"] for answer in update["outputs"]["answers"]] == [0.61, 0.91, 0.72]
    assert [answer["probabilities"] for answer in update["outputs"]["answers"]] == [answer["probabilities"] for answer in raw["answers"]]
    usage = {
        "input_tokens": 120, "output_tokens": 18, "total_tokens": 138,
        "input_token_details": {"cache_read": 37, "cache_creation": 0},
        "output_token_details": {"reasoning": 4},
    }
    assert update["outputs"]["usage_metadata"] == usage
    output_message = update["outputs"]["messages"][0]
    assert output_message["role"] == "assistant"
    assert json.loads(output_message["content"]) == {"answers": raw["answers"]}
    metadata = update["extra"]["metadata"]
    assert metadata["ls_provider"] == "openai"
    assert metadata["ls_model_name"] == ACTUAL_MODEL
    assert metadata["api_endpoint"] == "/v1/decisions"
    assert metadata["usage_metadata"] == usage
    assert_no_client_or_secret_capture(traces)


def test_disabled_tracing_passes_sdk_payload_and_identity_without_serialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    response = sdk_decision()
    sdk = FakeSDKClient(response)

    def unexpected(*args: Any, **kwargs: Any):
        pytest.fail("Disabled Decisions tracing must not construct a span, client, or serialized output.")

    monkeypatch.setattr(observability, "get_cached_client", unexpected)
    monkeypatch.setattr(observability, "trace", unexpected)
    monkeypatch.setattr(Decision, "model_dump", unexpected)
    result = asyncio.run(observability.request_decision(
        sdk, model=REQUEST_MODEL, input=INPUT, questions=QUESTIONS,
    ))
    assert result is response
    assert sdk.decisions.calls == [{"model": REQUEST_MODEL, "input": INPUT, "questions": QUESTIONS}]
    assert sdk.decisions.calls[0]["questions"] is QUESTIONS


def test_native_graph_failure_span_is_sanitized_and_same_sdk_error_is_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    traces = FakeClient()
    error = OpenAIError(PROVIDER_SECRET)
    sdk = FakeSDKClient(sdk_decision(), error)
    enable_tracing(monkeypatch, traces, "agent")
    caught: list[OpenAIError] = []

    async def decide(state: DecisionState) -> DecisionState:
        try:
            await observability.request_decision(
                sdk, model=REQUEST_MODEL, input=state["input"], questions=state["questions"],
            )
        except OpenAIError as exc:
            # Like the production graph, convert provider failure to a safe outcome.
            caught.append(exc)
            return {"failure": type(exc).__name__}
        pytest.fail("The failing SDK endpoint must propagate its error.")

    result = asyncio.run(native_decide_graph(decide).ainvoke(
        {"input": INPUT, "questions": QUESTIONS},
        config=observability.tracing_config("offline-failure-ticket"),
    ))
    assert caught == [error]
    assert caught[0] is error
    assert result["failure"] == "OpenAIError"
    llms = [span for span in traces.created if span["run_type"] == "llm"]
    assert len(llms) == 1
    update = completed_span(traces, llms[0])
    assert update["error"] == "OpenAI decision request failed (OpenAIError)."
    assert not update["outputs"]
    assert len(sdk.decisions.calls) == 1
    assert_no_client_or_secret_capture(traces)


def test_decision_service_uses_the_traced_sdk_boundary_inside_native_decide(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    traces = FakeClient()
    sdk = FakeSDKClient(sdk_decision())
    enable_tracing(monkeypatch, traces, "agent")
    service = DecisionService(SimpleNamespace(decision_model=REQUEST_MODEL), sdk)
    ticket = TicketInput(
        ticket_id="offline-service-ticket", customer_id="offline-customer",
        messages=[TicketMessage(text="An isolated request requiring specialist investigation.")],
    )
    analysis = TicketAnalysis(
        product=None, issue_type="test_issue", customer_sentiment="neutral",
        issues=[ExtractedIssue(issue_type="test_issue", description="Unresolved investigation", retrieval_query="support investigation")],
    )

    async def decide(state: DecisionState) -> DecisionState:
        triage = await service.decide(ticket, analysis, ToolContext(), handbook="Fictional test guidance.")
        return {"triage": triage}

    result = asyncio.run(native_decide_graph(decide).ainvoke(
        {"input": INPUT, "questions": QUESTIONS},
        config=observability.tracing_config(ticket.ticket_id),
    ))
    assert result["triage"].action.value == "route_to_specialist"
    assert len(sdk.decisions.calls) == 1
    request = sdk.decisions.calls[0]
    assert request["model"] == REQUEST_MODEL
    assert {question["name"] for question in request["questions"]} == {"urgency_0", "action_0", "owner_0"}
    llms = [span for span in traces.created if span["run_type"] == "llm"]
    assert len(llms) == 1
    assert llms[0]["inputs"]["input"] == request["input"]
    assert llms[0]["inputs"]["questions"] == request["questions"]
    update = completed_span(traces, llms[0])
    assert update["outputs"]["answers"] == sdk.decisions.response.model_dump(mode="json")["answers"]
    assert_no_client_or_secret_capture(traces)


class SelectivelyFailingTraceClient(FakeClient):
    """Leave native graph callbacks operational while failing only the LLM span."""

    def __init__(self, failure: str) -> None:
        super().__init__()
        self.failure = failure
        self.failed_creates: list[dict] = []
        self.failed_updates: list[dict] = []

    def create_run(self, **kwargs: Any) -> None:
        if self.failure == "create" and kwargs.get("name") == "OpenAI Decisions":
            self.failed_creates.append(kwargs)
            raise RuntimeError(TELEMETRY_SECRET)
        super().create_run(**kwargs)

    def update_run(self, **kwargs: Any) -> None:
        if self.failure == "update" and kwargs.get("name") == "OpenAI Decisions":
            self.failed_updates.append(kwargs)
            raise RuntimeError(TELEMETRY_SECRET)
        super().update_run(**kwargs)


@pytest.mark.parametrize("failure", ["create", "update", "output"])
def test_telemetry_failures_preserve_sdk_result_and_restore_native_parent(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, failure: str,
) -> None:
    traces = SelectivelyFailingTraceClient(failure)
    response = sdk_decision()
    sdk = FakeSDKClient(response)
    enable_tracing(monkeypatch, traces, "agent")
    output_attempts: list[dict] = []
    if failure == "output":
        original_set = RunTree.set

        def failing_set(run: RunTree, **kwargs: Any):
            if run.name == "OpenAI Decisions" and "usage_metadata" in kwargs:
                output_attempts.append(kwargs)
                raise RuntimeError(TELEMETRY_SECRET)
            return original_set(run, **kwargs)

        monkeypatch.setattr(RunTree, "set", failing_set)

    parents: list[tuple[Any, Any]] = []

    async def decide(state: DecisionState) -> DecisionState:
        before = get_tracing_context()["parent"]
        assert before.name == "decide"
        result = await observability.request_decision(
            sdk, model=REQUEST_MODEL, input=state["input"], questions=state["questions"],
        )
        after = get_tracing_context()["parent"]
        parents.append((before, after))
        assert after is before
        assert result is response
        return {"response": result}

    with caplog.at_level(logging.WARNING):
        result = asyncio.run(native_decide_graph(decide).ainvoke(
            {"input": INPUT, "questions": QUESTIONS},
            config=observability.tracing_config("offline-telemetry-failure"),
        ))

    assert result["response"] is response
    assert len(sdk.decisions.calls) == 1
    assert len(parents) == 1
    assert parents[0][0] is parents[0][1]
    # The ordinary graph spans still complete; telemetry is not a triage failure.
    assert {span["name"] for span in traces.updated if span["name"] != "OpenAI Decisions"} == {
        "SupportTicketTriage", "decide"
    }
    assert all(span["error"] is None for span in traces.updated)
    if failure == "create":
        assert len(traces.failed_creates) == 1
        assert "did not start (RuntimeError)" in caplog.text
    elif failure == "update":
        assert len(traces.failed_updates) == 1
        assert "did not finish (RuntimeError)" in caplog.text
    else:
        assert len(output_attempts) == 1
        assert "output was not recorded (RuntimeError)" in caplog.text
    assert TELEMETRY_SECRET not in caplog.text
    assert PROVIDER_SECRET not in caplog.text
    assert SDK_SECRET not in caplog.text
    assert INPUT not in caplog.text
    assert_no_client_or_secret_capture(traces)


def test_telemetry_exit_failure_does_not_replace_original_sdk_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    traces = SelectivelyFailingTraceClient("update")
    original_error = OpenAIError(PROVIDER_SECRET)
    sdk = FakeSDKClient(sdk_decision(), original_error)
    enable_tracing(monkeypatch, traces, "agent")
    caught: list[OpenAIError] = []
    parents: list[tuple[Any, Any]] = []

    async def decide(state: DecisionState) -> DecisionState:
        before = get_tracing_context()["parent"]
        assert before.name == "decide"
        try:
            await observability.request_decision(
                sdk, model=REQUEST_MODEL, input=state["input"], questions=state["questions"],
            )
        except OpenAIError as exc:
            after = get_tracing_context()["parent"]
            caught.append(exc)
            parents.append((before, after))
            assert after is before
            return {"failure": type(exc).__name__}
        pytest.fail("Telemetry failure must not swallow the provider failure.")

    with caplog.at_level(logging.WARNING):
        result = asyncio.run(native_decide_graph(decide).ainvoke(
            {"input": INPUT, "questions": QUESTIONS},
            config=observability.tracing_config("offline-double-failure"),
        ))

    assert result["failure"] == "OpenAIError"
    assert caught == [original_error]
    assert caught[0] is original_error
    assert len(sdk.decisions.calls) == 1
    assert len(parents) == 1
    assert parents[0][0] is parents[0][1]
    assert len(traces.failed_updates) == 1
    assert traces.failed_updates[0]["error"] == "OpenAI decision request failed (OpenAIError)."
    assert "did not finish (RuntimeError)" in caplog.text
    assert TELEMETRY_SECRET not in caplog.text
    assert PROVIDER_SECRET not in caplog.text
    assert SDK_SECRET not in caplog.text
    assert_no_client_or_secret_capture(traces)
