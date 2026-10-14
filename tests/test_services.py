"""Core triage contracts with SDK-shaped responses; no credentials or network."""

import asyncio
from datetime import datetime, timezone
import json
from types import SimpleNamespace

import httpx2
import pytest
from openai import APIConnectionError
from openai.types.decision import Decision
from openai.types.responses import Response
from openai.types.responses.parsed_response import ParsedResponse

from support_ticket_triage.decision_service import DecisionService
from support_ticket_triage.response_service import ResponseService, fallback_handoff_reply
from support_ticket_triage.schemas import (
    ExtractedIssue, HandoffReceipt, KnowledgeBaseArticle, NextAction, SpecialistTeam,
    TicketAnalysis, TicketInput, TicketMessage, ToolContext, TriageDecision, Urgency,
)
from support_ticket_triage.triage_policy import TriageServiceError

SETTINGS = SimpleNamespace(decision_model="gpt-6-luna", response_model="gpt-6-luna")
HANDBOOK = "Follow verified evidence. Preserve each unresolved issue."


class FakeEndpoint:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result

    async def parse(self, **kwargs):
        return await self.create(**kwargs)


def ticket():
    return TicketInput(ticket_id="ticket-test", customer_id="customer-test", messages=[
        TicketMessage(text="An existing problem still blocks my work.", time_label="yesterday"),
        TicketMessage(text="Also, can I change a preference?", time_label="today"),
    ])


def analysis(*issues, language="en"):
    return TicketAnalysis(
        product=None, issue_type="usage", customer_sentiment="concerned", language=language,
        issues=list(issues) or [ExtractedIssue(
            issue_type="usage", description="Need supported guidance",
            retrieval_query="documented usage guidance",
        )],
    )


def context(**overrides):
    articles = overrides.pop("articles", [KnowledgeBaseArticle(
        article_id="guide:usage", title="Usage guidance", content="Collect relevant context.",
    )])
    return ToolContext(**{
        "articles": articles,
        "article_ids_by_issue": {"0": [article.article_id for article in articles]},
        **overrides,
    })


def answer(name, value, confidence=0.95):
    return {
        "name": name, "type": "choice", "choice": value, "confidence": confidence,
        "probabilities": [{"value": value, "probability": confidence}],
    }


def issue_answers(index, urgency="low", action="auto_respond", owner="general_support", confidence=0.95):
    return [
        answer(f"urgency_{index}", urgency, confidence),
        answer(f"action_{index}", action, confidence),
        answer(f"owner_{index}", owner, confidence),
    ]


def sdk_decision(answers):
    return Decision.model_validate({
        "model": "gpt-6-luna", "answers": answers,
        "usage": {
            "input_tokens": 10, "output_tokens": 3, "total_tokens": 13,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    })


def sdk_response(*, text="Helpful guidance", parsed=None, status="completed", refusal=False):
    content = (
        {"type": "refusal", "refusal": "Unable to answer"} if refusal else
        {"type": "output_text", "text": text, "annotations": [], "parsed": parsed}
    )
    raw = {
        "id": "resp_test", "created_at": 0, "model": "gpt-6-luna", "object": "response",
        "parallel_tool_calls": False, "tool_choice": "auto", "tools": [], "status": status,
        "output": [{"id": "msg_test", "role": "assistant", "type": "message",
                    "status": "completed", "content": [content]}],
    }
    if parsed is not None:
        return ParsedResponse[TicketAnalysis].model_validate(raw)
    return Response.model_validate(raw)


def decision_service(answers):
    endpoint = FakeEndpoint(result=sdk_decision(answers))
    return DecisionService(SETTINGS, SimpleNamespace(decisions=endpoint)), endpoint


def decide(service, analysis_data=None, tool_context=None):
    return asyncio.run(service.decide(
        ticket(), analysis_data or analysis(), tool_context or context(), handbook=HANDBOOK,
    ))


def test_highest_unresolved_urgency_and_most_demanding_action_are_independent():
    data = analysis(
        ExtractedIssue(issue_type="resolved", description="Old problem", resolved=True),
        ExtractedIssue(issue_type="billing", description="Financial blocker", retrieval_query="billing"),
        ExtractedIssue(issue_type="technical", description="Unknown cause", retrieval_query="technical"),
    )
    service, endpoint = decision_service(
        issue_answers(1, "high", "route_to_specialist", "billing")
        + issue_answers(2, "medium", "escalate_to_human", "technical_support")
    )
    result = decide(service, data)

    assert result.urgency == Urgency.HIGH
    assert result.action == NextAction.ESCALATE_TO_HUMAN
    assert result.owner == SpecialistTeam.TECHNICAL_SUPPORT
    assert [item.issue_index for item in result.issue_decisions] == [1, 2]
    assert {question["name"] for question in endpoint.calls[0]["questions"]} == {
        "urgency_1", "action_1", "owner_1", "urgency_2", "action_2", "owner_2",
    }
    assert all(question["type"] == "choice" for question in endpoint.calls[0]["questions"])
    assert all(HANDBOOK in question["instructions"] for question in endpoint.calls[0]["questions"])


@pytest.mark.parametrize("values,expected", [
    (("low", "medium"), Urgency.MEDIUM),
    (("medium", "high"), Urgency.HIGH),
    (("high", "critical"), Urgency.CRITICAL),
])
def test_urgency_aggregation_uses_explicit_severity_order(values, expected):
    data = analysis(*[ExtractedIssue(
        issue_type="usage", description=f"Problem {i}", retrieval_query=f"usage {i}",
    ) for i in range(2)])
    service, _ = decision_service(issue_answers(0, values[0]) + issue_answers(1, values[1]))
    assert decide(service, data).urgency == expected


@pytest.mark.parametrize("condition,expected_action", [
    ("no_articles", NextAction.ROUTE_TO_SPECIALIST),
    ("account_action", NextAction.ROUTE_TO_SPECIALIST),
    ("failed_tool", NextAction.ESCALATE_TO_HUMAN),
    ("low_confidence", NextAction.ESCALATE_TO_HUMAN),
])
def test_unsupported_auto_response_is_blocked_without_inventing_urgency(condition, expected_action):
    data = analysis()
    ctx = context()
    confidence = 0.95
    if condition == "no_articles":
        ctx.articles = []
    elif condition == "account_action":
        data.issues[0].requires_account_action = True
    elif condition == "failed_tool":
        ctx.tool_errors = ["Customer lookup unavailable"]
    else:
        confidence = 0.2
    service, _ = decision_service(issue_answers(0, "medium", confidence=confidence))
    result = decide(service, data, ctx)
    assert result.urgency == Urgency.MEDIUM
    assert result.action == expected_action
    assert result.issue_decisions[0].action == expected_action


def test_critical_impact_cannot_be_automatically_answered():
    service, _ = decision_service(issue_answers(0, "critical"))
    result = decide(service)
    assert result.urgency == Urgency.CRITICAL
    assert result.action == NextAction.ESCALATE_TO_HUMAN


@pytest.mark.parametrize("fault", ["refusal", "missing", "invalid", "duplicate", "nan_confidence"])
def test_untrustworthy_decision_outputs_raise_explicit_safe_error(fault):
    answers = issue_answers(0)
    if fault == "refusal":
        answers[0] = {"type": "refusal", "name": "urgency_0"}
    elif fault == "missing":
        answers.pop()
    elif fault == "invalid":
        answers[0] = answer("urgency_0", "unsupported")
    elif fault == "duplicate":
        answers.append(answers[0])
    else:
        answers[0] = answer("urgency_0", "low", float("nan"))
    service, _ = decision_service(answers)
    with pytest.raises(TriageServiceError):
        decide(service)


def test_decision_api_failure_is_safe_and_does_not_echo_provider_text():
    error = APIConnectionError(request=httpx2.Request("POST", "https://example.test"))
    endpoint = FakeEndpoint(error=error)
    service = DecisionService(SETTINGS, SimpleNamespace(decisions=endpoint))
    with pytest.raises(TriageServiceError, match="human review"):
        decide(service)


def test_resolved_issues_do_not_trigger_decision_calls():
    endpoint = FakeEndpoint()
    service = DecisionService(SETTINGS, SimpleNamespace(decisions=endpoint))
    result = decide(service, analysis(ExtractedIssue(
        issue_type="usage", description="Customer confirmed resolution", resolved=True,
    )))
    assert endpoint.calls == []
    assert result.urgency == Urgency.LOW
    assert result.issue_decisions == []


def test_extraction_uses_whole_history_prior_hint_and_typed_sdk_parse():
    output = analysis(
        ExtractedIssue(issue_type="blocker", description="Earlier unresolved problem",
                       retrieval_query="unresolved workflow failure"),
        ExtractedIssue(issue_type="preference", description="New preference question",
                       retrieval_query="configuration preferences"),
        language="th",
    )
    endpoint = FakeEndpoint(result=sdk_response(parsed=output))
    service = ResponseService(SETTINGS, SimpleNamespace(responses=endpoint))
    previous = analysis()
    result = asyncio.run(service.extract_information(
        ticket(), context(previous_analysis=previous), handbook=HANDBOOK,
    ))
    request = endpoint.calls[0]
    evidence = json.loads(request["input"][0]["content"])
    assert result == output
    assert request["text_format"] is TicketAnalysis
    assert request["store"] is False
    assert evidence["ticket"]["messages"] == ticket().model_dump()["messages"]
    assert evidence["context"]["previous_analysis"] == previous.model_dump()
    assert HANDBOOK in request["instructions"]
    assert "untrusted data" in request["instructions"]
    assert "English retrieval_query" in request["instructions"]


def test_temporal_reference_reaches_both_model_services_without_rewriting_old_labels():
    incoming = TicketInput(
        ticket_id="timed-ticket", customer_id="customer-test",
        as_of=datetime(2026, 1, 3, 9, tzinfo=timezone.utc),
        messages=[TicketMessage(
            text="I have a presentation in two hours.", time_label="just now",
            observed_at=datetime(2026, 1, 1, 9, tzinfo=timezone.utc),
        )],
    )
    data = analysis()
    responses = FakeEndpoint(result=sdk_response(parsed=data))
    decisions = FakeEndpoint(result=sdk_decision(issue_answers(0)))

    async def run_services():
        await ResponseService(SETTINGS, SimpleNamespace(responses=responses)).extract_information(
            incoming, context(), handbook=HANDBOOK,
        )
        await DecisionService(SETTINGS, SimpleNamespace(decisions=decisions)).decide(
            incoming, data, context(), handbook=HANDBOOK,
        )

    asyncio.run(run_services())
    payloads = [
        json.loads(responses.calls[0]["input"][0]["content"]),
        json.loads(decisions.calls[0]["input"]),
    ]
    for payload in payloads:
        assert payload["ticket"] == incoming.model_dump(mode="json")
        assert payload["ticket"]["messages"][0]["time_label"] == "just now"
    for instructions in [responses.calls[0]["instructions"], decisions.calls[0]["questions"][0]["instructions"]]:
        assert "Ticket.as_of" in instructions
        assert "historical timing is unknown" in instructions
        assert "not the current evaluation time" in instructions


@pytest.mark.parametrize("fault", ["refusal", "incomplete", "no_analysis", "missing_query"])
def test_extraction_rejects_unusable_output(fault):
    output = analysis()
    if fault == "missing_query":
        output.issues[0].retrieval_query = ""
    response = sdk_response(
        parsed=None if fault in {"refusal", "no_analysis"} else output,
        refusal=fault == "refusal", status="incomplete" if fault == "incomplete" else "completed",
    )
    if fault == "no_analysis":
        response = SimpleNamespace(status="completed", output=[], output_parsed=None)
    endpoint = FakeEndpoint(result=response)
    service = ResponseService(SETTINGS, SimpleNamespace(responses=endpoint))
    with pytest.raises(TriageServiceError):
        asyncio.run(service.extract_information(ticket(), context(), handbook=HANDBOOK))


def test_reply_uses_customer_language_and_all_unresolved_issues_as_evidence():
    endpoint = FakeEndpoint(result=sdk_response(text="ขอข้อมูลเพิ่มเติมเพื่อช่วยตรวจสอบครับ"))
    service = ResponseService(SETTINGS, SimpleNamespace(responses=endpoint))
    data = analysis(language="th")
    decision = TriageDecision(urgency="low", action="auto_respond", confidence=0.95)
    reply = asyncio.run(service.draft_reply(
        ticket(), data, decision, context(), handbook=HANDBOOK,
    ))
    request = endpoint.calls[0]
    assert reply == "ขอข้อมูลเพิ่มเติมเพื่อช่วยตรวจสอบครับ"
    assert json.loads(request["input"][0]["content"])["analysis"]["language"] == "th"
    assert "every unresolved issue" in request["instructions"]
    assert "these read-only tools perform none" in request["instructions"]
    assert request["store"] is False


@pytest.mark.parametrize("fault", ["specialist", "no_evidence", "account_action", "tool_failure"])
def test_reply_is_blocked_before_model_call_when_automation_is_unsupported(fault):
    endpoint = FakeEndpoint()
    service = ResponseService(SETTINGS, SimpleNamespace(responses=endpoint))
    data, ctx = analysis(), context()
    decision = TriageDecision(urgency="low", action="auto_respond")
    if fault == "specialist":
        decision.action = NextAction.ROUTE_TO_SPECIALIST
    elif fault == "no_evidence":
        ctx.articles = []
    elif fault == "account_action":
        data.issues[0].requires_account_action = True
    else:
        ctx.tool_errors = ["Lookup failed"]
    with pytest.raises(TriageServiceError):
        asyncio.run(service.draft_reply(ticket(), data, decision, ctx, handbook=HANDBOOK))
    assert endpoint.calls == []


@pytest.mark.parametrize("fault", ["refusal", "empty", "incomplete"])
def test_reply_refusal_or_incomplete_text_cannot_be_published(fault):
    endpoint = FakeEndpoint(result=sdk_response(
        text=" " if fault == "empty" else "Guidance", refusal=fault == "refusal",
        status="incomplete" if fault == "incomplete" else "completed",
    ))
    service = ResponseService(SETTINGS, SimpleNamespace(responses=endpoint))
    decision = TriageDecision(urgency="low", action="auto_respond")
    with pytest.raises(TriageServiceError):
        asyncio.run(service.draft_reply(ticket(), analysis(), decision, context(), handbook=HANDBOOK))


def test_evidence_for_one_issue_cannot_authorize_another_issue():
    data = analysis(
        ExtractedIssue(issue_type="usage", description="Supported question", retrieval_query="usage"),
        ExtractedIssue(issue_type="capability", description="Undocumented question", retrieval_query="capability"),
    )
    ctx = context(article_ids_by_issue={"0": ["guide:usage"], "1": []})
    service, _ = decision_service(issue_answers(0) + issue_answers(1, owner="product"))

    result = decide(service, data, ctx)

    assert result.issue_decisions[0].action == NextAction.AUTO_RESPOND
    assert result.issue_decisions[1].action == NextAction.ROUTE_TO_SPECIALIST
    assert result.action == NextAction.ROUTE_TO_SPECIALIST
    assert result.owner == SpecialistTeam.PRODUCT


@pytest.mark.parametrize("mapping", [
    {},
    {"0": []},
    {"0": ["guide:missing"]},
    {"0": ["guide:usage", "guide:missing"]},
])
def test_auto_response_requires_complete_valid_issue_article_references(mapping):
    service, _ = decision_service(issue_answers(0))
    result = decide(service, tool_context=context(article_ids_by_issue=mapping))
    assert result.urgency == Urgency.LOW
    assert result.action == NextAction.ROUTE_TO_SPECIALIST


@pytest.mark.parametrize("mapping", [{}, {"0": ["guide:missing"]}, {"0": ["guide:usage"], "1": []}])
def test_draft_reply_rechecks_evidence_for_every_unresolved_issue(mapping):
    endpoint = FakeEndpoint()
    service = ResponseService(SETTINGS, SimpleNamespace(responses=endpoint))
    data = analysis(
        ExtractedIssue(issue_type="usage", description="First question", retrieval_query="usage"),
        ExtractedIssue(issue_type="usage", description="Second question", retrieval_query="usage"),
    )
    decision = TriageDecision(urgency="low", action="auto_respond", confidence=0.95)
    with pytest.raises(TriageServiceError, match="evidence"):
        asyncio.run(service.draft_reply(
            ticket(), data, decision, context(article_ids_by_issue=mapping), handbook=HANDBOOK,
        ))
    assert endpoint.calls == []


def test_resolved_issues_do_not_require_article_references_for_a_safe_reply():
    data = analysis(
        ExtractedIssue(issue_type="old", description="Confirmed resolved", resolved=True),
        ExtractedIssue(issue_type="usage", description="Current question", retrieval_query="usage"),
    )
    ctx = context(article_ids_by_issue={"1": ["guide:usage"]})
    service, _ = decision_service(issue_answers(1))
    decision = decide(service, data, ctx)
    assert decision.action == NextAction.AUTO_RESPOND
    endpoint = FakeEndpoint(result=sdk_response(text="Supported next step"))
    response_service = ResponseService(SETTINGS, SimpleNamespace(responses=endpoint))
    assert asyncio.run(response_service.draft_reply(
        ticket(), data, decision, ctx, handbook=HANDBOOK,
    )) == "Supported next step"


def test_low_urgency_confidence_does_not_override_confident_specialist_action():
    service, _ = decision_service([
        answer("urgency_0", "medium", 0.36),
        answer("action_0", "route_to_specialist", 1.0),
        answer("owner_0", "technical_support", 0.95),
    ])
    result = decide(service)
    assert result.urgency == Urgency.MEDIUM
    assert result.action == NextAction.ROUTE_TO_SPECIALIST
    assert result.confidence == 1.0


def test_low_urgency_confidence_does_not_override_supported_auto_response():
    service, _ = decision_service([
        answer("urgency_0", "low", 0.36),
        answer("action_0", "auto_respond", 1.0),
        answer("owner_0", "general_support", 0.95),
    ])
    result = decide(service)
    assert result.urgency == Urgency.LOW
    assert result.action == NextAction.AUTO_RESPOND
    assert result.confidence == 1.0


def test_low_owner_confidence_does_not_override_confident_action():
    service, _ = decision_service([
        answer("urgency_0", "medium", 0.95),
        answer("action_0", "route_to_specialist", 1.0),
        answer("owner_0", "technical_support", 0.1),
    ])
    result = decide(service)
    assert result.action == NextAction.ROUTE_TO_SPECIALIST
    assert result.owner == SpecialistTeam.TECHNICAL_SUPPORT
    assert result.confidence == 1.0


def test_low_action_confidence_still_requires_human_review():
    service, _ = decision_service([
        answer("urgency_0", "medium", 1.0),
        answer("action_0", "route_to_specialist", 0.74),
        answer("owner_0", "technical_support", 1.0),
    ])
    result = decide(service)
    assert result.urgency == Urgency.MEDIUM
    assert result.action == NextAction.ESCALATE_TO_HUMAN
    assert result.confidence == 0.74


def test_multi_issue_confident_specialist_actions_keep_maximum_urgency_without_escalation():
    data = analysis(
        ExtractedIssue(issue_type="malfunction", description="Unresolved malfunction", retrieval_query="malfunction"),
        ExtractedIssue(issue_type="enhancement", description="Additional request", retrieval_query="enhancement"),
    )
    service, _ = decision_service([
        answer("urgency_0", "medium", 0.36),
        answer("action_0", "route_to_specialist", 1.0),
        answer("owner_0", "technical_support", 0.95),
        answer("urgency_1", "low", 0.99),
        answer("action_1", "route_to_specialist", 1.0),
        answer("owner_1", "product", 0.95),
    ])
    result = decide(service, data)
    assert result.urgency == Urgency.MEDIUM
    assert result.action == NextAction.ROUTE_TO_SPECIALIST
    assert all(item.action == NextAction.ROUTE_TO_SPECIALIST for item in result.issue_decisions)
    assert result.confidence == 1.0


def test_action_confidence_is_minimum_across_unresolved_action_answers_only():
    data = analysis(
        ExtractedIssue(issue_type="old", description="Confirmed resolved", resolved=True),
        ExtractedIssue(issue_type="malfunction", description="Current failure", retrieval_query="malfunction"),
        ExtractedIssue(issue_type="usage", description="Separate usage question", retrieval_query="usage"),
    )
    service, endpoint = decision_service([
        answer("urgency_1", "medium", 0.36),
        answer("action_1", "route_to_specialist", 0.9),
        answer("owner_1", "technical_support", 0.1),
        answer("urgency_2", "low", 0.99),
        answer("action_2", "route_to_specialist", 0.8),
        answer("owner_2", "general_support", 0.2),
    ])
    result = decide(service, data)
    assert result.confidence == 0.8
    assert result.action == NextAction.ROUTE_TO_SPECIALIST
    assert result.urgency == Urgency.MEDIUM
    assert [item.issue_index for item in result.issue_decisions] == [1, 2]
    assert all(not question["name"].endswith("_0") for question in endpoint.calls[0]["questions"])


def test_one_low_confidence_action_escalates_the_multi_issue_ticket():
    data = analysis(
        ExtractedIssue(issue_type="usage", description="First question", retrieval_query="usage"),
        ExtractedIssue(issue_type="usage", description="Second question", retrieval_query="usage"),
    )
    service, _ = decision_service([
        answer("urgency_0", "medium", 1.0),
        answer("action_0", "route_to_specialist", 0.99),
        answer("owner_0", "general_support", 1.0),
        answer("urgency_1", "low", 1.0),
        answer("action_1", "route_to_specialist", 0.6),
        answer("owner_1", "general_support", 1.0),
    ])
    result = decide(service, data)
    assert result.confidence == 0.6
    assert result.urgency == Urgency.MEDIUM
    assert result.action == NextAction.ESCALATE_TO_HUMAN


def handoff_decision(action=NextAction.ROUTE_TO_SPECIALIST):
    return TriageDecision(
        urgency=Urgency.MEDIUM, action=action, owner=SpecialistTeam.TECHNICAL_SUPPORT,
        reason="An unresolved malfunction requires review.",
    )


def handoff_receipt(decision, *, follow_up_requested=False):
    return HandoffReceipt(
        dispatch_id="offline-dispatch-id", action=decision.action, owner=decision.owner,
        destination=(
            f"mock-specialist:{decision.owner.value}"
            if decision.action == NextAction.ROUTE_TO_SPECIALIST else "mock-staff:review-queue"
        ),
        status="accepted", is_mock=True, follow_up_requested=follow_up_requested,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


@pytest.mark.parametrize("action", [NextAction.ROUTE_TO_SPECIALIST, NextAction.ESCALATE_TO_HUMAN])
@pytest.mark.parametrize("language", ["en", "th"])
@pytest.mark.parametrize("accepted", [True, False])
def test_handoff_acknowledgement_preserves_action_and_verifiable_status(
    action, language, accepted
):
    body = "Thanks for explaining the unresolved access problem." if language == "en" else "ขอบคุณที่แจ้งปัญหาการเข้าใช้งานที่ยังไม่ได้รับการแก้ไข"
    endpoint = FakeEndpoint(result=sdk_response(text=body))
    sdk = SimpleNamespace(responses=endpoint, api_key="synthetic-sdk-secret-not-in-payload")
    service = ResponseService(SETTINGS, sdk)
    decision = handoff_decision(action)
    original_decision = decision.model_dump()
    receipt = handoff_receipt(decision, follow_up_requested=True) if accepted else None
    data = analysis(ExtractedIssue(
        issue_type="access", description="The customer still cannot access export tools.",
        retrieval_query="access failure", requires_account_action=True,
    ), language=language)
    ctx = context(articles=[], tool_errors=["Knowledge lookup unavailable"])

    reply = asyncio.run(service.draft_handoff_reply(
        ticket(), data, decision, ctx, receipt, handbook=HANDBOOK,
    ))

    assert reply == f"{fallback_handoff_reply(language, decision, receipt)}\n\n{body}"
    assert "mock" not in reply.lower()
    assert "simulat" not in reply.lower()
    assert "no real staff" not in reply.lower()
    assert "จำลอง" not in reply
    assert "ไม่มีการติดต่อเจ้าหน้าที่จริง" not in reply
    if receipt:
        assert receipt.is_mock is True
    assert decision.model_dump() == original_decision
    request = endpoint.calls[0]
    payload = json.loads(request["input"][0]["content"])
    assert payload["analysis"]["language"] == language
    assert payload["decision"]["action"] == action.value
    assert payload["dispatch_receipt"] == (receipt.model_dump(mode="json") if receipt else None)
    assert payload["context"]["tool_errors"] == ["Knowledge lookup unavailable"]
    assert request["store"] is False
    assert HANDBOOK in request["instructions"]
    assert "acknowledgement body" in request["instructions"]
    assert "missing receipt" in request["instructions"]
    assert "synthetic-sdk-secret-not-in-payload" not in json.dumps(request)
    assert "api_key" not in payload


@pytest.mark.parametrize("fault", ["action", "owner", "status", "not_mock"])
def test_unmatched_or_unconfirmed_handoff_receipt_is_rejected_before_model_call(fault):
    decision = handoff_decision()
    receipt = handoff_receipt(decision)
    changes = {
        "action": {"action": NextAction.ESCALATE_TO_HUMAN},
        "owner": {"owner": SpecialistTeam.BILLING},
        "status": {"status": "pending"},
        "not_mock": {"is_mock": False},
    }
    receipt = receipt.model_copy(update=changes[fault])
    endpoint = FakeEndpoint(result=sdk_response())
    service = ResponseService(SETTINGS, SimpleNamespace(responses=endpoint))

    with pytest.raises(TriageServiceError, match="receipt"):
        asyncio.run(service.draft_handoff_reply(
            ticket(), analysis(), decision, context(), receipt, handbook=HANDBOOK,
        ))
    with pytest.raises(TriageServiceError, match="receipt"):
        fallback_handoff_reply("en", decision, receipt)
    assert endpoint.calls == []


def test_handoff_acknowledgement_cannot_reclassify_an_auto_response():
    decision = TriageDecision(urgency=Urgency.LOW, action=NextAction.AUTO_RESPOND)
    endpoint = FakeEndpoint(result=sdk_response())
    service = ResponseService(SETTINGS, SimpleNamespace(responses=endpoint))

    with pytest.raises(TriageServiceError, match="handoff decision"):
        asyncio.run(service.draft_handoff_reply(
            ticket(), analysis(), decision, context(), None, handbook=HANDBOOK,
        ))
    assert endpoint.calls == []
    assert decision.action == NextAction.AUTO_RESPOND


@pytest.mark.parametrize("fault", ["provider", "refusal", "incomplete", "empty"])
def test_failed_or_unusable_handoff_acknowledgement_requires_fallback(fault):
    endpoint = FakeEndpoint(result=sdk_response(
        text=" " if fault == "empty" else "Acknowledgement",
        refusal=fault == "refusal", status="incomplete" if fault == "incomplete" else "completed",
    ))
    if fault == "provider":
        endpoint.error = APIConnectionError(request=httpx2.Request("POST", "https://example.test"))
    service = ResponseService(SETTINGS, SimpleNamespace(responses=endpoint))
    decision = handoff_decision()

    with pytest.raises(TriageServiceError):
        asyncio.run(service.draft_handoff_reply(
            ticket(), analysis(), decision, context(), handoff_receipt(decision), handbook=HANDBOOK,
        ))
    assert len(endpoint.calls) == 1
    assert decision.action == NextAction.ROUTE_TO_SPECIALIST


@pytest.mark.parametrize("action", [NextAction.ROUTE_TO_SPECIALIST, NextAction.ESCALATE_TO_HUMAN])
@pytest.mark.parametrize("language", ["en", "th"])
@pytest.mark.parametrize("follow_up", [False, True])
def test_fallback_confirms_only_accepted_handoff_and_recorded_followup(action, language, follow_up):
    decision = handoff_decision(action)
    reply = fallback_handoff_reply(
        language, decision, handoff_receipt(decision, follow_up_requested=follow_up),
    )

    if language == "en":
        expected_status = "routed" if action == NextAction.ROUTE_TO_SPECIALIST else "escalated"
        assert expected_status in reply
        assert "Technical Support" in reply
        assert ("A team member will contact you to follow up." in reply) is follow_up
        assert "within" not in reply.lower()
    else:
        assert "ทีมสนับสนุนด้านเทคนิค" in reply
        assert ("จะติดต่อ" in reply) is follow_up
        assert "ภายใน" not in reply
    assert "mock" not in reply.lower()
    assert "simulat" not in reply.lower()
    assert "no real staff" not in reply.lower()
    assert "จำลอง" not in reply
    assert "ไม่มีการติดต่อเจ้าหน้าที่จริง" not in reply


@pytest.mark.parametrize("action", [NextAction.ROUTE_TO_SPECIALIST, NextAction.ESCALATE_TO_HUMAN])
@pytest.mark.parametrize("language", ["en", "th-TH"])
def test_fallback_without_receipt_never_claims_dispatch_or_notification(action, language):
    decision = handoff_decision(action)
    reply = fallback_handoff_reply(language, decision, None)

    if language == "en":
        assert "dispatch and staff notification have not been confirmed" in reply
        assert "has been routed" not in reply
        assert "has been escalated" not in reply
        assert "will contact" not in reply
    else:
        assert "ยังยืนยันการส่งต่อหรือการแจ้งเจ้าหน้าที่ไม่ได้" in reply
        assert "ส่งต่อคำขอของคุณแล้ว" not in reply
        assert "จะติดต่อ" not in reply
    assert "mock" not in reply.lower()
    assert "simulat" not in reply.lower()
    assert "no real staff" not in reply.lower()
    assert "จำลอง" not in reply
    assert "ไม่มีการติดต่อเจ้าหน้าที่จริง" not in reply
