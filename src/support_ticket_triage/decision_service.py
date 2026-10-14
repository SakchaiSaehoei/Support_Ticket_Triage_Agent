"""Decisions API boundary for per-issue urgency, action, and ownership choices."""

import math

from openai import AsyncOpenAI, OpenAIError

from support_ticket_triage.config import Setting
from support_ticket_triage.observability import request_decision
from support_ticket_triage.prompts import (
    evidence_payload, model_instructions, routing_instructions, urgency_instructions,
)
from support_ticket_triage.schemas import (
    IssueDecision, NextAction, SpecialistTeam, TicketAnalysis, TicketInput,
    ToolContext, TriageDecision, Urgency,
)
from support_ticket_triage.triage_policy import TriageServiceError, aggregate_decisions


class DecisionService:
    """Use fixed model choices; enforce ticket aggregation in Python."""

    def __init__(
        self, settings: Setting, client: AsyncOpenAI, *, minimum_confidence: float = 0.75,
    ) -> None:
        if not math.isfinite(minimum_confidence) or not 0 <= minimum_confidence <= 1:
            raise ValueError("Minimum decision confidence must be between zero and one.")
        self.model = settings.decision_model
        self.client = client
        self.minimum_confidence = minimum_confidence

    async def decide(
        self, ticket: TicketInput, analysis: TicketAnalysis, context: ToolContext,
        *, handbook: str,
    ) -> TriageDecision:
        unresolved = [i for i, issue in enumerate(analysis.issues) if not issue.resolved]
        if not unresolved:
            return aggregate_decisions(analysis, context, [], None)

        questions = []
        for index in unresolved:
            for field, choices, policy in (
                ("urgency", Urgency, urgency_instructions()),
                ("action", NextAction, routing_instructions()),
                ("owner", SpecialistTeam, routing_instructions()),
            ):
                questions.append({
                    "type": "choice", "name": f"{field}_{index}",
                    "instructions": model_instructions(
                        f"Choose {field} only for analysis.issues[{index}].\n{policy}", handbook,
                    ),
                    "choices": [{"value": choice.value} for choice in choices],
                })
        try:
            response = await request_decision(
                self.client,
                model=self.model,
                input=evidence_payload(
                    ticket=ticket.model_dump(mode="json"),
                    analysis=analysis.model_dump(mode="json"),
                    context=context.model_dump(mode="json"),
                ),
                questions=questions,
            )
        except OpenAIError as exc:
            raise TriageServiceError("OpenAI decision request failed; human review is required.") from exc

        expected = {question["name"] for question in questions}
        answers = {}
        returned_answers = getattr(response, "answers", None)
        if not isinstance(returned_answers, list):
            raise TriageServiceError("OpenAI returned no decision answers.")
        for answer in returned_answers:
            name = getattr(answer, "name", None)
            if name not in expected or name in answers:
                raise TriageServiceError("OpenAI returned unknown or duplicate decision answers.")
            if getattr(answer, "type", None) != "choice":
                raise TriageServiceError("OpenAI refused a decision or returned the wrong answer type.")
            confidence = getattr(answer, "confidence", None)
            if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or (
                not math.isfinite(confidence) or not 0 <= confidence <= 1
            ):
                raise TriageServiceError("OpenAI returned invalid decision confidence.")
            answers[name] = answer
        if set(answers) != expected:
            raise TriageServiceError("OpenAI did not answer every required issue decision.")
        try:
            issue_decisions = [IssueDecision(
                issue_index=index,
                urgency=Urgency(answers[f"urgency_{index}"].choice),
                action=NextAction(answers[f"action_{index}"].choice),
                owner=SpecialistTeam(answers[f"owner_{index}"].choice),
            ) for index in unresolved]
        except (ValueError, TypeError, AttributeError) as exc:
            raise TriageServiceError("OpenAI returned an unsupported decision choice.") from exc
        # Only uncertainty about an action can trigger the action-confidence gate.
        # Urgency and ownership are separate choices, not routing confidence.
        action_confidence = min(answers[f"action_{index}"].confidence for index in unresolved)
        return aggregate_decisions(
            analysis, context, issue_decisions, action_confidence,
            minimum_confidence=self.minimum_confidence,
        )
