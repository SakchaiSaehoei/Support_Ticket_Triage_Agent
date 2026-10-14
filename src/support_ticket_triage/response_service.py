"""Responses API boundary for complete-history extraction and grounded replies."""

from openai import AsyncOpenAI, OpenAIError
from pydantic import ValidationError

from support_ticket_triage.config import Setting
from support_ticket_triage.prompts import (
    evidence_payload, extraction_instructions, handoff_reply_instructions,
    model_instructions, reply_instructions,
)
from support_ticket_triage.schemas import (
    HandoffReceipt, NextAction, SpecialistTeam, TicketAnalysis, TicketInput,
    ToolContext, TriageDecision,
)
from support_ticket_triage.triage_policy import TriageServiceError, has_issue_evidence


def _check_response(response: object) -> None:
    """Refusals and incomplete generations must not become customer output."""
    if getattr(response, "status", None) != "completed":
        raise TriageServiceError("OpenAI did not complete the response.")
    for item in getattr(response, "output", []):
        for content in getattr(item, "content", []):
            if getattr(content, "type", None) == "refusal":
                raise TriageServiceError("OpenAI refused the requested response.")


def _validate_handoff_receipt(
    decision: TriageDecision, receipt: HandoffReceipt | None,
) -> None:
    if decision.action not in {
        NextAction.ROUTE_TO_SPECIALIST, NextAction.ESCALATE_TO_HUMAN,
    }:
        raise TriageServiceError("A handoff acknowledgement requires a handoff decision.")
    if receipt is not None and (
        receipt.action != decision.action or receipt.owner != decision.owner
        or receipt.status != "accepted" or receipt.is_mock is not True
    ):
        raise TriageServiceError("The handoff receipt does not confirm the selected mock dispatch.")


def fallback_handoff_reply(
    language: str, decision: TriageDecision, receipt: HandoffReceipt | None,
) -> str:
    """Describe confirmed handoff status in customer-facing support language."""
    _validate_handoff_receipt(decision, receipt)
    thai = language.strip().lower().split("-", 1)[0] == "th"
    teams = {
        SpecialistTeam.BILLING: ("Billing", "ทีมเรียกเก็บเงิน"),
        SpecialistTeam.TECHNICAL_SUPPORT: ("Technical Support", "ทีมสนับสนุนด้านเทคนิค"),
        SpecialistTeam.PRODUCT: ("Product", "ทีมผลิตภัณฑ์"),
        SpecialistTeam.GENERAL_SUPPORT: ("General Support", "ทีมสนับสนุนทั่วไป"),
    }
    english_team, thai_team = teams[decision.owner]
    team = thai_team if thai else english_team
    if thai:
        if receipt is None:
            return (
                f"ประเด็นของคุณยังต้องได้รับการตรวจสอบ แนะนำให้ {team} ตรวจสอบ "
                "แต่ยังยืนยันการส่งต่อหรือการแจ้งเจ้าหน้าที่ไม่ได้"
            )
        accepted = (
            f"ส่งเรื่องของคุณให้{team}ตรวจสอบแล้ว"
            if decision.action == NextAction.ROUTE_TO_SPECIALIST else
            f"แจ้งเจ้าหน้าที่และส่งเรื่องของคุณให้{team}ตรวจสอบแล้ว"
        )
        follow_up = (
            "ทีมงานจะติดต่อกลับเพื่อติดตามเรื่องนี้"
            if receipt.follow_up_requested else ""
        )
        return " ".join(part for part in (accepted, follow_up) if part)
    if receipt is None:
        return (
            f"Your concerns remain open. Review by {team} is recommended, but dispatch "
            "and staff notification have not been confirmed."
        )
    accepted = (
        f"Your ticket has been routed to our {team} team for review."
        if decision.action == NextAction.ROUTE_TO_SPECIALIST else
        f"Your ticket has been escalated to our {team} team for human review."
    )
    follow_up = (
        "A team member will contact you to follow up."
        if receipt.follow_up_requested else ""
    )
    return " ".join(part for part in (accepted, follow_up) if part)


class ResponseService:
    """Generate structured analysis and customer text through the shared client."""

    def __init__(self, settings: Setting, client: AsyncOpenAI) -> None:
        self.model = settings.response_model
        self.client = client

    async def extract_information(
        self, ticket: TicketInput, context: ToolContext, *, handbook: str,
    ) -> TicketAnalysis:
        try:
            response = await self.client.responses.parse(
                model=self.model,
                instructions=model_instructions(extraction_instructions(), handbook),
                input=[{"role": "user", "content": evidence_payload(
                    ticket=ticket.model_dump(mode="json"),
                    context=context.model_dump(mode="json"),
                )}],
                text_format=TicketAnalysis, store=False,
            )
            _check_response(response)
            if response.output_parsed is None:
                raise TriageServiceError("OpenAI returned no structured ticket analysis.")
            analysis = TicketAnalysis.model_validate(response.output_parsed)
        except OpenAIError as exc:
            raise TriageServiceError("OpenAI information extraction failed; human review is required.") from exc
        except ValidationError as exc:
            raise TriageServiceError("OpenAI returned an invalid ticket analysis.") from exc
        if not analysis.language.strip() or any(
            not issue.retrieval_query.strip() for issue in analysis.issues if not issue.resolved
        ):
            raise TriageServiceError("Ticket analysis lacks a reply language or issue search query.")
        return analysis

    async def draft_reply(
        self, ticket: TicketInput, analysis: TicketAnalysis, decision: TriageDecision,
        context: ToolContext, *, handbook: str,
    ) -> str:
        unresolved = [(index, issue) for index, issue in enumerate(analysis.issues) if not issue.resolved]
        if decision.action != NextAction.AUTO_RESPOND:
            raise TriageServiceError("A customer reply requires an approved auto-response decision.")
        if context.tool_errors or any(
            not has_issue_evidence(context, index) or issue.requires_account_action
            for index, issue in unresolved
        ):
            raise TriageServiceError("The available evidence does not permit an automatic reply.")
        try:
            response = await self.client.responses.create(
                model=self.model,
                instructions=model_instructions(reply_instructions(), handbook),
                input=[{"role": "user", "content": evidence_payload(
                    ticket=ticket.model_dump(mode="json"),
                    analysis=analysis.model_dump(mode="json"),
                    decision=decision.model_dump(mode="json"),
                    context=context.model_dump(mode="json"),
                )}],
                store=False,
            )
        except OpenAIError as exc:
            raise TriageServiceError("OpenAI reply generation failed; human review is required.") from exc
        _check_response(response)
        reply = response.output_text.strip()
        if not reply:
            raise TriageServiceError("OpenAI returned an empty customer reply.")
        return reply

    async def draft_handoff_reply(
        self, ticket: TicketInput, analysis: TicketAnalysis, decision: TriageDecision,
        context: ToolContext, receipt: HandoffReceipt | None, *, handbook: str,
    ) -> str:
        """Acknowledge unresolved concerns, adding truthful queue status in code."""
        verified_status = fallback_handoff_reply(analysis.language, decision, receipt)
        try:
            response = await self.client.responses.create(
                model=self.model,
                instructions=model_instructions(handoff_reply_instructions(), handbook),
                input=[{"role": "user", "content": evidence_payload(
                    ticket=ticket.model_dump(mode="json"),
                    analysis=analysis.model_dump(mode="json"),
                    decision=decision.model_dump(mode="json"),
                    context=context.model_dump(mode="json"),
                    dispatch_receipt=receipt.model_dump(mode="json") if receipt else None,
                )}],
                store=False,
            )
        except OpenAIError as exc:
            raise TriageServiceError("OpenAI handoff acknowledgement failed.") from exc
        _check_response(response)
        acknowledgement = response.output_text.strip()
        if not acknowledgement:
            raise TriageServiceError("OpenAI returned an empty handoff acknowledgement.")
        return f"{verified_status}\n\n{acknowledgement}"
