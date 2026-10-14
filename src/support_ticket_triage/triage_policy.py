"""Deterministic aggregation and auto-response safeguards; no I/O or storage."""

from support_ticket_triage.schemas import (
    IssueDecision, NextAction, SpecialistTeam, TicketAnalysis, ToolContext,
    TriageDecision, Urgency,
)

URGENCY_ORDER = {
    Urgency.LOW: 0, Urgency.MEDIUM: 1, Urgency.HIGH: 2, Urgency.CRITICAL: 3,
}
ACTION_ORDER = {
    NextAction.AUTO_RESPOND: 0,
    NextAction.ROUTE_TO_SPECIALIST: 1,
    NextAction.ESCALATE_TO_HUMAN: 2,
}


class TriageServiceError(RuntimeError):
    """A model call could not produce trustworthy application output."""


def has_issue_evidence(context: ToolContext, issue_index: int) -> bool:
    """Require traceable search results for this issue; another issue cannot supply them."""
    article_ids = context.article_ids_by_issue.get(str(issue_index), [])
    available = {article.article_id for article in context.articles}
    return bool(article_ids) and all(article_id in available for article_id in article_ids)


def aggregate_decisions(
    analysis: TicketAnalysis,
    context: ToolContext,
    decisions: list[IssueDecision],
    action_confidence: float | None,
    *,
    minimum_confidence: float = 0.75,
) -> TriageDecision:
    """Preserve impact classification while preventing unsupported automation."""
    unresolved = {i for i, issue in enumerate(analysis.issues) if not issue.resolved}
    if {item.issue_index for item in decisions} != unresolved or len(decisions) != len(unresolved):
        raise TriageServiceError("Decisions must cover each unresolved issue exactly once.")
    if not decisions:
        return TriageDecision(
            urgency=Urgency.LOW,
            action=NextAction.ESCALATE_TO_HUMAN if context.tool_errors else NextAction.AUTO_RESPOND,
            owner=SpecialistTeam.GENERAL_SUPPORT,
            reason="No unresolved issue remains in the supplied evidence.",
        )

    reasons = []
    guarded = []
    for item in decisions:
        action = item.action
        issue = analysis.issues[item.issue_index]
        if context.tool_errors:
            action = NextAction.ESCALATE_TO_HUMAN
            reasons.append("Evidence lookup failed; human review is required.")
        elif action_confidence is None or action_confidence < minimum_confidence:
            action = NextAction.ESCALATE_TO_HUMAN
            reasons.append("Action confidence requires human review.")
        elif item.urgency == Urgency.CRITICAL:
            action = NextAction.ESCALATE_TO_HUMAN
            reasons.append("Critical impact requires prompt human attention.")
        elif action == NextAction.AUTO_RESPOND and (
            not has_issue_evidence(context, item.issue_index) or issue.requires_account_action
        ):
            action = NextAction.ROUTE_TO_SPECIALIST
            reasons.append("Applicable guidance is unavailable or an authorized account action is needed.")
        guarded.append(item.model_copy(update={"action": action}))

    urgency = max((item.urgency for item in guarded), key=URGENCY_ORDER.__getitem__)
    action = max((item.action for item in guarded), key=ACTION_ORDER.__getitem__)
    # A handoff owner belongs to the issue requiring that action; break ties by impact.
    primary = max(guarded, key=lambda item: (
        ACTION_ORDER[item.action], URGENCY_ORDER[item.urgency], -item.issue_index,
    ))
    summary = f"Highest unresolved urgency: {urgency.value}; recommended action: {action.value}."
    return TriageDecision(
        urgency=urgency, action=action, owner=primary.owner,
        issue_decisions=guarded, confidence=action_confidence,
        reason=" ".join([summary, *dict.fromkeys(reasons)]),
    )
