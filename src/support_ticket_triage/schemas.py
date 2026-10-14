"""Data contracts shared by the tools, triage services, and ticket runner.

These models describe the data. The decision rules belong in your service logic.
"""

from enum import StrEnum
from typing import Literal

from pydantic import AwareDatetime, BaseModel, Field, field_validator


class Urgency(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class NextAction(StrEnum):
    AUTO_RESPOND = "auto_respond"
    ROUTE_TO_SPECIALIST = "route_to_specialist"
    ESCALATE_TO_HUMAN = "escalate_to_human"


class SpecialistTeam(StrEnum):
    BILLING = "billing"
    TECHNICAL_SUPPORT = "technical_support"
    PRODUCT = "product"
    GENERAL_SUPPORT = "general_support"


class TicketMessage(BaseModel):
    text: str = Field(min_length=1)
    time_label: str = "just now"
    message_id: str | None = None
    role: Literal["customer", "support"] = "customer"
    observed_at: AwareDatetime | None = Field(
        default=None,
        description="When the relative time_label was captured; not a verified send time.",
    )

    @field_validator("message_id")
    @classmethod
    def reject_blank_message_id(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("message_id must not be blank.")
        return value


class TicketInput(BaseModel):
    ticket_id: str = Field(min_length=1)
    customer_id: str = Field(min_length=1)
    # Keep messages in their supplied order so earlier issues remain visible.
    messages: list[TicketMessage] = Field(min_length=1)
    as_of: AwareDatetime | None = Field(
        default=None, description="Current processing reference time for this transcript."
    )


class TicketUpdate(BaseModel):
    """An incoming message; previous turns are loaded from SQLite."""

    ticket_id: str = Field(min_length=1)
    customer_id: str = Field(min_length=1)
    message: TicketMessage


class CustomerProfile(BaseModel):
    customer_id: str
    plan: str
    display_name: str | None = None
    tenure_months: int | None = Field(default=None, ge=0)
    region: str | None = None
    seats: int | None = None
    # None means the brief did not provide the customer's total ticket count.
    prior_support_tickets: int | None = Field(default=None, ge=0)
    notes: list[str] = Field(default_factory=list)


class TicketSummary(BaseModel):
    """Persistent ticket metadata for selecting an existing conversation."""

    ticket_id: str
    customer_id: str
    title: str
    status: Literal["open", "pending", "resolved"] = "open"
    created_at: AwareDatetime
    updated_at: AwareDatetime
    message_count: int = Field(ge=0)


class KnowledgeBaseArticle(BaseModel):
    article_id: str
    title: str
    content: str
    source_url: str | None = None
    doc_type: Literal["support_guide", "handbook"] = "support_guide"
    version: str = "1.0"
    audience: Literal["customer", "internal"] = "internal"
    is_mock: bool = False


class ExtractedIssue(BaseModel):
    issue_type: str
    description: str
    resolved: bool = False
    retrieval_query: str = ""
    evidence: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    requires_account_action: bool = False


class TicketAnalysis(BaseModel):
    # A missing product name is represented explicitly rather than invented.
    product: str | None
    issue_type: str
    customer_sentiment: str
    issues: list[ExtractedIssue] = Field(min_length=1)
    language: str = "en"
    needs_system_status: bool = False


class ToolContext(BaseModel):
    customer: CustomerProfile | None = None
    articles: list[KnowledgeBaseArticle] = Field(default_factory=list)
    article_ids_by_issue: dict[str, list[str]] = Field(default_factory=dict)
    tool_errors: list[str] = Field(default_factory=list)
    previous_analysis: TicketAnalysis | None = None
    system_status: dict | None = None


class IssueDecision(BaseModel):
    issue_index: int = Field(ge=0)
    urgency: Urgency
    action: NextAction
    owner: SpecialistTeam


class TriageDecision(BaseModel):
    # Provider failures request human review without inventing customer severity.
    urgency: Urgency | None
    action: NextAction
    owner: SpecialistTeam = SpecialistTeam.GENERAL_SUPPORT
    issue_decisions: list[IssueDecision] = Field(default_factory=list)
    reason: str = ""
    confidence: float | None = Field(
        default=None, ge=0, le=1,
        description="Minimum confidence across unresolved action choices; urgency and owner are separate.",
    )


class HandoffReceipt(BaseModel):
    """Acceptance by a local mock receiver, never proof of external contact."""

    dispatch_id: str
    action: Literal[
        NextAction.ROUTE_TO_SPECIALIST, NextAction.ESCALATE_TO_HUMAN
    ]
    owner: SpecialistTeam
    destination: str
    status: Literal["accepted"] = "accepted"
    is_mock: Literal[True] = True
    follow_up_requested: bool = False
    created_at: AwareDatetime


class TicketResult(BaseModel):
    ticket_id: str
    analysis: TicketAnalysis
    decision: TriageDecision
    reply: str | None = None
    handoff: dict | None = None
    errors: list[str] = Field(default_factory=list)
