"""History and tool orchestration; model services own the triage policy."""

import asyncio
import os
import sys
from datetime import datetime, timezone
from typing import Any, Literal, TypedDict
from uuid import uuid4

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from support_ticket_triage.decision_service import DecisionService
from support_ticket_triage.database import SQLiteStore
from support_ticket_triage.handoffs import MockHandoffService
from support_ticket_triage.observability import graph_tracing
from support_ticket_triage.response_service import ResponseService, fallback_handoff_reply
from support_ticket_triage.retrieval import load_handbook
from support_ticket_triage.schemas import (
    NextAction,
    CustomerProfile,
    ExtractedIssue,
    HandoffReceipt,
    KnowledgeBaseArticle,
    SpecialistTeam,
    TicketAnalysis,
    TicketInput,
    TicketResult,
    TicketUpdate,
    ToolContext,
    TriageDecision,
    Urgency,
)


class TriageState(TypedDict, total=False):
    incoming: TicketInput | TicketUpdate
    ticket: TicketInput | None
    handbook: str
    context: ToolContext | None
    analysis: TicketAnalysis | None
    decision: TriageDecision | None
    reply: str | None
    result: TicketResult | None
    errors: list[str]
    analysis_failed: bool
    reply_failed: bool
    handoff: dict | None
    dispatch_receipt: HandoffReceipt | None


def _fallback_language(ticket: TicketInput) -> str:
    """Keep failure acknowledgments usable for the English/Thai assignment inputs."""
    latest_customer_text = next((
        message.text for message in reversed(ticket.messages) if message.role == "customer"
    ), "")
    return "th" if any("\u0e00" <= char <= "\u0e7f" for char in latest_customer_text) else "en"


class TicketTriageAgent:
    """Own workflow connections while services own the model-facing logic."""

    def __init__(
        self,
        decision_service: DecisionService,
        response_service: ResponseService,
        mcp_client: MultiServerMCPClient | None = None,
        store: SQLiteStore | None = None,
        handoff_service: MockHandoffService | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> None:
        self.decision_service = decision_service
        self.response_service = response_service
        self.store = store
        self.checkpointer = checkpointer
        self.handoff_service = handoff_service or MockHandoffService(store)
        self.mcp_client = mcp_client or MultiServerMCPClient(
            {
                "support": {
                    "transport": "stdio",
                    "command": sys.executable,
                    "args": ["-m", "support_ticket_triage.mcp_server"],
                    "env": {
                        **os.environ,
                        **({"SUPPORT_DATABASE_PATH": str(store.path)} if store else {}),
                    },
                }
            }
        )
        self.tools: dict[str, BaseTool] = {}
        self.graph = self._build_graph()

    def _build_graph(self) -> CompiledStateGraph:
        builder = StateGraph(TriageState)
        builder.add_node("load_history", self._load_history)
        builder.add_node("gather_context", self._gather_context)
        builder.add_node("extract_information", self._extract_information)
        builder.add_node("retrieve_evidence", self._retrieve_evidence)
        builder.add_node("decide", self._decide)
        builder.add_node("draft_reply", self._draft_reply)
        builder.add_node("dispatch_handoff", self._dispatch_handoff)
        builder.add_node("draft_handoff_reply", self._draft_handoff_reply)
        builder.add_node("finish", self._finish)
        builder.add_edge(START, "load_history")
        builder.add_edge("load_history", "gather_context")
        builder.add_edge("gather_context", "extract_information")
        builder.add_edge("extract_information", "retrieve_evidence")
        builder.add_edge("retrieve_evidence", "decide")
        builder.add_conditional_edges(
            "decide", self._next_step,
            {"draft_reply": "draft_reply", "dispatch_handoff": "dispatch_handoff"},
        )
        builder.add_conditional_edges(
            "draft_reply", self._after_reply,
            {"finish": "finish", "dispatch_handoff": "dispatch_handoff"},
        )
        builder.add_edge("dispatch_handoff", "draft_handoff_reply")
        builder.add_edge("draft_handoff_reply", "finish")
        builder.add_edge("finish", END)
        return builder.compile(checkpointer=self.checkpointer)

    async def load_tools(self) -> None:
        """Discover the tools from the local MCP server."""
        discovered = await self.mcp_client.get_tools()
        self.tools = {tool.name: tool for tool in discovered}

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Invoke a discovered tool from your context-gathering logic."""
        if name not in self.tools:
            raise ValueError(f"MCP tool is not loaded: {name}")
        result = await self.tools[name].ainvoke({
            "type": "tool_call", "name": name, "id": uuid4().hex, "args": arguments,
        })
        if not isinstance(result, ToolMessage):
            return result
        if result.status == "error":
            raise RuntimeError(f"MCP tool failed: {name}.")
        artifact = result.artifact or {}
        if "structured_content" not in artifact:
            raise ValueError(f"MCP tool returned no structured result: {name}.")
        payload = artifact["structured_content"]
        return payload["result"] if isinstance(payload, dict) and "result" in payload else payload

    async def _load_history(self, state: TriageState) -> TriageState:
        """Commit the incoming message and reconstruct the authoritative transcript."""
        incoming = state["incoming"]
        if self.store is None:
            if isinstance(incoming, TicketUpdate):
                raise ValueError("A SQLite store is required for latest-message input.")
            return {"ticket": incoming.model_copy(update={
                "as_of": datetime.now(timezone.utc),
            })}
        if isinstance(incoming, TicketUpdate):
            await asyncio.to_thread(
                self.store.append_message, incoming.ticket_id, incoming.customer_id, incoming.message
            )
        else:
            await asyncio.to_thread(self.store.ingest_ticket, incoming)
        ticket = await asyncio.to_thread(
            self.store.load_history, incoming.ticket_id, incoming.customer_id
        )
        return {"ticket": ticket}

    async def _gather_context(self, state: TriageState) -> TriageState:
        """Load profile context and optional summary memory before extraction."""
        context = ToolContext()
        errors = list(state.get("errors", []))
        if self.store is not None:
            try:
                context.previous_analysis = await asyncio.to_thread(
                    self.store.get_memory, state["ticket"].ticket_id
                )
            except Exception as exc:
                errors.append(
                    f"Previous analysis unavailable ({type(exc).__name__}); "
                    "using the complete conversation."
                )
        try:
            if not self.tools:
                await self.load_tools()
            profile = await self.call_tool(
                "check_customer_history", {"customer_id": state["ticket"].customer_id}
            )
            if profile is None:
                context.tool_errors.append("Customer profile not found; account details are unverified.")
            else:
                context.customer = CustomerProfile.model_validate(profile)
        except Exception as exc:
            context.tool_errors.append(f"Customer history lookup failed ({type(exc).__name__}).")
        return {"context": context, "errors": errors}

    async def _extract_information(self, state: TriageState) -> TriageState:
        try:
            analysis = await self.response_service.extract_information(
                state["ticket"], state["context"], handbook=state["handbook"]
            )
            return {"analysis": analysis}
        except Exception as exc:
            analysis = TicketAnalysis(
                product=None, issue_type="unclassified", customer_sentiment="unknown",
                language=_fallback_language(state["ticket"]),
                issues=[ExtractedIssue(
                    issue_type="unclassified",
                    description="Conversation requires human review because extraction failed.",
                    evidence=[message.text for message in state["ticket"].messages],
                )],
            )
            return {
                "analysis": analysis, "analysis_failed": True,
                "errors": [*state.get("errors", []),
                           f"Information extraction failed ({type(exc).__name__})."],
            }

    async def _retrieve_evidence(self, state: TriageState) -> TriageState:
        """Search each unresolved issue using the normalized query from extraction."""
        if state.get("analysis_failed"):
            return {}
        context = state["context"].model_copy(deep=True)
        issue_queries = {
            str(index): issue.retrieval_query.strip() or issue.description
            for index, issue in enumerate(state["analysis"].issues) if not issue.resolved
        }
        queries = list(dict.fromkeys(issue_queries.values()))
        results = await asyncio.gather(*(
            self.call_tool("lookup_knowledge_base", {"query": query}) for query in queries
        ), return_exceptions=True)
        articles = {}
        matches_by_query = {}
        for query, result in zip(queries, results):
            matches_by_query[query] = []
            try:
                if isinstance(result, BaseException):
                    raise result
                if not isinstance(result, list):
                    raise ValueError("Knowledge lookup did not return an article list.")
                for item in result:
                    article = KnowledgeBaseArticle.model_validate(item)
                    articles[article.article_id] = article
                    matches_by_query[query].append(article.article_id)
            except Exception as exc:
                context.tool_errors.append(f"Knowledge lookup failed ({type(exc).__name__}).")
        context.articles = list(articles.values())
        context.article_ids_by_issue = {
            index: matches_by_query[query] for index, query in issue_queries.items()
        }
        if state["analysis"].needs_system_status:
            try:
                status = await self.call_tool("check_system_status", {
                    "region": context.customer.region if context.customer else None,
                })
                if not isinstance(status, dict):
                    raise ValueError("Status lookup did not return a structured observation.")
                context.system_status = status
            except Exception as exc:
                context.tool_errors.append(f"System status lookup failed ({type(exc).__name__}).")
        return {"context": context}

    @staticmethod
    def _human_review(reason: str, urgency: Urgency | None = None) -> TriageDecision:
        return TriageDecision(
            urgency=urgency, action=NextAction.ESCALATE_TO_HUMAN,
            owner=SpecialistTeam.GENERAL_SUPPORT, reason=reason,
        )

    async def _decide(self, state: TriageState) -> TriageState:
        if state.get("analysis_failed"):
            return {"decision": self._human_review("Information extraction failed; review the full conversation.")}
        try:
            decision = await self.decision_service.decide(
                state["ticket"], state["analysis"], state["context"], handbook=state["handbook"]
            )
            return {"decision": decision}
        except Exception as exc:
            error = f"Triage decision failed ({type(exc).__name__})."
            return {
                "decision": self._human_review(error),
                "errors": [*state.get("errors", []), error],
            }

    def _next_step(self, state: TriageState) -> Literal["draft_reply", "dispatch_handoff"]:
        # The classification policy decides the action. This edge only follows it.
        if state["decision"].action == NextAction.AUTO_RESPOND:
            return "draft_reply"
        return "dispatch_handoff"

    @staticmethod
    def _after_reply(state: TriageState) -> Literal["finish", "dispatch_handoff"]:
        return "finish" if state["decision"].action == NextAction.AUTO_RESPOND else "dispatch_handoff"

    async def _draft_reply(self, state: TriageState) -> TriageState:
        try:
            reply = await self.response_service.draft_reply(
                state["ticket"], state["analysis"], state["decision"], state["context"],
                handbook=state["handbook"],
            )
            return {"reply": reply}
        except Exception as exc:
            error = f"Reply generation failed ({type(exc).__name__})."
            return {
                "decision": state["decision"].model_copy(update={
                    "action": NextAction.ESCALATE_TO_HUMAN,
                    "reason": f"{state['decision'].reason} {error} Human review is required.".strip(),
                }),
                "errors": [*state.get("errors", []), error],
                "reply_failed": True,
            }

    @staticmethod
    def _handoff_payload(state: TriageState) -> dict:
        return {
            "ticket_id": state["ticket"].ticket_id,
            "customer_id": state["ticket"].customer_id,
            "customer_context": (
                state["context"].customer.model_dump(mode="json") if state["context"].customer else None
            ),
            "recommended_owner": state["decision"].owner.value,
            "reason": state["decision"].reason,
            "issue_decisions": [item.model_dump(mode="json") for item in state["decision"].issue_decisions],
            "unresolved_issues": [
                issue.model_dump(mode="json") for issue in state["analysis"].issues if not issue.resolved
            ],
            "observations_and_attempts": [message.model_dump(mode="json") for message in state["ticket"].messages],
            "evidence_sources": [article.source_url for article in state["context"].articles],
            "status_observation": state["context"].system_status,
            "assignment_performed": False,
            "mock_dispatch_performed": False,
            "mock_dispatch": None,
        }

    async def _dispatch_handoff(self, state: TriageState) -> TriageState:
        """Execute a local mock handoff before confirming it to the customer."""
        payload = self._handoff_payload(state)
        try:
            receipt = HandoffReceipt.model_validate(await asyncio.to_thread(
                self.handoff_service.dispatch, state["ticket"], state["decision"], payload,
            ))
            if receipt.action != state["decision"].action or receipt.owner != state["decision"].owner:
                raise ValueError("Mock dispatch receipt does not match the selected handoff.")
        except Exception as exc:
            error = f"Mock handoff dispatch failed ({type(exc).__name__})."
            updated = {**state, "decision": state["decision"].model_copy(update={
                "action": NextAction.ESCALATE_TO_HUMAN,
                "reason": f"{state['decision'].reason} {error} Notification is unconfirmed.".strip(),
            })}
            return {
                "decision": updated["decision"], "handoff": self._handoff_payload(updated),
                "dispatch_receipt": None, "errors": [*state.get("errors", []), error],
            }
        return {
            "dispatch_receipt": receipt,
            "handoff": {**payload, "mock_dispatch_performed": True,
                        "mock_dispatch": receipt.model_dump(mode="json")},
        }

    async def _draft_handoff_reply(self, state: TriageState) -> TriageState:
        """Acknowledge every handoff, including dispatch/model failure paths."""
        receipt = state.get("dispatch_receipt")
        if state.get("analysis_failed") or state.get("reply_failed"):
            return {"reply": fallback_handoff_reply(state["analysis"].language, state["decision"], receipt)}
        try:
            reply = await self.response_service.draft_handoff_reply(
                state["ticket"], state["analysis"], state["decision"], state["context"], receipt,
                handbook=state["handbook"],
            )
            if not reply or not reply.strip():
                raise ValueError("Handoff acknowledgment is empty.")
            return {"reply": reply}
        except Exception as exc:
            return {
                "reply": fallback_handoff_reply(state["analysis"].language, state["decision"], receipt),
                "errors": [*state.get("errors", []),
                           f"Handoff acknowledgment used fallback ({type(exc).__name__})."],
            }

    async def _finish(self, state: TriageState) -> TriageState:
        handoff = state.get("handoff") if state["decision"].action != NextAction.AUTO_RESPOND else None
        result = TicketResult(
            ticket_id=state["ticket"].ticket_id,
            analysis=state["analysis"],
            decision=state["decision"],
            reply=state.get("reply"),
            handoff=handoff,
            errors=[*state.get("errors", []), *state["context"].tool_errors],
        )
        if self.store is not None:
            await asyncio.to_thread(self.store.save_result, result)
        return {"result": result}

    async def run(self, ticket: TicketInput | TicketUpdate) -> TicketResult:
        """Read mandatory policy, then run a full transcript or latest-message update."""
        # Mandatory policy is loaded before tool discovery or model execution.
        handbook = load_handbook()
        if self.store is not None:
            await asyncio.to_thread(
                self.store.validate_ticket_owner, ticket.ticket_id, ticket.customer_id
            )
        with graph_tracing(ticket.ticket_id) as trace_config:
            config = {
                **trace_config,
                "configurable": {**trace_config.get("configurable", {}), "thread_id": ticket.ticket_id},
            }
            if self.checkpointer is not None:
                snapshot = await self.graph.aget_state(config)
                previous = snapshot.values.get("incoming") or snapshot.values.get("ticket")
                if previous is not None and previous.customer_id != ticket.customer_id:
                    raise ValueError("Ticket checkpoint belongs to a different customer.")
            state = await self.graph.ainvoke(
                {
                    "incoming": ticket, "handbook": handbook,
                    "ticket": None, "context": None, "analysis": None, "decision": None,
                    "reply": None, "result": None, "handoff": None, "dispatch_receipt": None,
                    "errors": [], "analysis_failed": False, "reply_failed": False,
                },
                config=config,
            )
        return state["result"]
