"""Expose the lookup methods as local stdio MCP tools."""

from fastmcp import FastMCP

from support_ticket_triage.schemas import CustomerProfile, KnowledgeBaseArticle
from support_ticket_triage.retrieval import configured_retriever
from support_ticket_triage.tools import SupportTools

mcp = FastMCP("Support Ticket Triage Tools")
support_tools = SupportTools()


@mcp.tool()
def check_customer_history(customer_id: str) -> CustomerProfile | None:
    """Retrieve plan, tenure, region, seats, and known support history by ID."""
    return support_tools.check_customer_history(customer_id)


@mcp.tool()
async def lookup_knowledge_base(query: str) -> list[KnowledgeBaseArticle]:
    """Retrieve internal diagnostic playbooks relevant to a support question."""
    if support_tools.retriever is None:
        support_tools.retriever = configured_retriever(
            source_path=support_tools.knowledge_base_path
        )
    return await support_tools.lookup_knowledge_base(query)


@mcp.tool()
def check_system_status(region: str | None = None) -> dict:
    """Read mock availability observations cached for five minutes; this is not live status."""
    return support_tools.check_system_status(region)


if __name__ == "__main__":
    # Knowledge lookup creates its embedding client lazily and reads saved Chroma.
    # Tool discovery does not load credentials or make an embedding request.
    # Keep stdout for MCP protocol messages; send diagnostics to stderr.
    mcp.run(transport="stdio", show_banner=False, log_level="WARNING")
