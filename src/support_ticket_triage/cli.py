"""Run triage from a full transcript or a latest message backed by SQLite history."""

import argparse
import asyncio
import json
import os
import sqlite3
import sys
from contextlib import AsyncExitStack
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import ValidationError
from langgraph.checkpoint.base import BaseCheckpointSaver
from rich.console import Console

from support_ticket_triage.checkpoints import checkpoint_path_for, open_checkpointer
from support_ticket_triage.config import REPOSITORY_ROOT, Setting, repository_path
from support_ticket_triage.database import DEFAULT_FIXTURE_DIRECTORY, SQLiteStore, stable_message_id
from support_ticket_triage.decision_service import DecisionService
from support_ticket_triage.graph import TicketTriageAgent
from support_ticket_triage.interactive import InteractiveCLI
from support_ticket_triage.observability import flush_traces, graph_tracing
from support_ticket_triage.response_service import ResponseService
from support_ticket_triage.schemas import TicketInput, TicketMessage, TicketResult, TicketUpdate


def load_ticket(path: Path) -> TicketInput | TicketUpdate:
    """Read a complete transcript (messages) or a single incoming turn (message)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or ("message" in data and "messages" in data):
        raise ValueError("Use one ticket object with either 'message' or 'messages'.")
    model = TicketUpdate if "message" in data else TicketInput
    return model.model_validate(data)


def load_sample_tickets(path: Path) -> list[TicketInput]:
    """Read an array of tickets while retaining their message order."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("The sample ticket file must contain a JSON array.")
    return [TicketInput.model_validate(item) for item in data]


def configured_agent(
    settings: Setting, client: AsyncOpenAI, store: SQLiteStore,
    *, checkpointer: BaseCheckpointSaver | None = None,
) -> TicketTriageAgent:
    return TicketTriageAgent(
        DecisionService(settings, client), ResponseService(settings, client),
        store=store, checkpointer=checkpointer,
    )


def openai_client(settings: Setting) -> AsyncOpenAI:
    client = AsyncOpenAI(api_key=settings.apikey, timeout=60, max_retries=1)
    if settings.langsmith_tracing_status:
        from langsmith.wrappers import wrap_openai

        client = wrap_openai(client)
    return client


async def run_ticket(
    ticket: TicketInput | TicketUpdate, *, database_path: Path | None = None
) -> TicketResult:
    """Connect one shared OpenAI client to both model services and the history store."""
    settings = Setting()
    store = SQLiteStore(database_path or settings.database_path)
    try:
        async with (
            openai_client(settings) as client,
            open_checkpointer(store.path, settings.checkpoint_path) as checkpointer,
        ):
            return await configured_agent(
                settings, client, store, checkpointer=checkpointer
            ).run(ticket)
    finally:
        await asyncio.to_thread(flush_traces)


async def _run_tickets(tickets: list[TicketInput | TicketUpdate], store: SQLiteStore) -> None:
    settings = Setting()
    try:
        async with (
            openai_client(settings) as client,
            open_checkpointer(store.path, settings.checkpoint_path) as checkpointer,
        ):
            agent = configured_agent(settings, client, store, checkpointer=checkpointer)
            for ticket in tickets:
                result = await agent.run(ticket)
                print(result.model_dump_json(indent=2))
    finally:
        await asyncio.to_thread(flush_traces)


class InteractiveRunner:
    """Acquire provider resources only when the operator requests triage or resume."""

    def __init__(self, store: SQLiteStore) -> None:
        self.store = store
        self.agent: TicketTriageAgent | None = None
        self.stack: AsyncExitStack | None = None
        override = os.getenv("SUPPORT_CHECKPOINT_PATH", "").strip()
        self.checkpoint_path = repository_path(override) if override else None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        if self.stack is not None:
            try:
                await self.stack.aclose()
            finally:
                await asyncio.to_thread(flush_traces)

    async def _get_agent(self) -> TicketTriageAgent:
        if self.agent is not None:
            return self.agent
        settings = Setting()
        stack = AsyncExitStack()
        try:
            client = await stack.enter_async_context(openai_client(settings))
            saver = await stack.enter_async_context(open_checkpointer(self.store.path, self.checkpoint_path))
            agent = configured_agent(settings, client, self.store, checkpointer=saver)
        except BaseException:
            await stack.aclose()
            raise
        self.stack, self.agent = stack, agent
        return agent

    @staticmethod
    def _verify_snapshot(snapshot, ticket_id: str, customer_id: str) -> None:
        previous = snapshot.values.get("incoming") or snapshot.values.get("ticket")
        if previous is not None and (previous.ticket_id, previous.customer_id) != (ticket_id, customer_id):
            raise ValueError("Saved progress belongs to a different ticket or customer.")

    async def pending(self, ticket_id: str, customer_id: str) -> bool:
        self.store.validate_ticket_owner(ticket_id, customer_id)
        config = {"configurable": {"thread_id": ticket_id}}
        if self.agent is not None:
            snapshot = await self.agent.graph.aget_state(config)
        else:
            path = checkpoint_path_for(self.store.path, self.checkpoint_path)
            if not path.is_file():
                return False
            async with open_checkpointer(self.store.path, self.checkpoint_path) as saver:
                # Building an inspection graph does not execute any nodes or providers.
                inspection = TicketTriageAgent(None, None, mcp_client=object(), store=self.store, checkpointer=saver)
                snapshot = await inspection.graph.aget_state(config)
        self._verify_snapshot(snapshot, ticket_id, customer_id)
        return bool(snapshot.next)

    async def run(self, update: TicketUpdate) -> TicketResult:
        agent = await self._get_agent()
        return await agent.run(update)

    async def resume(self, ticket_id: str, customer_id: str) -> TicketResult:
        self.store.validate_ticket_owner(ticket_id, customer_id)
        if not await self.pending(ticket_id, customer_id):
            raise ValueError("This ticket has no interrupted processing to resume.")
        agent = await self._get_agent()
        with graph_tracing(ticket_id) as trace_config:
            config = {**trace_config, "configurable": {
                **trace_config.get("configurable", {}), "thread_id": ticket_id,
            }}
            snapshot = await agent.graph.aget_state(config)
            self._verify_snapshot(snapshot, ticket_id, customer_id)
            state = await agent.graph.ainvoke(None, config=config)
        return state["result"]


async def run_interactive(store: SQLiteStore, *, no_color: bool = False) -> None:
    store.seed_samples(
        DEFAULT_FIXTURE_DIRECTORY / "customer_accounts.json",
        DEFAULT_FIXTURE_DIRECTORY / "sample_tickets.json",
    )
    async with InteractiveRunner(store) as runner:
        await InteractiveCLI(store, runner, console=Console(highlight=False, no_color=no_color or None)).run()


def sample_updates(tickets: list[TicketInput]) -> list[TicketUpdate]:
    """Replay only the last fixture turn; repeat sample runs reuse its message ID."""
    return [TicketUpdate(
        ticket_id=ticket.ticket_id, customer_id=ticket.customer_id,
        message=ticket.messages[-1].model_copy(update={
            "message_id": ticket.messages[-1].message_id or stable_message_id(
                ticket.ticket_id, len(ticket.messages), ticket.messages[-1]
            ),
        }),
    ) for ticket in tickets]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Support Ticket Triage Agent",
        epilog="Run without arguments in a terminal for the interactive application. Existing flags print JSON.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--interactive", action="store_true", help="Open the customer and ticket application")
    source.add_argument("--seed-samples", action="store_true", help="Seed prior sample history without API calls")
    source.add_argument("--ticket", type=Path, help="Path to one ticket JSON object")
    source.add_argument("--message", help="Latest message text; requires ticket and customer IDs")
    source.add_argument(
        "--samples", type=Path, nargs="?",
        const=DEFAULT_FIXTURE_DIRECTORY / "sample_tickets.json",
        help="Run sample tickets from a JSON array; defaults to the supplied fixtures",
    )
    parser.add_argument("--ticket-id", help="Ticket ID for --message")
    parser.add_argument("--customer-id", help="Customer ID for --message")
    parser.add_argument("--message-id", help="Stable incoming ID to make retries idempotent")
    parser.add_argument("--time-label", default="just now", help="Observed timing for --message")
    parser.add_argument("--database", type=Path, help="Override the SQLite database path")
    parser.add_argument("--no-color", action="store_true", help="Disable terminal colors")
    args = parser.parse_args()
    if args.message is not None and (not args.ticket_id or not args.customer_id):
        parser.error("--message requires --ticket-id and --customer-id.")
    if args.message is None and any((args.ticket_id, args.customer_id, args.message_id)):
        parser.error("Ticket/customer/message IDs are used with --message; JSON input carries its own IDs.")
    interactive = args.interactive or not any((args.seed_samples, args.ticket is not None,
                                               args.message is not None, args.samples is not None))
    if interactive and not sys.stdin.isatty():
        parser.error("The interactive application requires a terminal. Use --ticket, --message, or --samples for scripted input.")
    try:
        load_dotenv(REPOSITORY_ROOT / ".env")
        database_path = repository_path(
            args.database or os.getenv("SUPPORT_DATABASE_PATH", ".state/support.sqlite3")
        )
        store = SQLiteStore(database_path)
        if interactive:
            asyncio.run(run_interactive(store, no_color=args.no_color))
            return
        if args.seed_samples:
            counts = store.seed_samples(
                DEFAULT_FIXTURE_DIRECTORY / "customer_accounts.json",
                DEFAULT_FIXTURE_DIRECTORY / "sample_tickets.json",
            )
            print(json.dumps(counts))
            return
        if args.message is not None:
            tickets = [TicketUpdate(
                ticket_id=args.ticket_id, customer_id=args.customer_id,
                message=TicketMessage(text=args.message, time_label=args.time_label, message_id=args.message_id),
            )]
        elif args.ticket is not None:
            tickets = [load_ticket(args.ticket)]
        else:
            samples = load_sample_tickets(args.samples)
            store.seed_samples(DEFAULT_FIXTURE_DIRECTORY / "customer_accounts.json", args.samples)
            tickets = sample_updates(samples)
        asyncio.run(_run_tickets(tickets, store))
    except (OSError, KeyError, ValueError, ValidationError, sqlite3.Error) as exc:
        parser.exit(2, f"{exc}\n")
    except KeyboardInterrupt:
        Console(no_color=args.no_color or None).print("\nGoodbye. Your saved conversations are available next time.", style="green")


if __name__ == "__main__":
    main()
