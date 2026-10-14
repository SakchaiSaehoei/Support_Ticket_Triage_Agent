"""A numbered, persistent terminal interface around the existing ticket agent."""

import json
from typing import Protocol
from uuid import uuid4
from zoneinfo import ZoneInfo

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from support_ticket_triage.database import DEFAULT_FIXTURE_DIRECTORY, SQLiteStore, stable_message_id
from support_ticket_triage.schemas import CustomerProfile, TicketInput, TicketMessage, TicketResult, TicketUpdate


BANGKOK = ZoneInfo("Asia/Bangkok")
URGENCY_STYLES = {"critical": "bold red", "high": "yellow", "medium": "cyan", "low": "green"}


class Runner(Protocol):
    async def run(self, update: TicketUpdate) -> TicketResult: ...
    async def resume(self, ticket_id: str, customer_id: str) -> TicketResult: ...
    async def pending(self, ticket_id: str, customer_id: str) -> bool: ...


class _ExitRequested(Exception):
    pass


class InteractiveCLI:
    """Keep navigation offline; persist each input before asking the runner to act."""

    def __init__(self, store: SQLiteStore, runner: Runner, *, console: Console | None = None) -> None:
        self.store = store
        self.runner = runner
        self.console = console or Console(highlight=False)
        path = DEFAULT_FIXTURE_DIRECTORY / "sample_tickets.json"
        self.samples = {
            ticket.ticket_id: ticket for ticket in (
                TicketInput.model_validate(item) for item in json.loads(path.read_text(encoding="utf-8"))
            )
        }
        self.sample_message_ids = {
            message.message_id or stable_message_id(ticket.ticket_id, index, message)
            for ticket in self.samples.values()
            for index, message in enumerate(ticket.messages, start=1)
        }

    def _heading(self, title: str, context: str | None = None) -> None:
        self.console.print()
        self.console.rule(Text(title, style="bold cyan"), style="cyan")
        if context:
            self.console.print(Text(context, style="dim"))

    def _menu(self, options: list[tuple[str, str]], *, default: str | None = None) -> str:
        for key, label in options:
            suffix = " [Enter]" if key == default else ""
            self.console.print(Text.assemble((f"  {key:>2}  ", "bold cyan"), (label + suffix, "")))
        allowed = {key.upper() for key, _ in options}
        while True:
            choice = self.console.input(Text("\nChoice > ", style="bold cyan")).strip().upper()
            if not choice and default:
                choice = default
            if choice in allowed:
                if choice == "0":
                    raise _ExitRequested
                return choice
            self.console.print(Text("Choose one of the listed numbers or letters.", style="red"))

    @staticmethod
    def _label(customer: CustomerProfile) -> str:
        if customer.display_name:
            return customer.display_name
        sample = {"customer-001": "Sample customer 1", "customer-002": "Sample customer 2",
                  "customer-003": "Sample customer 3"}
        return sample.get(customer.customer_id, customer.customer_id)

    def _profile(self, customer: CustomerProfile) -> None:
        grid = Table.grid(padding=(0, 2), expand=False)
        grid.add_column(style="dim")
        grid.add_column()
        values = (
            ("Customer", self._label(customer)), ("Plan", customer.plan),
            ("Customer for", f"{customer.tenure_months} months" if customer.tenure_months is not None else "Unknown"),
            ("Region", customer.region or "Unknown"),
            ("Seats", str(customer.seats) if customer.seats is not None else "Unknown"),
            ("Previous support tickets", str(customer.prior_support_tickets)
             if customer.prior_support_tickets is not None else "Unknown"),
        )
        for key, value in values:
            grid.add_row(Text(key), Text(value))
        self.console.print(grid)
        for note in customer.notes:
            self.console.print(Text("Note: " + note, style="dim"))

    def _optional_integer(self, prompt: str, *, minimum: int = 0) -> int | None:
        while True:
            value = self.console.input(Text(f"{prompt} [Enter = unknown] > ", style="cyan")).strip()
            if not value:
                return None
            try:
                number = int(value)
                if number >= minimum:
                    return number
            except ValueError:
                pass
            self.console.print(Text(f"Enter a whole number of at least {minimum}, or press Enter to skip.", style="red"))

    def _wizard(self) -> CustomerProfile | None:
        self._heading("Add customer · Step 1 of 2", "Choose a plan. B goes back; 0 exits.")
        plan = self._menu([("1", "Free"), ("2", "Pro"), ("3", "Enterprise"), ("B", "Back"), ("0", "Exit")])
        if plan == "B":
            return None
        self._heading("Add customer · Step 2 of 2", "You can leave every additional detail unknown.")
        details = self._menu([("1", "Skip and save"), ("2", "Add account details"), ("B", "Cancel customer setup"), ("0", "Exit")], default="1")
        if details == "B":
            return None
        fields = {}
        if details == "2":
            fields["display_name"] = self.console.input(Text("Customer label [Enter = automatic] > ", style="cyan")).strip() or None
            fields["tenure_months"] = self._optional_integer("Months as a customer")
            self.console.print(Text("Region", style="bold"))
            region = self._menu([("1", "Thailand"), ("2", "Other region"), ("3", "Unknown")], default="3")
            if region == "1":
                fields["region"] = "Thailand"
            elif region == "2":
                fields["region"] = self.console.input(Text("Region name > ", style="cyan")).strip() or None
            fields["seats"] = self._optional_integer("Number of seats", minimum=1)
            fields["prior_support_tickets"] = self._optional_integer("Previous support-ticket count")
            note = self.console.input(Text("Account note [Enter = skip] > ", style="cyan")).strip()
            fields["notes"] = [note] if note else []
        customer = self.store.create_customer(plan={"1": "Free", "2": "Pro", "3": "Enterprise"}[plan], **fields)
        self.console.print(Text(f"Customer saved: {self._label(customer)} · {customer.plan}", style="green"))
        return customer

    def _read_message(self) -> TicketMessage | None:
        self.console.print(Text("Enter your message. Enter cancels; /paste accepts multiple lines.", style="dim"))
        body = self.console.input(Text("Message > ", style="bold cyan")).strip()
        if body == "/paste":
            self.console.print(Text("Paste the message, then enter a single . on its own line.", style="dim"))
            lines = []
            while (line := self.console.input(Text("… ", style="cyan"))) != ".":
                lines.append(line)
            body = "\n".join(lines).strip()
        if not body or body == "/back":
            self.console.print(Text("Message cancelled.", style="dim"))
            return None
        return TicketMessage(text=body, message_id="message-" + uuid4().hex)

    def _history(self, ticket: TicketInput, customer: CustomerProfile) -> None:
        self.console.print(Text("Conversation history · times shown in Bangkok", style="bold"))
        for message in ticket.messages:
            author = self._label(customer) if message.role == "customer" else "Support"
            if message.message_id in self.sample_message_ids:
                when = f"Sample timing: {message.time_label}"
            elif message.observed_at is not None:
                when = message.observed_at.astimezone(BANGKOK).strftime("%Y-%m-%d %H:%M:%S")
                when += f" · {message.time_label}"
            else:
                when = f"{message.time_label} · observation time unknown"
            self.console.print(Text.assemble((f"\n{message.role.title()} · {author}", "bold cyan" if message.role == "customer" else "bold green"),
                                             (f"  {when}", "dim")))
            self.console.print(Text(message.text))

    def _result(self, result: TicketResult) -> None:
        self.console.print()
        self.console.print(Text("Last saved triage", style="bold magenta"))
        urgency = result.decision.urgency.value if result.decision.urgency else "Unknown"
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="dim")
        grid.add_column()
        for label, value in (
            ("Urgency", Text(urgency, style=URGENCY_STYLES.get(urgency, "dim"))),
            ("Next action", Text(result.decision.action.value.replace("_", " "))),
            ("Owner", Text(result.decision.owner.value.replace("_", " "))),
            ("Reason", Text(result.decision.reason)),
        ):
            grid.add_row(Text(label), value)
        self.console.print(grid)
        if result.reply:
            self.console.print(Panel(Text(result.reply), title="Reply draft", border_style="green"))
        if result.errors:
            self.console.print(Text("Processing notes", style="bold yellow"))
            for error in result.errors:
                self.console.print(Text("• " + error, style="yellow"))

    async def _process(self, update: TicketUpdate) -> None:
        self.console.print(Text("Message saved. Processing triage…", style="cyan"))
        try:
            with self.console.status("Reading history and processing triage…", spinner="dots", spinner_style="cyan"):
                await self.runner.run(update)
        except Exception as exc:
            self.console.print(Text(f"Processing failed: {exc}", style="red"))
            self.console.print(Text("Your message is saved. Resume an interrupted run or choose T to retry the latest message.", style="yellow"))

    def _pending_sample(self, ticket_id: str, customer_id: str) -> TicketMessage | None:
        sample = self.samples.get(ticket_id)
        if sample is None or sample.customer_id != customer_id:
            return None
        message = sample.messages[-1].model_copy(update={
            "message_id": sample.messages[-1].message_id or stable_message_id(
                ticket_id, len(sample.messages), sample.messages[-1]
            ),
        })
        return None if self.store.contains_message(ticket_id, message.message_id, customer_id) else message

    async def _ticket(self, customer: CustomerProfile, ticket_id: str) -> str:
        while True:
            summary = self.store.get_ticket_summary(ticket_id, customer.customer_id)
            self._heading(summary.title, f"Customers › {self._label(customer)} › {ticket_id} · {summary.status.title()}")
            ticket = self.store.load_history(ticket_id, customer.customer_id) if summary.message_count else None
            if ticket:
                self._history(ticket, customer)
            else:
                self.console.print(Text("No messages yet.", style="dim"))
            result = self.store.latest_result(ticket_id, customer.customer_id)
            if result:
                self._result(result)
            try:
                pending = await self.runner.pending(ticket_id, customer.customer_id)
            except Exception as exc:
                pending = False
                self.console.print(Text(f"Could not inspect saved progress: {exc}", style="yellow"))
            sample = self._pending_sample(ticket_id, customer.customer_id)
            options = [("1", "Write a follow-up" + (" (starts a fresh triage pass)" if pending else ""))]
            if sample:
                self.console.print(Text("\nA supplied homework message is ready to process:", style="yellow"))
                self.console.print(Text(sample.text))
                options.append(("2", "Process the supplied latest sample message"))
            if pending:
                self.console.print(Text("\nSaved processing is unfinished. Resume it explicitly with R.", style="yellow"))
                options.append(("R", "Resume interrupted processing"))
            if ticket:
                options.append(("T", "Reprocess the latest customer message (keeps its ID)"))
            options += [("B", "Back to tickets"), ("C", "Choose another customer"), ("0", "Exit")]
            choice = self._menu(options)
            if choice in {"B", "C"}:
                return choice
            if choice == "R":
                try:
                    with self.console.status("Resuming saved processing…", spinner_style="cyan"):
                        await self.runner.resume(ticket_id, customer.customer_id)
                except Exception as exc:
                    self.console.print(Text(f"Resume failed: {exc}", style="red"))
                continue
            if choice == "T":
                message = next((message for message in reversed(ticket.messages) if message.role == "customer"), None)
                if message is None:
                    self.console.print(Text("There is no customer message to reprocess.", style="yellow"))
                    continue
            else:
                message = sample if choice == "2" else self._read_message()
                if message is None:
                    continue
                message = self.store.append_message(ticket_id, customer.customer_id, message)
            await self._process(TicketUpdate(ticket_id=ticket_id, customer_id=customer.customer_id, message=message))

    async def _tickets(self, customer: CustomerProfile) -> None:
        while True:
            self._heading("Customer and tickets", f"Customers › {self._label(customer)}")
            self._profile(customer)
            tickets = self.store.list_tickets(customer.customer_id)
            self.console.print(Text("\nTickets", style="bold"))
            options = [(str(index), f"{ticket.title} · {ticket.status.title()} · {ticket.message_count} messages")
                       for index, ticket in enumerate(tickets, start=1)]
            if not tickets:
                self.console.print(Text("No tickets yet. Create one with N.", style="dim"))
            options += [("N", "Create a ticket"), ("B", "Choose another customer"), ("0", "Exit")]
            choice = self._menu(options)
            if choice == "B":
                return
            if choice == "N":
                self._heading("New ticket", self._label(customer))
                message = self._read_message()
                if message is None:
                    continue
                ticket = self.store.create_ticket(customer.customer_id, message)
                self.console.print(Text(f"Ticket created: {ticket.ticket_id}", style="green"))
                await self._process(TicketUpdate(ticket_id=ticket.ticket_id, customer_id=customer.customer_id,
                                                 message=ticket.messages[0]))
                ticket_id = ticket.ticket_id
            else:
                ticket_id = tickets[int(choice) - 1].ticket_id
            if await self._ticket(customer, ticket_id) == "C":
                return

    async def run(self) -> None:
        try:
            while True:
                self._heading("SUPPORT · Ticket triage", "Choose a customer. Saved conversations stay available between sessions.")
                customers = self.store.list_customers()
                options = [(str(index), f"{self._label(customer)} · {customer.plan}" +
                            (f" · {customer.region}" if customer.region else ""))
                           for index, customer in enumerate(customers, start=1)]
                options += [("N", "Add a customer"), ("0", "Exit")]
                choice = self._menu(options)
                customer = self._wizard() if choice == "N" else customers[int(choice) - 1]
                if customer:
                    await self._tickets(customer)
        except (_ExitRequested, EOFError, KeyboardInterrupt):
            self.console.print(Text("\nGoodbye. Saved customers and conversations are available next time.", style="green"))
