"""Trace graph and Decisions calls to one destination; flush the shared client."""

import json
import logging
from contextlib import asynccontextmanager, contextmanager

from langchain_core.tracers.langchain import LangChainTracer
from langsmith import get_tracing_context, trace, tracing_context, utils
from langsmith.address import AgentAddress
from langsmith.run_trees import get_cached_client
from openai import AsyncOpenAI
from openai.types.decision import Decision

logger = logging.getLogger(__name__)


class _AgentTracer(LangChainTracer):
    """Preserve agent addressing when LangChain would inject its default project."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # RunTree resolves LANGSMITH_AGENT_* itself. A project alongside that
        # address is rejected by LangSmith, so _start_trace must not add one.
        # Keep this in __init__ so copied callbacks preserve the same behavior.
        self.project_name = None


def tracing_config(ticket_id: str) -> dict:
    """Attach an explicit callback, preserving agent mode or ordinary project mode."""
    if not utils.tracing_is_enabled():
        return {}
    tracer_type = _AgentTracer if AgentAddress.from_env() is not None else LangChainTracer
    return {
        "callbacks": [tracer_type(client=get_cached_client())],
        "run_name": "SupportTicketTriage",
        "metadata": {"ticket_id": ticket_id},
    }


@contextmanager
def graph_tracing(ticket_id: str):
    """Disable tracing only for this invocation when optional setup fails."""
    try:
        config = tracing_config(ticket_id)
    except Exception as exc:
        logger.warning("LangSmith graph trace did not start (%s).", type(exc).__name__)
        # Empty callbacks alone permit LangChain to add its automatic tracer.
        # The public context override also disables nested SDK/model tracing,
        # then restores the previous context without changing environment vars.
        with tracing_context(enabled=False):
            yield {}
    else:
        yield config


@asynccontextmanager
async def _decision_span(request: dict):
    """Keep trace lifecycle failures from interrupting or repeating model calls."""
    # Restore the parent even if LangSmith's exit fails before resetting context.
    with tracing_context(**get_tracing_context()):
        try:
            manager = trace(
                "OpenAI Decisions", run_type="llm",
                inputs={**request, "messages": [{
                    "role": "user", "content": json.dumps({
                        "input": request["input"], "questions": request["questions"],
                    }),
                }]},
                metadata={"ls_provider": "openai", "ls_model_name": request["model"],
                          "api_endpoint": "/v1/decisions"},
                client=get_cached_client(),
            )
            run = await manager.__aenter__()
        except Exception as exc:
            logger.warning("LangSmith decision trace did not start (%s).", type(exc).__name__)
            yield None
            return
        try:
            yield run
        finally:
            try:
                # The caller records a sanitized error explicitly. Passing the
                # SDK exception here would upload its body and chained traceback.
                await manager.__aexit__(None, None, None)
            except Exception as exc:
                logger.warning("LangSmith decision trace did not finish (%s).", type(exc).__name__)


def _record_decision(run, response: Decision | None = None, error: BaseException | None = None):
    """Record raw output or an error type; telemetry never changes model results."""
    if run is None:
        return
    try:
        if error is not None:
            run.end(error=f"OpenAI decision request failed ({type(error).__name__}).")
        else:
            output = response.model_dump(mode="json")
            usage = output["usage"]
            usage_metadata = {key: usage[key] for key in (
                "input_tokens", "output_tokens", "total_tokens",
            )}
            usage_metadata["input_token_details"] = {
                "cache_read": usage["input_tokens_details"]["cached_tokens"],
                "cache_creation": usage["input_tokens_details"]["cache_write_tokens"],
            }
            usage_metadata["output_token_details"] = {
                "reasoning": usage["output_tokens_details"]["reasoning_tokens"],
            }
            run.metadata["ls_model_name"] = response.model
            run.set(usage_metadata=usage_metadata)
            run.end(outputs={
                **output, "usage_metadata": usage_metadata,
                "messages": [{"role": "assistant", "content": json.dumps({
                    "answers": output["answers"],
                })}],
            })
    except Exception as exc:
        logger.warning("LangSmith decision output was not recorded (%s).", type(exc).__name__)


async def request_decision(
    client: AsyncOpenAI, *, model: str, input: str, questions: list[dict],
) -> Decision:
    """Trace explicit request data and raw choices without serializing the client."""
    request = {"model": model, "input": input, "questions": questions}
    if not utils.tracing_is_enabled():
        return await client.decisions.create(**request)

    async with _decision_span(request) as run:
        try:
            response = await client.decisions.create(**request)
        except BaseException as exc:
            _record_decision(run, error=exc)
            raise
        _record_decision(run, response=response)
    return response


def flush_traces() -> None:
    """Drain pending traces with a bounded wait; disabled tracing creates no client."""
    if utils.tracing_is_enabled():
        try:
            get_cached_client().flush(timeout=10)
        except Exception as exc:
            logger.warning("LangSmith trace upload did not finish (%s).", type(exc).__name__)
