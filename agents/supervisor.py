"""
supervisor.py — reads the request and writes an execution plan.

Plan-and-execute pattern: one LLM call decides which specialists run,
in what order, with what instructions, and whether a specialist should
work on the previous specialist's output (chaining). The graph then
executes the plan deterministically — fewer LLM calls, no routing loops.
"""
from typing import Literal

from pydantic import BaseModel, Field

from agents.common import today
from core.agents_meta import AGENT_META, AGENT_NAMES, OWN_DOCS_NOTE
from core.config import MAX_PLAN_STEPS
from core.llm import structured_call
from core.state import AgentState
from memory.vector_store import get_store

AgentName = Literal[tuple(AGENT_NAMES)]  # type: ignore[valid-type]


class PlanStep(BaseModel):
    agent: AgentName
    task: str = Field(description="specific instruction for that agent")
    use_previous_output: bool = Field(
        default=False,
        description="true if this agent should work on the text produced by the previous step")


class Plan(BaseModel):
    reasoning: str = Field(description="one or two sentences on why this plan")
    steps: list[PlanStep] = Field(min_length=1)


def _roster() -> str:
    return "\n".join(f"- {name}: {meta['description']}" for name, meta in AGENT_META.items())


def plan(state: AgentState) -> dict:
    request = state.get("request", "")
    attachment = (state.get("attachment") or "").strip()

    # Manual override from the UI: skip planning
    forced = state.get("forced_agent")
    if forced in AGENT_META:
        steps = [{"agent": forced, "task": request, "use_previous_output": False}]
        return {"plan": steps, "plan_reasoning": f"Routed directly to {AGENT_META[forced]['label']}."}

    store = get_store()
    memory_note = ("The Second Brain contains: " + ", ".join(store.stats()["sources"][:15])
                   if not store.is_empty else "The Second Brain is empty.")
    attach_note = (f"The user attached text ({len(attachment)} chars). It begins:\n"
                   f"{attachment[:1500]}" if attachment else "No attachment.")

    out = structured_call(
        Plan,
        f"Today is {today()}. You are the supervisor of a team of AI specialists at a "
        "consulting firm. Choose the smallest set of specialists that fully answers "
        f"the request, in execution order, at most {MAX_PLAN_STEPS} steps, each "
        "specialist at most once. Give each a precise task. Set use_previous_output "
        "when a step should process an earlier step's output (e.g. fact-check a "
        "rewritten proposal). Write tasks from the user's point of view: the "
        "firm in the team memory is the user's own firm. " + OWN_DOCS_NOTE
        + "\n\nSpecialists:\n" + _roster(),
        f"Request: {request}\n\n{attach_note}\n\n{memory_note}",
        temperature=0.0,
    )

    steps, seen = [], set()
    for step in out.steps:
        if step.agent not in seen:
            seen.add(step.agent)
            steps.append(step.model_dump())
    if steps:
        steps[0]["use_previous_output"] = False  # nothing before the first step
    return {"plan": steps[:MAX_PLAN_STEPS], "plan_reasoning": out.reasoning}