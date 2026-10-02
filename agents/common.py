"""
common.py — helpers shared by all agents.

Agents can chain: when the supervisor sets use_previous_output=True,
an agent works on the text produced by the agent before it
(e.g. Proposal Coach rewrites a proposal, then Fact Checker verifies it).
"""
from datetime import date

from core.state import AgentState


def today() -> str:
    return date.today().isoformat()


def current_step(state: AgentState) -> dict:
    plan = state.get("plan") or []
    cursor = state.get("cursor", 0)
    return plan[cursor] if cursor < len(plan) else {}


def previous_text_output(state: AgentState) -> tuple[str, str]:
    """Most recent text_output produced earlier in this run, with its agent name."""
    plan = state.get("plan") or []
    results = state.get("results") or {}
    for step in reversed(plan[: state.get("cursor", 0)]):
        out = results.get(step["agent"], {})
        if out.get("text_output"):
            return out["text_output"], step["agent"]
    return "", ""


def resolve_input_text(state: AgentState) -> tuple[str, str]:
    """
    Decide which text an agent should work on. Priority:
      1. previous agent's output (if the plan asks for it)
      2. the user's attachment
      3. the request itself
    Returns (text, where_it_came_from).
    """
    step = current_step(state)
    if step.get("use_previous_output"):
        text, agent = previous_text_output(state)
        if text:
            return text, f"output of {agent}"
    attachment = (state.get("attachment") or "").strip()
    if attachment:
        return attachment, "attachment"
    return (state.get("request") or "").strip(), "request"


def result(summary_md: str, status: str = "ok", text_output: str = "", **data) -> dict:
    """Standard shape every agent returns."""
    return {"status": status, "summary_md": summary_md, "text_output": text_output, **data}


def needs_input(message: str) -> dict:
    return result(f"**Input needed.** {message}", status="needs_input")
