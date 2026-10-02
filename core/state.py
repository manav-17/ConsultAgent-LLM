"""
state.py — the shared state that flows through the LangGraph.

The supervisor writes a plan; each agent reads its task from
plan[cursor], writes its output into results[agent_name] and
moves the cursor forward. The synthesizer reads all results.
"""
from typing import Any, Optional, TypedDict


class AgentState(TypedDict, total=False):
    request:        str                 # what the user asked
    attachment:     str                 # pasted text / uploaded file content
    forced_agent:   Optional[str]       # skip supervisor and run one agent
    plan:           list[dict]          # [{"agent", "task", "use_previous_output"}]
    plan_reasoning: str
    cursor:         int                 # index of the next plan step
    results:        dict[str, Any]      # agent_name -> result dict
    trace:          list[dict]          # timeline events for the UI
    final_report:   str