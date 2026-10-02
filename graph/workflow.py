"""
workflow.py — the top-level multi-agent graph.

                ┌──────────────┐
    START ────► │  supervisor  │  writes the plan
                └──────┬───────┘
                       │  route_next() reads plan[cursor]
     ┌──────────┬──────┼───────────┬─────────────┐
     ▼          ▼      ▼           ▼             ▼
 second_brain  fact  client   competitor     proposal      ← each is its own
     │       checker health      watch          │            compiled subgraph
     └──────────┴──────┼───────────┴─────────────┘
                       │  back to route_next() until the plan is done
                       ▼
                ┌──────────────┐
                │constraint chk│  recommendations vs the firm's own decisions
                └──────┬───────┘
                       ▼
                ┌──────────────┐
                │ synthesizer  │  merges everything into one briefing
                └──────┬───────┘
                       ▼
                      END

Hierarchical multi-agent design: a supervisor graph whose nodes are
themselves LangGraph subgraphs.
"""
import time

from langgraph.graph import StateGraph, START, END

from agents import (client_health, competitive_intel, hallucination,
                    proposal, second_brain)
from agents.constraints import check_and_fix, ground_internal_figures, internal_constraints
from agents.supervisor import plan as supervisor_plan
from agents.synthesizer import adjustments_md, audit_briefing, audit_md, synthesize
from core.agents_meta import AGENT_META
from core.state import AgentState
from memory.vector_store import get_store

AGENT_RUNNERS = {
    "second_brain":      second_brain.run,
    "hallucination":     hallucination.run,
    "client_health":     client_health.run,
    "competitive_intel": competitive_intel.run,
    "proposal":          proposal.run,
}


def _event(node: str, status: str, detail: str, seconds: float) -> dict:
    return {"node": node, "status": status, "detail": detail, "seconds": round(seconds, 1)}


# ── Nodes ────────────────────────────────────────────────────
def supervisor_node(state: AgentState) -> dict:
    t0 = time.time()
    out = supervisor_plan(state)
    steps = " then ".join(AGENT_META[s["agent"]]["label"] for s in out["plan"])
    return {
        **out,
        "cursor": 0,
        "results": {},
        "trace": (state.get("trace") or []) + [
            _event("supervisor", "ok", f"Plan: {steps}", time.time() - t0)],
    }


def _headline(summary_md: str) -> str:
    """First meaningful line of an agent's summary (skips headings and blank lines)."""
    for line in summary_md.split("\n"):
        line = line.strip()
        if line and not line.startswith("#"):
            return line[:160]
    return ""


def make_agent_node(name: str):
    runner = AGENT_RUNNERS[name]

    def node(state: AgentState) -> dict:
        t0 = time.time()
        try:
            out = runner(state)
        except Exception as e:  # one failing agent should not kill the run
            out = {"status": "error", "summary_md": f"**{AGENT_META[name]['label']} failed:** {e}",
                   "text_output": ""}
        detail = _headline(out.get("summary_md", ""))
        return {
            "results": {**(state.get("results") or {}), name: out},
            "cursor": state.get("cursor", 0) + 1,
            "trace": (state.get("trace") or []) + [
                _event(name, out.get("status", "ok"), detail, time.time() - t0)],
        }

    node.__name__ = f"{name}_node"
    return node


def constraint_check_node(state: AgentState) -> dict:
    """
    Verification layer: every recommendation from Competitor Watch and Client Health
    is checked against the firm's own decisions in memory. Verified conflicts are
    replaced with compliant versions before the briefing is written.
    """
    t0 = time.time()
    results = {k: dict(v) for k, v in (state.get("results") or {}).items()}
    passages = internal_constraints()
    trace = state.get("trace") or []
    if not passages:
        return {"trace": trace + [_event("constraint_check", "ok",
                                         "No internal documents in memory to check against",
                                         time.time() - t0)]}

    # Collect recommendations with a pointer back to where each came from
    recs, where = [], []
    ci = results.get("competitive_intel")
    if ci and ci.get("status") == "ok":
        for i, m in enumerate(ci["strategy"]["recommended_moves"]):
            recs.append(m); where.append(("ci", i))
    ch = results.get("client_health")
    if ch and ch.get("status") == "ok":
        for ci_idx, client in enumerate(ch["clients"]):
            for a_idx, a in enumerate(client["recommendation"]["actions"]):
                recs.append(f"[{client['client']}] {a}"); where.append(("ch", ci_idx, a_idx))
    if not recs:
        return {"trace": trace + [_event("constraint_check", "ok", "No recommendations to check",
                                         time.time() - t0)]}

    try:
        fixed, adjustments = check_and_fix(recs, passages)
    except Exception as e:
        fixed, adjustments = recs, []
        trace = trace + [_event("constraint_check", "error",
                                f"Decision check could not run: {str(e)[:100]}", time.time() - t0)]
    fixed, figure_fixes = ground_internal_figures(fixed)   # pure code, always runs
    adjustments = adjustments + figure_fixes

    for new, loc in zip(fixed, where):
        if loc[0] == "ci":
            results["competitive_intel"]["strategy"]["recommended_moves"][loc[1]] = new
        else:
            prefix = f"[{results['client_health']['clients'][loc[1]]['client']}] "
            results["client_health"]["clients"][loc[1]]["recommendation"]["actions"][loc[2]] = \
                new[len(prefix):] if new.startswith(prefix) else new

    # Keep agent summaries (which feed the briefing) in line with the fixed text
    for adj in adjustments:
        for key in ("competitive_intel", "client_health"):
            if key in results:
                results[key]["summary_md"] = results[key]["summary_md"].replace(
                    adj["original"].split("] ", 1)[-1], adj["revised"].split("] ", 1)[-1])
    results["constraint_check"] = {"status": "ok", "adjustments": adjustments,
                                   "checked": len(recs), "summary_md": "", "text_output": ""}
    n_conf = len(adjustments) - len(figure_fixes)
    if adjustments:
        detail = (f"Checked {len(recs)} recommendations: {n_conf} conflicted with internal "
                  f"decisions, {len(figure_fixes)} quoted figures not in your documents — fixed")
    else:
        detail = f"Checked {len(recs)} recommendations; no conflicts or unverified figures"
    return {"results": results,
            "trace": trace + [_event("constraint_check", "ok", detail, time.time() - t0)]}


def synthesizer_node(state: AgentState) -> dict:
    t0 = time.time()
    try:
        body = synthesize(state)
        status = "ok"
    except Exception as e:
        body = "\n\n".join(r.get("summary_md", "") for r in (state.get("results") or {}).values())
        status = f"error: {e}"
    # Audit the briefing itself: the merge step can introduce new errors
    body, notes = audit_briefing(body)
    fixed = [n for n in notes if n["removed"]]
    if get_store().is_empty:
        detail = "Briefing ready; no internal documents loaded, so nothing to check against"
        warning = ("\n\n⚠️ **No internal documents loaded.** These recommendations were not "
                   "checked against your firm's own decisions, clients or commitments. Add "
                   "your documents in the sidebar and run again for a checked briefing.")
    else:
        detail = (f"Briefing ready; corrected {len(fixed)} statement(s) that did not match "
                  "your documents") if fixed else \
                 "Briefing ready; statements checked against your documents"
        warning = ""
    return {
        "final_report": body + warning + audit_md(notes) + adjustments_md(state),
        "trace": (state.get("trace") or []) + [
            _event("synthesizer", status, detail, time.time() - t0)],
    }


# ── Routing ──────────────────────────────────────────────────
def route_next(state: AgentState) -> str:
    plan = state.get("plan") or []
    cursor = state.get("cursor", 0)
    return plan[cursor]["agent"] if cursor < len(plan) else "synthesizer"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("supervisor", supervisor_node)
    for name in AGENT_RUNNERS:
        g.add_node(name, make_agent_node(name))
    g.add_node("constraint_check", constraint_check_node)
    g.add_node("synthesizer", synthesizer_node)

    routes = {name: name for name in AGENT_RUNNERS} | {"synthesizer": "constraint_check"}
    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", route_next, routes)
    for name in AGENT_RUNNERS:
        g.add_conditional_edges(name, route_next, routes)
    g.add_edge("constraint_check", "synthesizer")
    g.add_edge("synthesizer", END)
    return g.compile()


def initial_state(request: str, attachment: str = "", forced_agent: str | None = None) -> AgentState:
    return {"request": request, "attachment": attachment, "forced_agent": forced_agent,
            "plan": [], "cursor": 0, "results": {}, "trace": [], "final_report": ""}


def stream_run(graph, state: AgentState):
    """
    Yields (node_name, merged_state) after every node finishes, so the UI
    can draw the relay timeline live. The last yield holds the final state.
    """
    merged = dict(state)
    for update in graph.stream(state, stream_mode="updates"):
        for node_name, partial in update.items():
            if partial:
                merged.update(partial)
            yield node_name, merged