"""
proposal.py — scores a proposal, rewrites it, and re-scores in a loop.

Subgraph (a reflection loop with a grounding guard):

    score --(win prob >= target or rounds used up)--> END
      ^  └--(needs work)--> rewrite --> ground --> consistency --+
      +----------------------------------------------------------+

- score:   LLM grades 8 rubric dimensions 0-10 with the specific issue
           and fix for each. The weighted total and win probability are
           computed in code, so the maths is transparent and repeatable.
- rewrite: rewrites the whole proposal focused on the weakest dimensions,
           using placeholders where evidence is needed.
- consistency: pure code, no LLM. Cost rows must add up to the total (pinned
           to the draft's price), payment % to 100 and amounts to % x price,
           and the stated delivery time must match the plan's phases.
- ground:  LLMs ignore "don't invent facts" instructions, so a second call
           lists every specific claim in the rewrite that is NOT supported by
           the original draft or client context (names, numbers, certifications,
           client data, products). Code then replaces each one with a
           placeholder — the model cannot choose to keep it.

Note: win probability is a rubric-based estimate, not a model trained on
historical win/loss data. Swap in real data later to calibrate it.
"""
import math
import re
from typing import Literal, TypedDict

from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

from agents.common import resolve_input_text, current_step, result
from agents.consistency import make_consistent
from core.config import MAX_PROPOSAL_ROUNDS, TARGET_WIN_PROB
from core.llm import structured_call, text_call
from core.state import AgentState

RUBRIC = {
    "problem_understanding": (0.15, "Shows a specific understanding of the client's problem"),
    "value_and_roi":         (0.20, "Quantifies the business outcome and return"),
    "pricing_justification": (0.15, "Pricing is clear and tied to value"),
    "scope_and_timeline":    (0.10, "Scope, milestones and timeline are realistic"),
    "differentiation":       (0.10, "Explains why us and not a competitor"),
    "social_proof":          (0.10, "Relevant case studies, references or credentials"),
    "risk_mitigation":       (0.10, "Addresses risks, assumptions and dependencies"),
    "clarity_and_next_step": (0.10, "Easy to read with a clear call to action"),
}
Dimension = Literal[tuple(RUBRIC.keys())]  # type: ignore[valid-type]


class DimensionScore(BaseModel):
    dimension: Dimension
    evidence: str = Field(default="", description="exact quote copied from the proposal that "
                          "shows this dimension is addressed; empty string if nothing does")
    score: int = Field(ge=0, le=10)
    issue: str = Field(description="the main weakness, one sentence")
    fix: str = Field(description="the specific change that would raise the score")


class RubricResult(BaseModel):
    scores: list[DimensionScore]


# Only these categories are factual claims that can be false. Plans, intentions
# and process descriptions ("we will run weekly reviews") are never redacted.
REDACT_CATEGORIES = {
    "product_name", "certification", "experience_claim", "named_client_or_person",
    "testimonial", "client_metric", "performance_figure",
}
# Prices are NOT redacted: a payment split of the draft's own price is a proposal
# decision, and agents/consistency.py already verifies the arithmetic.
# These two only count as fabricated if they actually contain a number.
NEEDS_NUMBER = {"client_metric", "performance_figure"}


class UnsupportedClaim(BaseModel):
    category: Literal["product_name", "certification", "experience_claim",
                      "named_client_or_person", "testimonial", "client_metric",
                      "performance_figure", "price", "plan_or_intention", "other"] = Field(
        description="what kind of statement this is; plans, approaches and promises "
                    "about how the work will be done are 'plan_or_intention'")
    text: str = Field(description="the unsupported phrase, copied EXACTLY character for "
                                  "character from the rewritten proposal, as short as possible")
    reason: str = Field(description="why it is unsupported, a few words")
    placeholder: str = Field(description="replacement like '[Add: your certifications]' — "
                                         "must NOT contain example names or numbers")


class GroundingReport(BaseModel):
    claims: list[UnsupportedClaim]


EVIDENCE_CAP = 2  # max score for a dimension with no verifiable quote


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def apply_evidence_cap(scores: list[dict], proposal: str) -> list[dict]:
    """
    Small models are lenient judges: they write "No case studies" and still give 8/10.
    So a score only counts if the model quoted text that really exists in the proposal.
    No quote, or a quote that is not found, caps the dimension at EVIDENCE_CAP.
    """
    body = _norm(proposal)
    for sc in scores:
        quote = _norm(sc.get("evidence") or "")
        found = len(quote) >= 8 and (quote in body or quote[:60] in body)
        sc["evidence_found"] = found
        if not found and sc["score"] > EVIDENCE_CAP:
            sc["model_score"] = sc["score"]
            sc["score"] = EVIDENCE_CAP
    return scores


def weighted_total(scores: list[dict]) -> float:
    by_dim = {s["dimension"]: s["score"] for s in scores}
    return sum(w * by_dim.get(d, 0) * 10 for d, (w, _) in RUBRIC.items())


def win_probability(total: float) -> int:
    """Logistic mapping from rubric total (0-100) to an estimated win chance."""
    p = 100 / (1 + math.exp(-(total - 62) / 8))
    return int(max(3, min(95, round(p))))


# ── Subgraph ─────────────────────────────────────────────────
class PCState(TypedDict, total=False):
    draft: str
    context: str
    current: str
    round: int
    history: list[dict]     # one entry per scoring pass
    latest: dict
    redactions: list[dict]  # unsupported claims removed by the ground step
    grounding_failed: bool  # True if the fact guard could not run
    scored_text: str        # last version that was successfully scored
    stop_reason: str        # set if a later round failed and the loop stopped early
    consistency_notes: list[str]  # arithmetic fixes made by code
    grounding_error: str


def _score_dimensions(dims: list[str], s: PCState) -> list[dict]:
    rubric_text = "\n".join(f"- {d}: {RUBRIC[d][1]}" for d in dims)
    out = structured_call(
        RubricResult,
        "You are a strict bid manager who has reviewed hundreds of consulting proposals. "
        "For EACH dimension listed: first copy into 'evidence' the exact sentence from "
        "the proposal that addresses it (empty string if none), then score it. "
        "Anchors: 0-2 = missing or a bare claim; 3-4 = mentioned but vague; "
        "5-6 = addressed with some specifics; 7-8 = specific and convincing; "
        "9-10 = exceptional, rarely given. The score must agree with the issue you "
        "write: if the issue says something is missing, the score is 0-2. Keep issue "
        "and fix to one short sentence each. Score ONLY these dimensions:\n" + rubric_text,
        f"Client context: {s.get('context') or 'not provided'}\n\nProposal:\n{s['current'][:9000]}",
        temperature=0.0,
    )
    return [x.model_dump() for x in out.scores if x.dimension in dims]


def score(s: PCState) -> PCState:
    if s.get("stop_reason"):  # a rewrite failed: nothing new to score, don't waste calls
        return {}
    # Two smaller calls (4 dimensions each) are far more reliable than one
    # big call for small reasoning models that can run out of output budget.
    names = list(RUBRIC)
    try:
        scores = _score_dimensions(names[:4], s) + _score_dimensions(names[4:], s)
    except Exception as e:
        if s.get("history"):  # a later round failed: keep the best scored version
            return {"current": s["scored_text"],
                    "stop_reason": f"Scoring failed in round {s.get('round', 0)}: {str(e)[:120]}"}
        raise  # the very first score failing means we have nothing to show

    seen = {x["dimension"] for x in scores}
    for d in RUBRIC:  # fill any dimension the model skipped
        if d not in seen:
            scores.append({"dimension": d, "evidence": "", "score": 0,
                           "issue": "Not addressed", "fix": "Add this section"})
    scores = apply_evidence_cap(scores, s["current"])
    total = weighted_total(scores)
    entry = {"round": s.get("round", 0), "total": round(total, 1),
             "win_probability": win_probability(total), "scores": scores,
             "text": s["current"], "notes": list(s.get("consistency_notes") or [])}
    return {"latest": entry, "history": (s.get("history") or []) + [entry],
            "scored_text": s["current"]}


def should_continue(s: PCState) -> str:
    if s.get("stop_reason"):
        return "done"
    if s["latest"]["win_probability"] >= TARGET_WIN_PROB:
        return "done"
    if s.get("round", 0) >= MAX_PROPOSAL_ROUNDS:
        return "done"
    return "rewrite"


def rewrite(s: PCState) -> PCState:
    weakest = sorted(s["latest"]["scores"], key=lambda x: x["score"])[:4]
    focus = "\n".join(f"- {w['dimension']} ({w['score']}/10): {w['fix']}" for w in weakest)
    try:
        improved = _rewrite_text(s, focus)
    except Exception as e:
        return {"stop_reason": f"Rewrite failed in round {s.get('round', 0) + 1}: {str(e)[:120]}"}
    if not improved.strip():
        return {"stop_reason": "Rewrite returned empty text"}
    return {"current": improved, "round": s.get("round", 0) + 1}


def _rewrite_text(s: PCState, focus: str) -> str:
    return text_call(
        "Rewrite this consulting proposal to fix the listed weaknesses while keeping "
        "the client, scope and facts the same. Use clear headings. NEVER invent "
        "facts: no product names, certifications, years of experience, client names, "
        "testimonials, the client's own metrics, performance figures or prices that "
        "are not in the draft. Where evidence is needed, insert a placeholder such as "
        "[Add: relevant case study] — placeholders must not contain example names or "
        "numbers. Prefer milestone-based payment terms over full advance payment. "
        "Return only the full rewritten proposal in markdown.",
        f"Client context: {s.get('context') or 'not provided'}\n\nFix these first:\n{focus}"
        f"\n\nProposal:\n{s['current']}",
        temperature=0.2,
    )


def ground(s: PCState) -> PCState:
    if s.get("stop_reason"):
        return {}
    try:
        report = _grounding_report(s)
    except Exception as e:  # never crash the whole agent; flag it loudly instead
        return {"grounding_failed": True,
                "grounding_error": str(e)[:200]}
    return _apply_redactions(s, report)


def _grounding_report(s: PCState) -> "GroundingReport":
    return structured_call(
        GroundingReport,
        "You audit a rewritten proposal for fabricated FACTS. List statements in the "
        "REWRITTEN proposal that assert a verifiable fact not supported by the ORIGINAL "
        "DRAFT or CLIENT CONTEXT, and label each with its category: product names, "
        "certifications, years or breadth of experience, named clients or people, "
        "testimonials or quotes, the client's own metrics, performance or percentage "
        "figures, prices. Do NOT list plans, approaches, phases, deliverables or "
        "promises about how the work will be done — those are proposals, not facts; "
        "if unsure, label them plan_or_intention. Do not list existing [placeholders]. "
        "Copy each phrase exactly as it appears so it can be found and replaced.",
        f"ORIGINAL DRAFT:\n{s['draft']}\n\nCLIENT CONTEXT:\n{s.get('context') or 'none'}"
        f"\n\nREWRITTEN:\n{s['current']}",
        temperature=0.0,
    )


def _apply_redactions(s: PCState, report: "GroundingReport") -> PCState:
    text, removed = s["current"], []
    for c in report.claims:
        if c.category not in REDACT_CATEGORIES:
            continue  # plans and intentions stay: code decides what counts as a fact
        phrase = c.text.strip()
        if c.category in NEEDS_NUMBER and not re.search(r"\d", phrase):
            continue  # "faster responses" is a goal, not a fabricated figure
        if len(phrase) >= 3 and phrase in text and phrase not in s["draft"]:
            placeholder = c.placeholder.strip() or "[Add: verified detail]"
            if not placeholder.startswith("["):
                placeholder = f"[{placeholder}]"
            text = text.replace(phrase, placeholder)
            removed.append({"removed": phrase, "category": c.category, "reason": c.reason,
                            "placeholder": placeholder, "round": s.get("round", 0)})
    return {"current": text, "redactions": (s.get("redactions") or []) + removed}


def consistency(s: PCState) -> PCState:
    if s.get("stop_reason"):
        return {}
    fixed, notes = make_consistent(s["current"], s["draft"])
    return {"current": fixed, "consistency_notes": notes}  # notes for the latest version only


_graph = None


def _build():
    g = StateGraph(PCState)
    g.add_node("score", score)
    g.add_node("rewrite", rewrite)
    g.add_node("ground", ground)
    g.add_node("consistency", consistency)
    g.add_edge(START, "score")
    g.add_conditional_edges("score", should_continue, {"rewrite": "rewrite", "done": END})
    g.add_edge("rewrite", "ground")
    g.add_edge("ground", "consistency")
    g.add_edge("consistency", "score")
    return g.compile()


# ── Entry point ──────────────────────────────────────────────
def run(state: AgentState) -> dict:
    global _graph
    draft, origin = resolve_input_text(state)
    if len(draft) < 200:
        return result("**Input needed.** Paste the proposal draft in the attachment box.",
                      status="needs_input")

    context = current_step(state).get("task", "")
    _graph = _graph or _build()
    out = _graph.invoke({"draft": draft, "current": draft, "context": context,
                         "round": 0, "history": [], "redactions": []})

    history = out["history"]
    first, last = history[0], history[-1]  # 'last' is replaced by the best round below
    # Keep the best-scoring rewrite from ANY round (ties -> newest). Only if no
    # rewrite beat the original do we return nothing.
    best = max(reversed(history), key=lambda h: h["total"])
    improved = best["text"] if best["round"] > 0 else ""
    last = best

    by_first = {x["dimension"]: x for x in first["scores"]}
    weakest = sorted(first["scores"], key=lambda x: x["score"])[:3]
    lines = [
        "### Proposal Coach\n",
        f"Estimated win probability **{first['win_probability']}% → {last['win_probability']}%** "
        f"after {len(history) - 1} rewrite round(s). Rubric total "
        f"{first['total']} → {last['total']}.",
        "\n**Biggest weaknesses in the original**",
    ]
    lines += [f"- {w['dimension'].replace('_', ' ')} ({w['score']}/10): {w['issue']}" for w in weakest]
    redactions = out.get("redactions", [])
    notes = best.get("notes") or []
    if improved and notes:
        lines.append("\n**Number check:** fixed by code so the figures agree:")
        lines += [f"- {n}" for n in notes]
    if out.get("stop_reason"):
        lines.append(f"\n⚠️ Stopped early and kept the best scored version. {out['stop_reason']}")
    if out.get("grounding_failed"):
        lines.append("\n⚠️ **Fact guard could not run** on the rewrite, so it may contain "
                     "unverified claims (names, numbers, certifications). Review it before "
                     "sending, or run it again.")
    if redactions:
        lines.append(f"\n**Fact guard:** removed {len(redactions)} unsupported claim(s) "
                     "the rewrite invented, replaced with placeholders:")
        lines += [f"- ~~{r['removed']}~~ → `{r['placeholder']}` ({r['category'].replace('_', ' ')})"
                  for r in redactions[:8]]
        if len(redactions) > 8:
            lines.append(f"- …and {len(redactions) - 8} more")
    capped = [x for x in first["scores"] if "model_score" in x]
    if capped:
        lines.append(f"\n**Evidence check:** {len(capped)} dimension(s) in the original were "
                     "scored without supporting text in the proposal and capped at "
                     f"{EVIDENCE_CAP}/10.")
    lines.append("\n_Win probability is a rubric-based estimate, not a trained prediction._")

    dims = []
    by_last = {x["dimension"]: x for x in last["scores"]}
    for d in RUBRIC:
        dims.append({"dimension": d.replace("_", " "),
                     "weight": RUBRIC[d][0],
                     "before": by_first[d]["score"],
                     "after": by_last[d]["score"],
                     "evidence found": "yes" if by_last[d].get("evidence_found") else "no",
                     "issue": by_first[d]["issue"],
                     "fix": by_first[d]["fix"]})

    return result("\n".join(lines),
                  text_output=improved or draft,
                  source=origin,
                  history=[{k: v for k, v in h.items() if k not in ("scores", "text", "notes")} for h in history],
                  dimensions=dims,
                  original=draft,
                  improved=improved,
                  redactions=redactions,
                  grounding_failed=bool(out.get("grounding_failed")),
                  consistency_notes=notes if improved else [])