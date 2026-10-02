"""
hallucination.py — an AI that audits other AI output.

Subgraph:
  extract_claims -> verify_claims -> score --(issues?)--> rewrite -> END
                                         └--(clean)----------------> END

- extract_claims: pulls out atomic, checkable factual claims
- verify_claims:  for each claim, searches the web (and internal memory
                  if documents exist) and judges supported / contradicted /
                  unverifiable with evidence — runs claims in parallel
- score:          deterministic trust score from the verdicts
- rewrite:        only if needed — fixes contradicted claims and hedges
                  unverifiable ones, leaving everything else untouched
"""
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, Optional, TypedDict

from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

from agents.common import resolve_input_text, result, today
from core.config import MAX_CLAIMS, PARALLEL_WORKERS
from core.llm import structured_call, text_call
from core.state import AgentState
from core.tools import web_search, format_results
from memory.vector_store import get_store


# ── Schemas ──────────────────────────────────────────────────
class Claim(BaseModel):
    claim: str = Field(description="one self-contained factual statement")
    kind: Literal["date", "number", "person", "organisation", "event", "technical", "other"]


class ClaimList(BaseModel):
    claims: list[Claim]


class Verdict(BaseModel):
    verdict: Literal["supported", "contradicted", "unverifiable"]
    confidence: int = Field(ge=0, le=100)
    explanation: str = Field(description="one or two sentences citing the evidence")
    correction: Optional[str] = Field(default=None,
                                      description="the correct statement, only if contradicted")
    source_url: Optional[str] = None


VERDICT_WEIGHT = {"supported": 1.0, "unverifiable": 0.5, "contradicted": 0.0}


# ── Subgraph ─────────────────────────────────────────────────
class FCState(TypedDict, total=False):
    text: str
    claims: list[dict]
    checks: list[dict]
    trust_score: int
    corrected_text: str


def extract_claims(s: FCState) -> FCState:
    out = structured_call(
        ClaimList,
        "Extract the factual claims from the text that can be checked against "
        "public sources: dates, numbers, names, events, technical facts. Skip "
        "opinions, predictions and advice. Rewrite each claim so it stands alone "
        f"(resolve pronouns). Return at most {MAX_CLAIMS}, most important first.",
        s["text"][:6000],
    )
    return {"claims": [c.model_dump() for c in out.claims[:MAX_CLAIMS]]}


def _check_one(claim: dict) -> dict:
    evidence = format_results(web_search(claim["claim"]))
    store = get_store()
    if not store.is_empty:
        internal = store.search(claim["claim"], k=2)
        relevant = [h for h in internal if h["score"] > 0.45]
        if relevant:
            evidence += "\n\nInternal documents:\n" + "\n".join(
                f"(file {h['source']}) {h['text'][:500]}" for h in relevant)

    verdict = structured_call(
        Verdict,
        f"Today is {today()}. You are a strict fact checker. Judge the claim ONLY "
        "from the evidence given. 'supported' needs clear agreement, 'contradicted' "
        "needs clear disagreement, otherwise 'unverifiable'. Do not use outside "
        "knowledge to fill gaps. Pick the most relevant source URL.",
        f"Claim: {claim['claim']}\n\nEvidence:\n{evidence}",
        temperature=0.0,
    )
    return {**claim, **verdict.model_dump()}


def verify_claims(s: FCState) -> FCState:
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as pool:
        checks = list(pool.map(_check_one, s["claims"]))
    return {"checks": checks}


def score(s: FCState) -> FCState:
    checks = s["checks"]
    if not checks:
        return {"trust_score": 100}
    value = sum(VERDICT_WEIGHT[c["verdict"]] for c in checks) / len(checks)
    return {"trust_score": round(value * 100)}


def needs_rewrite(s: FCState) -> str:
    return "rewrite" if any(c["verdict"] != "supported" for c in s["checks"]) else "done"


def rewrite(s: FCState) -> FCState:
    issues = "\n".join(
        f"- {c['claim']} -> {c['verdict'].upper()}"
        + (f" (correct: {c['correction']})" if c.get("correction") else "")
        for c in s["checks"] if c["verdict"] != "supported"
    )
    fixed = text_call(
        "Edit the text minimally. Replace contradicted claims with the correction. "
        "Soften unverifiable claims with hedging ('reportedly', 'according to some "
        "sources') or remove them. Keep everything else word for word. Return only "
        "the edited text.",
        f"Text:\n{s['text']}\n\nIssues:\n{issues}",
        temperature=0.0,
    )
    return {"corrected_text": fixed}


_graph = None


def _build():
    g = StateGraph(FCState)
    g.add_node("extract_claims", extract_claims)
    g.add_node("verify_claims", verify_claims)
    g.add_node("score", score)
    g.add_node("rewrite", rewrite)
    g.add_edge(START, "extract_claims")
    g.add_edge("extract_claims", "verify_claims")
    g.add_edge("verify_claims", "score")
    g.add_conditional_edges("score", needs_rewrite, {"rewrite": "rewrite", "done": END})
    g.add_edge("rewrite", END)
    return g.compile()


# ── Entry point ──────────────────────────────────────────────
def run(state: AgentState) -> dict:
    global _graph
    text, origin = resolve_input_text(state)
    if len(text) < 40:
        return result("**Input needed.** Paste the text you want fact-checked "
                      "in the attachment box.", status="needs_input")

    _graph = _graph or _build()
    out = _graph.invoke({"text": text})
    checks = out.get("checks", [])
    trust = out.get("trust_score", 100)
    counts = {v: sum(c["verdict"] == v for c in checks) for v in VERDICT_WEIGHT}

    lines = [f"### Fact Checker\n\nChecked text from the {origin}. "
             f"**Trust score: {trust}/100** — {counts['supported']} supported, "
             f"{counts['contradicted']} contradicted, {counts['unverifiable']} unverifiable."]
    for c in checks:
        if c["verdict"] == "contradicted":
            fix = f" Correct: {c['correction']}" if c.get("correction") else ""
            lines.append(f"- ❌ {c['claim'].rstrip('.')}.{fix}")
        elif c["verdict"] == "unverifiable":
            lines.append(f"- ⚠️ {c['claim']} (could not verify)")
    if not checks:
        lines.append("No checkable factual claims were found.")

    corrected = out.get("corrected_text", "")
    return result(
        "\n".join(lines),
        text_output=corrected or text,
        source=origin,
        trust_score=trust,
        counts=counts,
        checks=checks,
        original_text=text,
        corrected_text=corrected,
    )
