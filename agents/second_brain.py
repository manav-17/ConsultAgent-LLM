"""
second_brain.py — perfect memory over the team's own documents.

Subgraph:  expand_query  ->  retrieve  ->  answer
- expand_query: rewrites the question into 2-4 search queries
                (multi-query retrieval finds more than one query would)
- retrieve:     searches FAISS for each, merges and de-duplicates
- answer:       grounded answer with [S1] citations + structured items
                (decisions, commitments, dates) when relevant
"""
from typing import Literal, Optional, TypedDict

from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

from agents.common import current_step, result, needs_input, previous_text_output
from core.agents_meta import OWN_DOCS_NOTE
from core.llm import structured_call
from core.state import AgentState
from memory.vector_store import get_store


# ── Schemas ──────────────────────────────────────────────────
class QueryPlan(BaseModel):
    mode: Literal["question", "commitments", "decisions", "timeline"] = Field(
        description="question=general Q&A, commitments=promises/action items, "
                    "decisions=what was decided, timeline=what happened when")
    search_queries: list[str] = Field(description="2-4 short search queries", min_length=1)


class MemoryItem(BaseModel):
    item: str = Field(description="the decision, commitment or event")
    owner: Optional[str] = Field(default=None, description="person responsible")
    counterparty: Optional[str] = Field(default=None, description="who it was promised to")
    date: Optional[str] = None
    status: Literal["done", "open", "overdue", "unclear"] = "unclear"
    source: str = Field(description="citation label like S2")


class MemoryAnswer(BaseModel):
    answer_md: str = Field(description="markdown answer that cites sources like [S1]")
    items: list[MemoryItem] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"]
    gaps: list[str] = Field(default_factory=list,
                            description="things the documents do not cover")


# ── Subgraph ─────────────────────────────────────────────────
class SBState(TypedDict, total=False):
    question: str
    prior: str          # output of the previous specialist, if the plan chained them
    plan: dict
    hits: list[dict]
    answer: dict


def expand_query(s: SBState) -> SBState:
    plan = structured_call(
        QueryPlan,
        "You turn a question about a company's internal documents into search queries "
        "for a semantic search engine. Pick the mode that best fits the question. "
        + OWN_DOCS_NOTE,
        f"Question: {s['question']}",
    )
    queries = [q for q in plan.search_queries if q.strip()][:4] or [s["question"]]
    if s["question"] not in queries:
        queries.append(s["question"])
    return {"plan": {"mode": plan.mode, "queries": queries}}


def retrieve(s: SBState) -> SBState:
    store = get_store()
    merged: dict[int, dict] = {}
    for q in s["plan"]["queries"]:
        for hit in store.search(q, k=5):
            if hit["id"] not in merged or hit["score"] > merged[hit["id"]]["score"]:
                merged[hit["id"]] = hit
    hits = sorted(merged.values(), key=lambda h: h["score"], reverse=True)[:8]
    for i, h in enumerate(hits, 1):
        h["label"] = f"S{i}"
    return {"hits": hits}


def answer(s: SBState) -> SBState:
    if not s["hits"]:
        return {"answer": {"answer_md": "Nothing relevant was found in the stored documents.",
                           "items": [], "confidence": "low", "gaps": [s["question"]]}}
    context = "\n\n".join(f"[{h['label']}] (file: {h['source']})\n{h['text']}" for h in s["hits"])
    mode = s["plan"]["mode"]
    out = structured_call(
        MemoryAnswer,
        "You are the company's second brain. Answer ONLY from the provided excerpts. "
        "Cite every fact with its label like [S1]. If the excerpts do not contain the "
        "answer, say so and list it under gaps — never invent details. "
        f"Mode is '{mode}': for commitments/decisions/timeline, also fill items with one "
        "entry per commitment, decision or event, and judge status from the evidence "
        "(overdue if a deadline passed with no sign of delivery). Never infer rules, "
        "approvals or exceptions the excerpts do not state (e.g. a freeze on "
        "non-technical roles says nothing about technical hires). Facts about our own "
        "firm and clients must come ONLY from the excerpts, never from the previous "
        "specialist's output. Draft proposals and plans are not proof of delivered "
        "work: call them 'draft' or 'proposed'. A client with adoption problems, "
        "complaints or overdue work is a risk, not 'no risk'. " + OWN_DOCS_NOTE,
        f"Question: {s['question']}\n\nExcerpts:\n{context}"
        + (f"\n\nOutput from the previous specialist (use as context; cite only the "
           f"excerpts):\n{s['prior'][:6000]}" if s.get("prior") else ""),
    )
    return {"answer": out.model_dump()}


_graph = None


def _build():
    g = StateGraph(SBState)
    g.add_node("expand_query", expand_query)
    g.add_node("retrieve", retrieve)
    g.add_node("answer", answer)
    g.add_edge(START, "expand_query")
    g.add_edge("expand_query", "retrieve")
    g.add_edge("retrieve", "answer")
    g.add_edge("answer", END)
    return g.compile()


# ── Entry point used by the main graph ───────────────────────
def run(state: AgentState) -> dict:
    global _graph
    store = get_store()
    if store.is_empty:
        return needs_input("The Second Brain is empty. Upload documents in the sidebar "
                           "(or load the sample knowledge base) and try again.")

    question = current_step(state).get("task") or state.get("request", "")
    _graph = _graph or _build()
    prior, _ = previous_text_output(state)   # e.g. Competitor Watch's report
    out = _graph.invoke({"question": question, "prior": prior})

    ans, hits = out["answer"], out["hits"]
    # Code check: numbers about our own clients must exist near that client in the docs
    from agents.constraints import ground_internal_figures
    lines, fig_notes = ground_internal_figures(ans["answer_md"].split("\n"))
    ans["answer_md"] = "\n".join(lines)
    if fig_notes:
        ans.setdefault("gaps", []).append(
            "Removed figures not found in your documents: "
            + ", ".join(n["decision"] for n in fig_notes))
    lines = [f"### Second Brain\n\n{ans['answer_md']}"]
    if ans["items"]:
        lines.append("\n**Found items**")
        for it in ans["items"]:
            who = f" — {it['owner']}" if it.get("owner") else ""
            when = f" ({it['date']})" if it.get("date") else ""
            lines.append(f"- [{it['status']}] {it['item']}{who}{when} [{it['source']}]")
    if ans["gaps"]:
        lines.append("\n**Not covered by documents:** " + "; ".join(ans["gaps"]))

    return result(
        "\n".join(lines),
        text_output=ans["answer_md"],
        mode=out["plan"]["mode"],
        queries=out["plan"]["queries"],
        answer=ans,
        sources=[{"label": h["label"], "source": h["source"],
                  "score": round(h["score"], 3), "text": h["text"]} for h in hits],
    )