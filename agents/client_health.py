"""
client_health.py — predicts which clients are happy and which are drifting.

Subgraph:  read_signals -> score -> recommend

- read_signals: the LLM reads emails / interaction logs and extracts
                observable signals per client (tone, response-time trend,
                engagement, missed meetings, escalations, hidden concerns)
- score:        DETERMINISTIC formula turns signals into a 0-100 health
                score — explainable, every point can be traced to a signal
- recommend:    actions, talking points and a suggested reply for each
                client that is not healthy

If no text is attached, it pulls the client's history from the Second Brain.
"""
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, TypedDict

from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

from agents.common import current_step, result, today
from core.config import PARALLEL_WORKERS
from core.llm import structured_call
from core.state import AgentState
from memory.vector_store import get_store


# ── Schemas ──────────────────────────────────────────────────
class ClientSignals(BaseModel):
    client: str
    contact: str = Field(description="full name of the main client contact who wrote the "
                                     "emails, exactly as written; use '' only if no name appears")
    tone: str = Field(description="one or two words, e.g. 'enthusiastic', 'politely distant'")
    sentiment: int = Field(ge=-100, le=100, description="-100 very negative to +100 very positive")
    response_trend: Literal["faster", "stable", "slower", "much_slower", "unknown"]
    engagement: Literal["high", "medium", "low", "disengaged"]
    missed_meetings: int = Field(ge=0)
    escalations: int = Field(ge=0, description="complaints, escalations to seniors, delay complaints")
    hidden_concerns: list[str] = Field(description="concerns implied but not said directly")
    evidence: list[str] = Field(description="2-4 short quotes that justify the reading")


class SignalsList(BaseModel):
    clients: list[ClientSignals]


class Recommendation(BaseModel):
    actions: list[str] = Field(description="3-4 concrete next steps, most urgent first")
    talking_points: list[str] = Field(description="3 points for the next conversation")
    suggested_reply: str = Field(description="short, warm, specific email reply")


# ── Deterministic scoring ────────────────────────────────────
TREND_POINTS = {"faster": 5, "stable": 0, "slower": -15, "much_slower": -30, "unknown": 0}
ENGAGEMENT_POINTS = {"high": 5, "medium": 0, "low": -15, "disengaged": -30}


def health_score(sig: dict) -> tuple[int, list[str]]:
    score, why = 70.0, ["Baseline 70"]

    sentiment_pts = sig["sentiment"] / 100 * 20
    score += sentiment_pts
    why.append(f"Sentiment {sig['sentiment']:+d} → {sentiment_pts:+.0f}")

    t = TREND_POINTS[sig["response_trend"]]
    score += t
    why.append(f"Response time {sig['response_trend'].replace('_', ' ')} → {t:+d}")

    e = ENGAGEMENT_POINTS[sig["engagement"]]
    score += e
    why.append(f"Engagement {sig['engagement']} → {e:+d}")

    m = -min(sig["missed_meetings"] * 8, 24)
    if m:
        score += m
        why.append(f"{sig['missed_meetings']} missed meeting(s) → {m:+d}")

    x = -min(sig["escalations"] * 10, 30)
    if x:
        score += x
        why.append(f"{sig['escalations']} escalation(s) → {x:+d}")

    return max(5, min(100, round(score))), why


def risk_band(score: int) -> str:
    if score >= 70:
        return "healthy"
    if score >= 45:
        return "watch"
    return "critical"


# ── Subgraph ─────────────────────────────────────────────────
class CHState(TypedDict, total=False):
    text: str
    focus: str
    clients: list[dict]


def read_signals(s: CHState) -> CHState:
    out = structured_call(
        SignalsList,
        f"Today is {today()}. You are an account manager reading client "
        "communications. For EACH distinct client, extract observable signals. "
        "Read between the lines: polite but vague replies, slower answers and "
        "skipped meetings are warning signs. Base every judgement on the text "
        "and quote it as evidence. Do not invent clients.",
        f"Focus: {s.get('focus', '')}\n\nCommunications:\n{s['text'][:12000]}",
    )
    clients = []
    for sig in out.clients:
        d = sig.model_dump()
        d["score"], d["breakdown"] = health_score(d)
        d["risk"] = risk_band(d["score"])
        d["churn_risk"] = 100 - d["score"]
        clients.append(d)
    clients.sort(key=lambda c: c["score"])
    return {"clients": clients}


def _recommend_one(client: dict) -> dict:
    if client["risk"] == "healthy":
        client["recommendation"] = {
            "actions": ["Keep the current rhythm", "Ask for a referral or case study"],
            "talking_points": [], "suggested_reply": ""}
        return client
    rec = structured_call(
        Recommendation,
        "You are a senior consulting partner saving an at-risk client relationship. "
        "Be specific to the signals; no generic advice. The reply must address the "
        "hidden concerns without sounding defensive. "
        "Do not state that work is finished, attached or sent unless the evidence says "
        "it is; commit to a specific date instead. "
        "For suggested_reply, write ONLY the email body: no greeting, no sign-off, and "
        "do not mention any person by name. Refer to people by role ('our security "
        "lead', 'your team'). In actions and talking_points, only name people who "
        "appear in the evidence.",
        f"Client: {client['client']}\n"
        f"Health: {client['score']}/100 ({client['risk']})\n"
        f"Tone: {client['tone']}\nHidden concerns: {client['hidden_concerns']}\n"
        f"Evidence: {client['evidence']}",
    )
    out = rec.model_dump()

    # Facts are handled by code, not the model: the greeting uses the contact
    # name extracted from the emails, so the model can never invent one.
    contact = (client.get("contact") or "").strip()
    first_name = contact.split()[0] if contact else "[Contact name]"
    if first_name.rstrip(".").lower() in {"dr", "mr", "ms", "mrs", "prof"} and len(contact.split()) > 1:
        first_name = " ".join(contact.split()[:2])  # keep "Dr. Leena"
    body = out["suggested_reply"].strip()
    out["suggested_reply"] = f"Hi {first_name},\n\n{body}\n\nBest regards,\n[Your name]"

    client["recommendation"] = out
    return client


def recommend(s: CHState) -> CHState:
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as pool:
        clients = list(pool.map(_recommend_one, s["clients"]))
    return {"clients": clients}


_graph = None


def _build():
    g = StateGraph(CHState)
    g.add_node("read_signals", read_signals)
    g.add_node("score_and_rank", lambda s: {})  # scoring happens inline; node keeps the trace readable
    g.add_node("recommend", recommend)
    g.add_edge(START, "read_signals")
    g.add_edge("read_signals", "score_and_rank")
    g.add_edge("score_and_rank", "recommend")
    g.add_edge("recommend", END)
    return g.compile()


# ── Entry point ──────────────────────────────────────────────
def run(state: AgentState) -> dict:
    global _graph
    task = current_step(state).get("task") or state.get("request", "")
    text = (state.get("attachment") or "").strip()
    origin = "attachment"

    if len(text) < 80:  # nothing pasted — try the Second Brain
        store = get_store()
        if not store.is_empty:
            hits = store.search(task, k=8)
            text = "\n\n".join(f"(file {h['source']})\n{h['text']}" for h in hits)
            origin = "Second Brain documents"
    if len(text) < 80:
        return result("**Input needed.** Paste client emails or an interaction log "
                      "in the attachment box.", status="needs_input")

    _graph = _graph or _build()
    clients = _graph.invoke({"text": text, "focus": task})["clients"]
    if not clients:
        return result("### Client Health\n\nNo clients could be identified in the text.")

    lines = [f"### Client Health\n\nRead from the {origin}. Lowest health first."]
    for c in clients:
        lines.append(f"- **{c['client']}** — {c['score']}/100, {c['risk']} "
                     f"(tone: {c['tone']}). Next step: {c['recommendation']['actions'][0]}")

    return result("\n".join(lines), source=origin, clients=clients)