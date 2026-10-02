"""
competitive_intel.py — tracks what competitors are doing.

Subgraph:  plan_research -> gather -> analyse -> strategise

- plan_research: identifies competitors and writes targeted searches per
                 signal type (launches, hiring, pricing, partnerships, funding)
- gather:        runs all searches in parallel, de-duplicates by URL
- analyse:       per competitor, turns raw results into dated findings with
                 an implication and a source URL (no source = no finding)
- strategise:    threats, opportunities and recommended moves for us
"""
import time
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from typing import Literal, Optional, TypedDict

from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

from agents.common import current_step, result, today
from agents.constraints import internal_constraints, constraints_text
from core.agents_meta import OWN_DOCS_NOTE
from core.config import MAX_COMPETITORS, PARALLEL_WORKERS
from core.llm import structured_call
from core.state import AgentState
from core.tools import web_search, format_results

SignalType = Literal["launch", "hiring", "pricing", "partnership", "funding", "news"]


# ── Schemas ──────────────────────────────────────────────────
class SearchTask(BaseModel):
    competitor: str
    signal: SignalType
    query: str


class ResearchPlan(BaseModel):
    our_company: str = Field(description="who 'we' are, from the request; default "
                                         "'an AI-native consulting firm'")
    competitors: list[str]
    searches: list[SearchTask] = Field(description="2-3 searches per competitor")


class Finding(BaseModel):
    signal: SignalType
    headline: str
    evidence: str = Field(description="what the source actually says, one sentence")
    implication: str = Field(description="what it means for us, one sentence")
    importance: Literal["high", "medium", "low"]
    date: Optional[str] = None
    source_url: str


class CompetitorReport(BaseModel):
    competitor: str
    summary: str
    threat_level: Literal["high", "medium", "low", "unknown"]
    findings: list[Finding]


class Strategy(BaseModel):
    threats: list[str]
    opportunities: list[str]
    recommended_moves: list[str] = Field(description="3-5 concrete moves for the next 90 days")


# ── Subgraph ─────────────────────────────────────────────────
class CIState(TypedDict, total=False):
    request: str
    plan: dict
    evidence: dict      # competitor -> list of results
    reports: list[dict]
    strategy: dict


def plan_research(s: CIState) -> CIState:
    plan = structured_call(
        ResearchPlan,
        f"Today is {today()}. Plan web searches to track recent competitor moves. "
        f"Use at most {MAX_COMPETITORS} competitors. Write specific queries that "
        "would surface news from the last few months (include the current year "
        "where useful). Cover different signal types per competitor.",
        s["request"],
    )
    competitors = plan.competitors[:MAX_COMPETITORS]
    searches = [t.model_dump() for t in plan.searches if t.competitor in competitors][:12]
    return {"plan": {"our_company": plan.our_company, "competitors": competitors,
                     "searches": searches}}


def gather(s: CIState) -> CIState:
    searches = s["plan"]["searches"]
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as pool:
        batches = list(pool.map(lambda t: web_search(t["query"], 5), searches))
    evidence: dict[str, list] = {c: [] for c in s["plan"]["competitors"]}
    seen = set()
    for task, rows in zip(searches, batches):
        for r in rows:
            if r["url"] and r["url"] not in seen:
                seen.add(r["url"])
                evidence.setdefault(task["competitor"], []).append(r)

    # Search engines sometimes return nothing (e.g. brief rate limiting).
    # Retry once with a simple query before concluding there is no news.
    year = today()[:4]
    for comp, rows in evidence.items():
        if rows:
            continue
        for q in (f"{comp} AI {year}", f"{comp} artificial intelligence news"):
            time.sleep(1.5)
            for r in web_search(q, 6):
                if r["url"] and r["url"] not in seen:
                    seen.add(r["url"])
                    rows.append(r)
            if rows:
                break
    return {"evidence": evidence}


FRESH_DAYS = 365
MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def parse_finding_date(text: str | None) -> date | None:
    """Best-effort date from '2026-06-29', 'Apr 22, 2026', '4 Jun 2025' or '2026'."""
    if not text:
        return None
    t = text.lower()
    m = re.search(r"(20\d\d)-(\d{1,2})-(\d{1,2})", t)
    if m:
        try:
            return date(int(m[1]), int(m[2]), int(m[3]))
        except ValueError:
            pass
    m = re.search(r"([a-z]{3})[a-z]*\.?\s+(\d{1,2}),?\s+(20\d\d)", t) or \
        re.search(r"(\d{1,2})\s+([a-z]{3})[a-z]*\.?,?\s+(20\d\d)", t)
    if m:
        a, b, y = m.groups()
        mon, day = (a, b) if a.isalpha() else (b, a)
        if mon[:3] in MONTHS:
            try:
                return date(int(y), MONTHS[mon[:3]], int(day))
            except ValueError:
                pass
    m = re.search(r"([a-z]{3})[a-z]*\s+(20\d\d)", t)
    if m and m[1][:3] in MONTHS:
        return date(int(m[2]), MONTHS[m[1][:3]], 15)
    m = re.search(r"\b(20\d\d)\b", t)
    if m:
        return date(int(m[1]), 6, 30)  # year only: assume mid-year
    return None


def apply_freshness(report: dict, today_str: str) -> dict:
    """
    Code, not the model, decides what counts as recent and how threatening it is:
    findings older than FRESH_DAYS move to 'older_findings'; the threat level is
    recomputed from recent findings only, so old news cannot inflate it.
    """
    now = datetime.strptime(today_str, "%Y-%m-%d").date()
    # Same article reported under two signal types counts once
    unique, seen = [], set()
    for f in report.get("findings", []):
        key = re.sub(r"[^a-z0-9]", "", (f.get("headline") or "").lower())[:80]
        if key and key in seen:
            continue
        seen.add(key)
        unique.append(f)
    report["findings"] = unique
    fresh, older = [], []
    for f in report.get("findings", []):
        d = parse_finding_date(f.get("date"))
        f["age_days"] = (now - d).days if d else None
        (older if d and (now - d).days > FRESH_DAYS else fresh).append(f)
    dated_fresh = [f for f in fresh if f["age_days"] is not None]
    basis = dated_fresh or fresh
    levels = {f["importance"] for f in basis}
    if not basis:
        threat = "unknown"
    elif "high" in levels and dated_fresh:
        threat = "high"
    elif "high" in levels or "medium" in levels:
        threat = "medium"      # undated-only evidence is capped at medium
    else:
        threat = "low"
    report["model_threat_level"] = report.get("threat_level")
    report["threat_level"] = threat
    report["findings"] = fresh
    report["older_findings"] = older
    if older:
        report["summary"] = (report.get("summary", "") + f" ({len(older)} item(s) older than "
                             "12 months were excluded from this assessment.)").strip()
    if not fresh:
        report["summary"] = ("No news from the last 12 months was found, so the current "
                             "threat is unknown. " + report.get("summary", "")).strip()
    return report


def _analyse_one(args) -> dict:
    competitor, rows, our_company = args
    if not rows:
        # No data is NOT the same as low threat
        return {"competitor": competitor, "threat_level": "unknown", "findings": [],
                "summary": "Web search returned no results for this competitor, so its threat "
                           "level is unknown — not low. Try again in a few minutes."}
    report = structured_call(
        CompetitorReport,
        f"Today is {today()}. You are a competitive intelligence analyst for "
        f"{our_company}. Use ONLY the search results. Every finding must cite a URL "
        "from the results. Prefer recent items; skip anything generic or undated "
        "marketing copy. If nothing meaningful is there, return no findings and set "
        "threat_level to 'unknown' — never call a competitor low threat for lack of data.",
        f"Competitor: {competitor}\n\nSearch results:\n{format_results(rows[:10], 500)}",
    )
    return apply_freshness(report.model_dump(), today())


def analyse(s: CIState) -> CIState:
    jobs = [(c, s["evidence"].get(c, []), s["plan"]["our_company"])
            for c in s["plan"]["competitors"]]
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as pool:
        reports = list(pool.map(_analyse_one, jobs))
    order = {"high": 0, "medium": 1, "low": 2, "unknown": 3}
    reports.sort(key=lambda r: order[r["threat_level"]])
    return {"reports": reports}


def strategise(s: CIState) -> CIState:
    digest = "\n\n".join(
        f"{r['competitor']} (threat {r['threat_level']}): {r['summary']}\n"
        + "\n".join(f"- [{f['signal']}] {f['headline']} — {f['implication']}"
                    for f in r["findings"])
        for r in s["reports"]
    )
    # Prevention: respect the firm's own decisions while writing the strategy
    passages = internal_constraints()
    rules = ("\n\nYour firm's internal decisions (every recommended move MUST comply "
             "with these; resolve any overdue deliverables to a client before proposing "
             f"new paid work to that client):\n{constraints_text(passages)}") if passages else ""
    strategy = structured_call(
        Strategy,
        f"You advise the leadership of {s['plan']['our_company']}. Turn competitor "
        "findings into a short strategy. Be concrete and tied to the findings. "
        "Never recommend something the firm's internal decisions rule out. "
        "Only mention our own clients, projects or commitments if they appear in the "
        "internal decisions provided below; never invent clients (no 'Client A'). "
        + OWN_DOCS_NOTE,
        f"Original request: {s['request']}\n\nFindings:\n{digest}{rules}",
    )
    return {"strategy": strategy.model_dump()}


_graph = None


def _build():
    g = StateGraph(CIState)
    g.add_node("plan_research", plan_research)
    g.add_node("gather", gather)
    g.add_node("analyse", analyse)
    g.add_node("strategise", strategise)
    g.add_edge(START, "plan_research")
    g.add_edge("plan_research", "gather")
    g.add_edge("gather", "analyse")
    g.add_edge("analyse", "strategise")
    g.add_edge("strategise", END)
    return g.compile()


# ── Entry point ──────────────────────────────────────────────
def run(state: AgentState) -> dict:
    global _graph
    request = current_step(state).get("task") or state.get("request", "")
    _graph = _graph or _build()
    out = _graph.invoke({"request": request})

    plan, reports, strategy = out["plan"], out["reports"], out["strategy"]
    if not plan["competitors"]:
        return result("**Input needed.** Name the competitors you want tracked.",
                      status="needs_input")

    lines = ["### Competitor Watch\n"]
    for r in reports:
        lines.append(f"**{r['competitor']}** — threat {r['threat_level']}. {r['summary']}")
        for f in r["findings"][:3]:
            lines.append(f"- {f['headline']} ([source]({f['source_url']}))")
    lines.append("\n**Recommended moves**")
    lines += [f"- {m}" for m in strategy["recommended_moves"]]

    report_text = "\n".join(lines)
    return result(report_text, text_output=report_text,
                  plan=plan, reports=reports, strategy=strategy)