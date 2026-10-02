"""
constraints.py — keeps recommendations consistent with the firm's own decisions.

Two layers, same principle as the rest of ConsultAgent (LLM judges, code verifies):

1. Prevention   internal_constraints() pulls decisions/policies from the Second
                Brain so agents can respect them while writing recommendations.
2. Verification check_and_fix() reviews every recommendation.
3. Grounding    ground_internal_figures() — pure code: numbers quoted about our own
                clients must exist in the documents, otherwise placeholder.
   (layer 2 detail follows) The LLM must QUOTE
                the exact internal decision a recommendation conflicts with; code
                checks that quote really exists in the stored documents. Only
                verified conflicts are replaced with a compliant rewrite, so the
                check can neither miss a real conflict silently nor invent one.
"""
import re

from pydantic import BaseModel, Field

from agents.common import today
from core.agents_meta import OWN_DOCS_NOTE
from core.llm import structured_call
from memory.vector_store import get_store

CONSTRAINT_QUERIES = [
    "leadership decisions policy",
    "we will not / only / must rule",
    "hiring freeze headcount",
    "pricing rules fixed-price limits",
    "technology and model usage policy",
]


def _norm(text: str) -> str:
    return " ".join(text.lower().replace("**", "").split())


_N = r"\d[\d,]*(?:\.\d+)?"
_RANGE = rf"{_N}(?:\s*(?:–|-|to)\s*{_N})?"
FIGURE = re.compile(
    rf"(?:Rs\.?|INR|₹)\s*{_RANGE}\s*(?:lakh|lac|crore|k|cr)?"     # money, incl. 'Rs 40–80 lakh'
    rf"|{_RANGE}\s*%"                                             # percentages, incl. '10–15%'
    rf"|{_N}\s*x\b", re.I)                                       # multiples
NAME = re.compile(r"\b[A-Z][a-zA-Z]{3,}(?:\s+[A-Z][a-zA-Z]+)*")
GENERIC = {"launch", "develop", "negotiate", "introduce", "publish", "include", "create",
           "build", "offer", "form", "deploy", "recruit", "sign", "roll", "run", "start",
           "indian", "india", "mumbai", "cloud", "google", "azure", "claude", "openai",
           "accenture", "infosys", "the", "this", "use", "add", "fixed", "price", "safe",
           "guard", "talent", "accelerator", "program", "rapid", "prototype", "esg"}


def _num_key(fig: str) -> str:
    """'Rs 12 lakh' / '₹12 lakh' / '12lakh' -> '12lakh'; '150 %' -> '150%'."""
    t = re.sub(r"(rs\.?|inr|₹|,|\s)", "", fig.lower())
    return t.replace("lac", "lakh")


def _figure_keys(fig: str) -> list[str]:
    """Every endpoint of a figure with its unit: 'Rs 40–80 lakh' -> ['40lakh', '80lakh']."""
    low = fig.lower().replace("lac", "lakh")
    unit = re.search(r"(lakh|crore|cr|k|%|x)\s*$", low.strip())
    unit = unit.group(1) if unit else ""
    nums = [n.replace(",", "") for n in re.findall(r"\d[\d,]*(?:\.\d+)?", low)]
    return [n + unit for n in nums]


PRICE_WORDS = re.compile(r"\b(pric\w*|fee|quote\w*|proposal|offering|package\w*|budget\w*|"
                         r"cap\w*|ceiling|under|up to|below|engagement|contract|pilot)\b", re.I)
RESULT_WORDS = re.compile(r"\b(sav\w*|avoid\w*|roi|return|reduc\w*|cut|gain\w*|"
                          r"improv\w*|increas\w*|revenue|benefit\w*)\b", re.I)


def _is_proposed_price(text: str, m: re.Match) -> bool:
    if not re.match(r"\s*(?:Rs|INR|₹)", m.group(), re.I):
        return False                      # only money can be a price
    before = text[max(0, m.start() - 45): m.start()]
    after = text[m.end(): m.end() + 30]
    if RESULT_WORDS.search(before[-25:]):
        return False                      # "cost-avoidance (Rs 12 lakh)" is a result
    if re.match(r"\s*(?:in|of|worth of)?\s*(?:sav\w*|reduction|returns?\b)", after, re.I):
        return False                      # "Rs 12 lakh in savings" is a result
    return bool(PRICE_WORDS.search(before) or PRICE_WORDS.search(after))


NEAR_CHARS = 250


def ground_internal_figures(recommendations: list[str]) -> tuple[list[str], list[dict]]:
    """
    A recommendation that names one of our own clients/projects (a name found in
    the internal documents) may only quote money, percentages or multiples that
    also appear in those documents. Anything else is an invented 'fact' about a
    real client and is replaced with a placeholder. Pure code — no LLM.
    """
    store = get_store()
    if store.is_empty:
        return recommendations, []
    corpus = " ".join(c["text"] for c in store.chunks)
    corpus_lower = corpus.lower()

    def figures_near(name: str) -> set[str]:
        """Figures that appear within NEAR_CHARS of this name in the documents.
        '30% over budget' in a leadership note does not verify '30%' for Orbitron."""
        keys: set[str] = set()
        for occ in re.finditer(re.escape(name.lower()), corpus_lower):
            window = corpus[max(0, occ.start() - NEAR_CHARS): occ.end() + NEAR_CHARS]
            keys |= {k for m in FIGURE.finditer(window) for k in _figure_keys(m.group())}
        return keys
    fixed, notes = [], []
    for rec in recommendations:
        names = {n for n in NAME.findall(rec)
                 if n.split()[0].lower() not in GENERIC and n.lower() in corpus_lower}
        if not names:
            fixed.append(rec)
            continue
        known_figs = set().union(*(figures_near(n) for n in names))
        removed = []

        def repl(m):
            if all(k in known_figs for k in _figure_keys(m.group())):
                return m.group()
            # A money amount used as OUR proposed price ("priced at Rs 38 lakh",
            # "a Rs 38 lakh proposal") is a decision, not a claim about a client.
            # Results (savings, %, ROI, multiples) are always checked.
            if _is_proposed_price(rec, m):
                return m.group()
            removed.append(m.group().strip())
            return "[Add: verified figure]"
        new = FIGURE.sub(repl, rec)
        if removed:
            notes.append({"original": rec, "revised": new, "label": "Not in internal documents",
                          "decision": ", ".join(removed),
                          "why": f"Figures quoted for {', '.join(sorted(names))} do not "
                                 "appear anywhere in your documents."})
        fixed.append(new)
    return fixed, notes


def internal_constraints(max_passages: int = 6) -> list[dict]:
    """Passages from memory most likely to contain decisions and policies."""
    store = get_store()
    if store.is_empty:
        return []
    merged: dict[int, dict] = {}
    for q in CONSTRAINT_QUERIES:
        for hit in store.search(q, k=3):
            if hit["id"] not in merged or hit["score"] > merged[hit["id"]]["score"]:
                merged[hit["id"]] = hit
    return sorted(merged.values(), key=lambda h: h["score"], reverse=True)[:max_passages]


def constraints_text(passages: list[dict]) -> str:
    return "\n\n".join(f"(file {p['source']})\n{p['text']}" for p in passages)


def _quote_exists(quote: str) -> bool:
    """True if the quoted decision really appears in a stored document."""
    q = _norm(quote)
    if len(q) < 12:
        return False
    body = [_norm(c["text"]) for c in get_store().chunks]
    return any(q in b or q[:80] in b for b in body)


# ── Schemas ──────────────────────────────────────────────────
class RecCheck(BaseModel):
    index: int = Field(description="index of the recommendation in the numbered list")
    conflicts: bool
    decision_quote: str = Field(default="", description="the internal decision it conflicts "
                                "with, copied EXACTLY from the documents; empty if no conflict")
    explanation: str = Field(default="", description="one sentence on why it conflicts")
    revised: str = Field(default="", description="a compliant rewrite that keeps the goal; "
                                                 "empty if no conflict")


class RecCheckList(BaseModel):
    checks: list[RecCheck]


# ── Main check ───────────────────────────────────────────────
def check_and_fix(recommendations: list[str], passages: list[dict]) -> tuple[list[str], list[dict]]:
    """
    Returns (fixed_recommendations, adjustments). adjustments only contains
    conflicts whose quoted decision was verified to exist in the documents.
    """
    if not recommendations or not passages:
        return recommendations, []
    numbered = "\n".join(f"{i}. {r}" for i, r in enumerate(recommendations))
    out = structured_call(
        RecCheckList,
        f"Today is {today()}. You check business recommendations against a firm's "
        "own internal decisions and policies. For EACH numbered recommendation, decide "
        "if it would violate a decision in the documents (e.g. a hiring freeze, a "
        "technology restriction, a pricing rule, a client policy). Only flag a conflict "
        "if a specific decision is clearly violated; copy that decision word for word "
        "into decision_quote. For conflicts, write a revised recommendation that keeps "
        "the same goal but complies. Return one entry per recommendation. "
        "Workarounds that defeat a decision's purpose (e.g. outsourcing roles a hiring "
        "freeze covers, splitting contracts to dodge a price cap) are conflicts too. "
        "Read each decision's SCOPE literally and only flag recommendations inside it: "
        "a freeze on 'non-technical roles' does not cover engineers or data scientists; "
        "a cap on 'fixed-price projects above Rs 40 lakh' does not cover smaller or "
        "time-and-material work. Never invent approvals, exceptions or conditions the "
        "documents do not state. When in doubt, it is not a conflict. "
        + OWN_DOCS_NOTE,
        f"Internal documents:\n{constraints_text(passages)}\n\nRecommendations:\n{numbered}",
        temperature=0.0,
    )
    fixed = list(recommendations)
    adjustments = []
    for c in out.checks:
        if not c.conflicts or not (0 <= c.index < len(fixed)):
            continue
        if not c.revised.strip() or not _quote_exists(c.decision_quote):
            continue  # unverifiable conflict: ignore rather than invent a problem
        adjustments.append({"original": fixed[c.index], "revised": c.revised.strip(),
                            "label": "Conflicts with",
                            "decision": c.decision_quote.strip(), "why": c.explanation.strip()})
        fixed[c.index] = c.revised.strip()
    return fixed, adjustments