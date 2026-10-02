"""
synthesizer.py — turns all agent outputs into one briefing, then audits it.

One agent ran  -> its own summary is the briefing (no extra LLM call).
Several ran    -> the LLM merges them into a single executive briefing.

The merge step can introduce NEW errors after every specialist was checked
(e.g. turning a client's complaint reminders into "strong interest"). So the
briefing is audited before anyone sees it:

1. Figure check (pure code)    numbers next to our own client/project names must
                               exist in the internal documents.
2. Decision check (LLM + code) EVERY action/bullet in the final briefing — whichever
                               specialist it came from — is checked against the
                               firm's own decisions; the conflicting decision must
                               be quoted and code verifies the quote exists.
3. Claim audit (LLM + code)    each statement about our own clients, projects or
                               decisions is checked against the documents.
4. Deadline check (pure code)  "by <date>" deadlines already in the past are flagged.
"""
from pydantic import BaseModel, Field

import re
from datetime import date

from agents.common import today
from agents.competitive_intel import parse_finding_date
from agents.constraints import check_and_fix, ground_internal_figures, internal_constraints
from core.agents_meta import AGENT_META, OWN_DOCS_NOTE
from core.llm import structured_call, text_call
from core.state import AgentState
from memory.vector_store import get_store


# ── Adjustments from the constraint check ────────────────────
def _adjustments(state: AgentState) -> list[dict]:
    return ((state.get("results") or {}).get("constraint_check") or {}).get("adjustments") or []


def adjustments_md(state: AgentState) -> str:
    adj = _adjustments(state)
    if not adj:
        return ""
    return "\n\n**Adjusted to fit internal decisions and documents**\n\n" + "\n".join(
        f"- ~~{a['original']}~~ → {a['revised']}  \n  _{a.get('label', 'Conflicts with')}:_ \"{a['decision']}\""
        for a in adj)


# ── Merge ────────────────────────────────────────────────────
def synthesize(state: AgentState) -> str:
    """Returns the briefing body (without the adjustments appendix)."""
    results = state.get("results") or {}
    plan = state.get("plan") or []
    ran = [(s["agent"], results[s["agent"]]) for s in plan if s["agent"] in results]
    if not ran:
        return "No specialist produced a result."
    if len(ran) == 1:
        return ran[0][1]["summary_md"]

    sections = "\n\n---\n\n".join(
        f"Specialist: {AGENT_META[name]['label']} (status: {r['status']})\n{r['summary_md']}"
        for name, r in ran
    )
    # Only the CORRECTED recommendations are shown to the writer. Showing the
    # removed originals let removed figures leak back into the briefing.
    adj = _adjustments(state)
    final_recs = ("\n\nFinal, already-verified versions of adjusted recommendations "
                  "(use these wording-for-wording):\n" + "\n".join(f"- {a['revised']}" for a in adj)
                  ) if adj else ""
    return text_call(
        "You write executive briefings for consulting partners. Merge the specialist "
        "outputs into one briefing in markdown with: a two-sentence bottom line, the "
        "key findings (grouped by theme, not by specialist), and a short prioritised "
        "action list written as a NUMBERED LIST (not a table), one action per line. "
        f"Today is {today()}: every deadline must be after today. "
        "Keep numbers, scores and links exactly as given. Do not add "
        "facts, figures or estimates that are not in the outputs. Describe each of our "
        "own clients exactly as the outputs do: overdue work, reminders, complaints and "
        "escalations are RISKS, never signs of interest. Overdue deliverables to a "
        "client must be resolved immediately and before proposing new paid work to that "
        "client. If a specialist needed more "
        "input, say what is missing. Never recommend anything the internal documents "
        "rule out. " + OWN_DOCS_NOTE,
        f"Original request: {state.get('request', '')}\n\n{sections}{final_recs}",
    )


# ── Audit ────────────────────────────────────────────────────
class ClaimCheck(BaseModel):
    text: str = Field(description="the statement, copied EXACTLY from the briefing, as short "
                                  "as possible while still containing the claim")
    supported: bool
    correction: str = Field(default="", description="for unsupported or misleading claims: a "
                            "replacement phrase that matches the documents; empty if supported")
    reason: str = Field(default="", description="one short sentence")


class ClaimAudit(BaseModel):
    claims: list[ClaimCheck]


AUDIT_SECTION_CHARS = 3500


def _sections(body: str, size: int = AUDIT_SECTION_CHARS) -> list[str]:
    """Split on blank lines into sections of roughly `size` characters."""
    parts, cur = [], ""
    for block in body.split("\n\n"):
        if cur and len(cur) + len(block) > size:
            parts.append(cur)
            cur = block
        else:
            cur = f"{cur}\n\n{block}" if cur else block
    if cur:
        parts.append(cur)
    return parts


def _audit_claims(body: str) -> tuple[str, list[dict]]:
    """
    Audit section by section: a long briefing in one call can exhaust the model's
    output budget and fail entirely. Each section is checked separately, so one
    failure only leaves that section unchecked (and it is reported).
    """
    store = get_store()
    if store.is_empty:
        return body, []
    documents = "\n\n".join(f"(file {c['source']})\n{c['text']}" for c in store.chunks)[:14000]
    fixed, notes, failed = body, [], 0
    for section in _sections(body):
        try:
            sec_fixed, sec_notes = _audit_section(section, documents)
        except Exception:
            failed += 1
            continue
        if sec_fixed != section:
            fixed = fixed.replace(section, sec_fixed, 1)
        notes += sec_notes
    if failed:
        raise PartialAudit(fixed, notes, failed)
    return fixed, notes


class PartialAudit(Exception):
    def __init__(self, body, notes, failed):
        super().__init__(f"{failed} section(s) unchecked")
        self.body, self.notes, self.failed = body, notes, failed


def _audit_section(body: str, documents: str) -> tuple[str, list[dict]]:
    audit = structured_call(
        ClaimAudit,
        f"Today is {today()}. You audit an executive briefing against the firm's own "
        "internal documents. List every statement about OUR OWN clients, projects, "
        "deliveries, people or internal decisions (ignore claims about competitors and "
        "the wider market). Mark it unsupported if the documents do not say it, or if "
        "it misrepresents them, e.g. calling complaint reminders 'strong interest', "
        "calling an unclear adoption 'retained', adding numbers the documents lack, or "
        "inventing requirements such as an 'exception' or 'approval' the documents never "
        "mention, stretching a decision beyond its literal scope, presenting a DRAFT "
        "proposal or plan as delivered or proven capability, or calling a client "
        "'no risk' when the documents show adoption problems, complaints or overdue work. "
        "For unsupported claims give a correction that states what the documents "
        "actually say, in a few words. Copy each claim exactly from the briefing. "
        + OWN_DOCS_NOTE,
        f"Internal documents:\n{documents}\n\nBriefing:\n{body}",
        temperature=0.0,
    )
    fixed, notes = body, []
    for c in audit.claims:
        phrase = c.text.strip()
        if c.supported or len(phrase) < 8 or phrase not in fixed:
            continue  # only act on real, locatable, unsupported claims
        replacement = c.correction.strip() or "[not supported by your documents]"
        fixed = fixed.replace(phrase, replacement)
        notes.append({"removed": phrase, "now": replacement, "reason": c.reason.strip()})
    return fixed, notes


TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}")
BULLET = re.compile(r"^(\s*(?:[-*+]|\d+[.)])\s+)(.*)$")


def _action_lines(lines: list[str]) -> list[tuple[int, str, str]]:
    """(line index, prefix to keep, text to check) for bullets and table body rows."""
    out = []
    for i, line in enumerate(lines):
        m = BULLET.match(line)
        if m and len(m.group(2)) > 15:
            out.append((i, m.group(1), m.group(2)))
        elif line.strip().startswith("|") and not TABLE_SEP.match(line):
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            if TABLE_SEP.match(nxt):
                continue  # header row
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if any(len(c) > 15 for c in cells):
                out.append((i, "|", " | ".join(cells)))
    return out[:40]


def decision_check(body: str) -> tuple[str, list[dict]]:
    """Every action in the FINAL briefing vs the firm's own decisions (quote-verified)."""
    passages = internal_constraints()
    lines = body.split("\n")
    candidates = _action_lines(lines)
    if not passages or not candidates:
        return body, []
    fixed, adjustments = check_and_fix([c[2] for c in candidates], passages)
    notes = []
    for (i, prefix, original), new in zip(candidates, fixed):
        if new == original:
            continue
        if prefix == "|":
            cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
            new_cells = [c.strip() for c in new.split("|")]
            if len(new_cells) != len(cells):            # revision lost the table shape:
                longest = max(range(len(cells)), key=lambda k: len(cells[k]))
                cells[longest] = new.replace("|", "/")   # put it in the main cell
                new_cells = cells
            lines[i] = "| " + " | ".join(new_cells) + " |"
        else:
            lines[i] = prefix + new
        adj = next((a for a in adjustments if a["revised"] == new), {})
        notes.append({"removed": original, "now": new,
                      "reason": f"conflicts with: \"{adj.get('decision', 'an internal decision')}\""})
    return "\n".join(lines), notes


DEADLINE = re.compile(
    r"\bby\s+(\d{1,2}\s+[A-Za-z]{3,9}\.?,?\s+20\d\d|[A-Za-z]{3,9}\.?\s+\d{1,2},?\s+20\d\d"
    r"|20\d\d-\d\d-\d\d)", re.I)


def deadline_check(body: str) -> tuple[str, list[dict]]:
    """Pure code: 'complete by 30 Sep 2026' when today is later gets flagged."""
    now = date.fromisoformat(today())
    notes = []

    def repl(m):
        d = parse_finding_date(m.group(1))
        if d and d < now:
            notes.append({"removed": m.group(0), "now": "by [date already passed — set a new date]",
                          "reason": f"{m.group(1)} is before today ({today()})"})
            return "by [date already passed — set a new date]"
        return m.group(0)
    return DEADLINE.sub(repl, body), notes


def audit_briefing(body: str) -> tuple[str, list[dict]]:
    """Figure check -> decision check -> claim audit -> deadline check."""
    lines = body.split("\n")
    fixed_lines, figure_notes = ground_internal_figures(lines)
    body = "\n".join(fixed_lines)
    notes = [{"removed": n["decision"], "now": "[Add: verified figure]",
              "reason": "figure not in your documents"} for n in figure_notes]
    failed = []
    for step in (decision_check, _audit_claims):
        try:
            body, step_notes = step(body)
            notes += step_notes
        except PartialAudit as p:          # some sections checked, some not
            body, notes = p.body, notes + p.notes
            failed.append(f"claim audit ({p.failed} section(s))")
        except Exception:
            failed.append(step.__name__)
    body, deadline_notes = deadline_check(body)
    notes += deadline_notes
    if failed:
        notes.append({"removed": "", "now": "", "reason": "check could not run: " + ", ".join(failed)})
    return body, notes


def audit_md(notes: list[dict]) -> str:
    real = [n for n in notes if n["removed"]]
    lines = []
    if real:
        lines.append("\n\n**Briefing check:** corrected statements that did not match your "
                     "documents or decisions\n")
        lines += [f"- ~~{n['removed']}~~ → {n['now']}" + (f" _({n['reason']})_" if n["reason"] else "")
                  for n in real]
    if any(not n["removed"] for n in notes):
        lines.append("\n\n⚠️ Part of the briefing check could not run this time; review "
                     "statements about your own clients and decisions before sharing.")
    return "\n".join(lines)