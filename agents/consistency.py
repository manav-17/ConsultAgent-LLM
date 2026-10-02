"""
consistency.py — makes the numbers in a proposal agree with each other.

LLMs are weak at arithmetic: a cost table whose rows add up to 35.5 under a
"Total: 35" line, or an executive summary promising 3 weeks while the plan
adds up to 5. Nothing here calls an LLM — it parses the markdown, does the
maths and fixes the text deterministically:

1. Cost tables      rows must sum to the Total row; the Total itself is pinned
                    to the price stated in the original draft. The difference
                    goes to the contingency/buffer row (or the largest row).
2. Payment tables   percentages must sum to 100; amounts = % x project price.
3. Timeline         the delivery duration stated in prose must match the sum of
                    the plan's phases (post-launch support is not counted).
4. Invented figures percentages that are not in the original draft and are not
                    part of the payment/contingency structure become
                    [Add: verified %]; example numbers inside [Add: ...]
                    placeholders become X so they can't be sent by mistake.

Every change is returned as a human-readable note so nothing is silent.
"""
import re

NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
DURATION = re.compile(r"(\d+(?:\.\d+)?)\s*(?:working\s+|business\s+|calendar\s+)?(day|week|month)s?", re.I)
WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
         "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twelve": 12}
WORD_DURATION = re.compile(r"\b(" + "|".join(WORDS) + r")\s+(day|week|month)s?\b", re.I)
SUPPORT_ROW = re.compile(r"support|warranty|hypercare|maintenance|post[- ]?(go[- ]?live|launch)", re.I)
PROSE_TOTAL_HINT = re.compile(r"deliver|complet|total|duration|end[- ]to[- ]end|go[- ]live|finish", re.I)
PROSE_SKIP_HINT = re.compile(r"prototype|pilot|phase|milestone|support|sprint|workshop", re.I)


# ── helpers ──────────────────────────────────────────────────
def _num(cell: str):
    if "[" in cell:  # placeholder
        return None
    m = NUM.search(cell.replace("**", ""))
    return float(m.group().replace(",", "")) if m else None


def _fmt(x: float) -> str:
    return str(int(round(x))) if abs(x - round(x)) < 1e-9 else f"{x:.2f}".rstrip("0").rstrip(".")


def _set_num(cell: str, value: float) -> str:
    return NUM.sub(_fmt(value), cell, count=1)


def _to_weeks(value: float, unit: str) -> float:
    unit = unit.lower()
    return value / 5 if unit == "day" else value * 4.33 if unit == "month" else value


def _parse_tables(lines: list[str]) -> list[dict]:
    """Find markdown tables: header line, separator line, body rows."""
    tables, i = [], 0
    while i < len(lines) - 1:
        if lines[i].strip().startswith("|") and re.match(r"^\s*\|?\s*:?-{2,}", lines[i + 1]):
            start = i
            j = i + 2
            while j < len(lines) and lines[j].strip().startswith("|"):
                j += 1
            header = _cells(lines[start])
            rows = list(range(start + 2, j))
            tables.append({"header": header, "header_line": start, "rows": rows})
            i = j
        else:
            i += 1
    return tables


def _cells(line: str) -> list[str]:
    parts = line.strip().strip("|").split("|")
    return [p.strip() for p in parts]


def _join(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _col(header: list[str], pattern: str, exclude: str = None):
    for idx, h in enumerate(header):
        if re.search(pattern, h, re.I) and not (exclude and re.search(exclude, h, re.I)):
            return idx
    return None


def project_price(draft: str):
    """The price stated in the ORIGINAL draft — a fact from the user, never changed."""
    m = re.search(r"(?:Rs\.?|INR|₹)\s*([\d,]+(?:\.\d+)?)\s*(lakh|lac|crore)?", draft, re.I)
    return float(m.group(1).replace(",", "")) if m else None


# ── checks ───────────────────────────────────────────────────
def fix_cost_tables(lines, price, notes):
    for t in _parse_tables(lines):
        if _col(t["header"], r"%|percent|share") is not None:
            continue  # payment table, handled separately
        c = _col(t["header"], r"cost|amount|price|fee|budget")
        if c is None:
            continue
        rows, total_row = [], None
        for r in t["rows"]:
            cells = _cells(lines[r])
            if c >= len(cells):
                continue
            if re.search(r"\btotal\b", cells[0], re.I):
                total_row = r
            elif _num(cells[c]) is not None:
                rows.append((r, cells, _num(cells[c])))
        if total_row is None or len(rows) < 2:
            continue
        total_cells = _cells(lines[total_row])
        stated = _num(total_cells[c]) if c < len(total_cells) else None
        if stated is None:
            continue
        target = stated
        # pin the total to the draft's price if the scales are comparable
        if price and 0.5 <= stated / price <= 2 and abs(stated - price) > 1e-6:
            total_cells[c] = _set_num(total_cells[c], price)
            lines[total_row] = _join(total_cells)
            notes.append(f"Cost table total changed from {_fmt(stated)} to {_fmt(price)} "
                         "to match the price in the original draft.")
            target = price
        row_sum = sum(v for _, _, v in rows)
        diff = target - row_sum
        if abs(diff) < 0.005:
            continue
        flex = next((x for x in rows if re.search(r"contingenc|buffer|reserve|misc", x[1][0], re.I)), None)
        flex = flex or max(rows, key=lambda x: x[2])
        r, cells, v = flex
        if v + diff < 0:
            notes.append(f"Cost rows add up to {_fmt(row_sum)} but the total is {_fmt(target)}; "
                         "could not fix automatically — review the cost table.")
            continue
        cells[c] = _set_num(cells[c], v + diff)
        lines[r] = _join(cells)
        notes.append(f"Cost rows added up to {_fmt(row_sum)}, not {_fmt(target)}: "
                     f"'{cells[0].replace('**', '')}' changed from {_fmt(v)} to {_fmt(v + diff)}.")


def fix_payment_tables(lines, price, notes):
    for t in _parse_tables(lines):
        p = _col(t["header"], r"%|percent|share")
        if p is None:
            continue
        a = _col(t["header"], r"amount|cost|value", exclude=r"%|percent")
        rows = []
        for r in t["rows"]:
            cells = _cells(lines[r])
            if p < len(cells) and not re.search(r"\btotal\b", cells[0], re.I) and _num(cells[p]) is not None:
                rows.append((r, cells))
        if len(rows) < 2:
            continue
        pct_sum = sum(_num(c[p]) for _, c in rows)
        if abs(pct_sum - 100) > 0.01:
            r, cells = rows[-1]
            new = _num(cells[p]) + (100 - pct_sum)
            if 0 < new < 100:
                notes.append(f"Payment percentages added up to {_fmt(pct_sum)}%: "
                             f"last milestone changed to {_fmt(new)}%.")
                cells[p] = _set_num(cells[p], new)
                lines[r] = _join(cells)
        if a is None or not price:
            continue
        changed = 0
        for r, cells in rows:
            if a >= len(cells) or _num(cells[a]) is None:
                continue
            expected = round(_num(cells[p]) / 100 * price, 2)
            if abs(_num(cells[a]) - expected) > 0.01 and 0.5 <= (_num(cells[a]) or 1) / max(expected, 1e-9) <= 2:
                cells[a] = _set_num(cells[a], expected)
                lines[r] = _join(cells)
                changed += 1
        if changed:
            notes.append(f"Recalculated {changed} payment amount(s) as % × project price ({_fmt(price)}).")


def plan_weeks(lines) -> float | None:
    for t in _parse_tables(lines):
        d = _col(t["header"], r"duration|time|days|weeks|length")
        if d is None:
            continue
        total, parsed = 0.0, 0
        for r in t["rows"]:
            cells = _cells(lines[r])
            if d >= len(cells) or SUPPORT_ROW.search(cells[0]) or re.search(r"\btotal\b", cells[0], re.I):
                continue
            m = DURATION.search(cells[d])
            if m:
                total += _to_weeks(float(m.group(1)), m.group(2))
                parsed += 1
        if parsed >= 2:
            return total
    return None


def fix_timeline_prose(lines, notes):
    weeks = plan_weeks(lines)
    if weeks is None:
        return
    target = max(1, round(weeks))
    for i, line in enumerate(lines):
        if line.strip().startswith("|"):
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", line):
            if not PROSE_TOTAL_HINT.search(sentence):
                continue
            for rx, is_word in ((DURATION, False), (WORD_DURATION, True)):
                m = rx.search(sentence)
                if not m:
                    continue
                # only skip if a sub-phase word sits right next to the duration
                # ("prototype within 2 weeks"), not anywhere in the sentence
                nearby = sentence[max(0, m.start() - 45): m.end() + 10]
                if PROSE_SKIP_HINT.search(nearby):
                    break
                value = WORDS[m.group(1).lower()] if is_word else float(m.group(1))
                stated = _to_weeks(value, m.group(2))
                if abs(stated - weeks) > 0.75:
                    new_sentence = sentence.replace(m.group(0), f"{target} weeks", 1)
                    lines[i] = lines[i].replace(sentence, new_sentence, 1)
                    notes.append(f"Text said '{m.group(0)}' but the implementation plan adds up "
                                 f"to about {target} weeks (excluding post-launch support): "
                                 "updated — confirm with your delivery team.")
                break


PCT = re.compile(r"[~≈<>+\-−–]?\s*\d+(?:\.\d+)?\s*%")
PAYMENT_HEADER = re.compile(r"payment|deposit|trigger|instal|invoice|milestone.*(amount|%)", re.I)
PAYMENT_LINE = re.compile(r"deposit|payment|invoice|instal|contingenc|buffer|reserve|on signing|"
                          r"acceptance|sign-?off|kick-?off|final delivery|go-?live", re.I)
EXAMPLE_PAREN = re.compile(r"\s*\((?:e\.g\.|eg|for example|such as)[^)]*\)", re.I)


def sanitize_placeholders(lines, notes):
    """[Add: client cut AHT by 28%] -> [Add: client cut AHT by X%]; drop '(e.g. ISO 27001)'."""
    changed = 0

    def clean(m):
        nonlocal changed
        inner = m.group(0)
        new = EXAMPLE_PAREN.sub("", inner)
        new = re.sub(r"\b(e\.g\.|for example|such as),?\s*[^\]]*", "", new, flags=re.I)
        new = re.sub(r"\d+(?:\.\d+)?", "X", new)
        if new != inner:
            changed += 1
        return new

    for i, line in enumerate(lines):
        lines[i] = re.sub(r"\[Add[^\]]*\]", clean, line)
    if changed:
        notes.append(f"Removed example numbers/names from {changed} placeholder(s) "
                     "so they can't be sent by mistake.")


def neutralize_unsupported_percentages(lines, draft, notes):
    """Percentages not in the draft and not part of payment/contingency become placeholders."""
    allowed = {re.sub(r"\s", "", m.group()).lstrip("~≈<>+-−–") for m in PCT.finditer(draft)}
    payment_rows = set()
    for t in _parse_tables(lines):
        if PAYMENT_HEADER.search(" ".join(t["header"])):
            payment_rows.update(t["rows"])
    replaced = 0
    for i, line in enumerate(lines):
        if i in payment_rows or PAYMENT_LINE.search(line):
            continue
        # do not touch text inside placeholders
        parts = re.split(r"(\[Add[^\]]*\])", line)
        for k, part in enumerate(parts):
            if part.startswith("[Add"):
                continue

            def repl(m):
                nonlocal replaced
                core = re.sub(r"\s", "", m.group()).lstrip("~≈<>+-−–")
                if core in allowed:
                    return m.group()
                replaced += 1
                lead = " " if m.group().startswith(" ") else ""
                return f"{lead}[Add: verified %]"
            parts[k] = PCT.sub(repl, part)
        lines[i] = "".join(parts)
    if replaced:
        notes.append(f"Replaced {replaced} percentage claim(s) that were not in the original "
                     "draft with [Add: verified %] placeholders.")


def make_consistent(text: str, draft: str) -> tuple[str, list[str]]:
    """Return (fixed_text, notes). Pure function: no LLM, fully repeatable."""
    lines = text.split("\n")
    notes: list[str] = []
    price = project_price(draft)
    fix_cost_tables(lines, price, notes)
    fix_payment_tables(lines, price, notes)
    fix_timeline_prose(lines, notes)
    sanitize_placeholders(lines, notes)
    neutralize_unsupported_percentages(lines, draft, notes)
    return "\n".join(lines), notes