"""
ConsultAgent-LLM — Streamlit front end.
Run from the project root:  python -m streamlit run app/streamlit_app.py

Two pages share one script:
  start  a short introduction with an example run
  desk   the working app: request, live relay timeline, audited briefing
"""
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # faiss + torch on macOS
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import markdown as md
import pandas as pd
import streamlit as st

from core.agents_meta import AGENT_META, SUPERVISOR_COLOR
from core.config import SAMPLES_DIR, provider_ready, search_mode
from graph.workflow import build_graph, initial_state, stream_run
from memory.vector_store import get_store, load_file_text

st.set_page_config(page_title="ConsultAgent-LLM", page_icon="🛰️", layout="wide",
                   initial_sidebar_state="expanded")

PAGE = st.session_state.setdefault("page", "start")

# ── Design tokens ────────────────────────────────────────────
INK, SLATE, RULE, CANVAS, PRUSSIAN = "#14181F", "#5B6575", "#D8DEE6", "#F3F5F8", "#10233F"


def html(markup: str):
    """Render HTML. Strings are built without leading indentation: indented
    HTML inside st.markdown is treated as a code block."""
    st.markdown(markup, unsafe_allow_html=True)


CSS = (
    "<style>"
    "@import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700"
    "&family=Newsreader:opsz,wght@6..72,400;6..72,500&display=swap');"
    # base type
    ".stApp p,.stApp li,.stApp label,.stApp textarea,.stApp input,.stApp button p,"
    ".stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp td,.stApp th"
    "{font-family:'Instrument Sans',system-ui,-apple-system,sans-serif;}"
    f".stApp{{color:{INK};}}"
    # page frame and Streamlit chrome
    ".block-container{max-width:1120px;padding-top:1.6rem;padding-bottom:4rem;}"
    "[data-testid='stToolbar']{display:none !important;}"
    "header[data-testid='stHeader']{background:transparent;}"
    "[data-testid='stDecoration']{display:none;}"
    # buttons and inputs
    ".stButton button,.stDownloadButton button{border-radius:8px;font-weight:500;"
    "padding:0.5rem 1rem;}"
    ".stTextArea textarea{font-size:1rem;line-height:1.5;}"
    "button:focus-visible,textarea:focus-visible{outline:2px solid #10233F;outline-offset:2px;}"
    # top bar
    ".wordmark{font-weight:700;font-size:1.12rem;letter-spacing:-0.01em;margin:0;line-height:1.2;}"
    f".wordmark-sub{{color:{SLATE};font-size:0.9rem;margin:0.1rem 0 0;}}"
    f".topbar-rule{{border:0;border-top:1px solid {RULE};margin:0.9rem 0 0;}}"
    # headings: beat Streamlit's own heading styles and hide its anchor icons
    ".stApp h1.hero-title,.stApp h1.desk-title,.stApp h2.section-title{padding:0 !important;}"
    "[data-testid='stHeaderActionElements']{display:none !important;}"
    ".stApp h2.section-title{font-size:1.4rem !important;font-weight:600 !important;"
    "letter-spacing:-0.015em;line-height:1.3 !important;}"
    ".stApp h1.desk-title{font-size:1.9rem !important;font-weight:600 !important;line-height:1.2 !important;}"
    ".stApp h1.hero-title{font-weight:600 !important;line-height:1.04 !important;"
    "margin:2.4rem 0 1.2rem !important;}"
    ".stApp h2.section-title{margin:0 0 0.5rem !important;}"
    ".stApp h1.desk-title{margin:1.6rem 0 0.4rem !important;}"
    ".stApp p.step-num{font-size:2.4rem !important;font-weight:600;letter-spacing:-0.03em;"
    f"color:{PRUSSIAN};line-height:1 !important;margin:0 0 0.7rem !important;}}"
    ".stApp p.step-title{font-weight:600;margin:0 0 0.3rem !important;}"
    ".stApp p.section-sub,.stApp p.desk-sub,.stApp p.hero-sub{margin-top:0 !important;}"
    ".hero-spacer{height:2.4rem;}"
    "@media (max-width:760px){.hero-spacer{height:1rem;}}"
    # start page
    ".hero-title{font-size:clamp(2.3rem,4.8vw,3.5rem);font-weight:600;letter-spacing:-0.035em;"
    "line-height:1.04;margin:2.6rem 0 1.1rem;max-width:15ch;}"
    f".hero-sub{{font-size:1.14rem;line-height:1.55;color:{SLATE};max-width:50ch;margin:0 0 1.6rem;}}"
    f".panel{{background:#FFFFFF;border:1px solid {RULE};border-radius:10px;padding:1.3rem 1.5rem 0.6rem;}}"
    f".panel-title{{font-weight:600;font-size:0.95rem;color:{SLATE};margin:0 0 0.4rem;}}"
    f".section{{border-top:1px solid {RULE};margin-top:3.4rem;padding-top:1.7rem;}}"
    ".section-title{font-size:1.4rem;font-weight:600;letter-spacing:-0.015em;margin:0 0 0.3rem;}"
    f".section-sub{{color:{SLATE};margin:0 0 1.4rem;max-width:60ch;}}"
    ".agents{display:grid;grid-template-columns:1fr 1fr;gap:1.1rem 3rem;}"
    ".agent{display:flex;gap:0.8rem;align-items:baseline;}"
    ".agent .dot{transform:translateY(1px);}"
    ".agent-name{font-weight:600;margin:0;}"
    f".agent-job{{color:{SLATE};margin:0.15rem 0 0;line-height:1.45;}}"
    ".steps{display:grid;grid-template-columns:repeat(4,1fr);gap:2rem;}"
    f".step-num{{font-size:2.2rem;font-weight:600;letter-spacing:-0.03em;color:{PRUSSIAN};"
    "line-height:1;margin:0 0 0.6rem;}"
    ".step-title{font-weight:600;margin:0 0 0.3rem;}"
    f".step-text{{color:{SLATE};margin:0;line-height:1.5;}}"
    ".checks{display:grid;grid-template-columns:1fr 1fr;gap:0.9rem 3rem;}"
    f".check{{border-left:2px solid {PRUSSIAN};padding:0.1rem 0 0.1rem 0.9rem;line-height:1.5;margin:0;}}"
    f".footer{{color:{SLATE};font-size:0.9rem;border-top:1px solid {RULE};margin-top:3.4rem;padding-top:1.2rem;}}"
    # desk
    ".desk-title{font-size:1.9rem;font-weight:600;letter-spacing:-0.025em;margin:1.6rem 0 0.3rem;}"
    f".desk-sub{{color:{SLATE};margin:0 0 1.3rem;max-width:62ch;}}"
    f".label{{font-weight:600;font-size:0.95rem;margin:1.1rem 0 0.5rem;}}"
    f".results-rule{{border:0;border-top:1px solid {RULE};margin:2.2rem 0 1.4rem;}}"
    ".col-title{font-size:1.1rem;font-weight:600;margin:0 0 0.5rem;}"
    # relay timeline (start page example and live runs)
    ".dot{width:10px;height:10px;border-radius:50%;display:inline-block;flex:none;}"
    ".relay{border-left:2px solid #C9D1DC;margin:0.4rem 0 0.4rem 0.45rem;padding-left:1.1rem;}"
    ".relay-row{position:relative;padding:0.35rem 0 0.75rem;}"
    f".relay-row .dot{{position:absolute;left:-1.55rem;top:0.62rem;width:12px;height:12px;"
    f"box-shadow:0 0 0 3px {CANVAS};}}"
    ".panel .relay-row .dot{box-shadow:0 0 0 3px #FFFFFF;}"
    ".relay-head{font-weight:600;font-size:0.98rem;}"
    f".relay-time{{color:{SLATE};font-weight:400;font-size:0.85rem;margin-left:0.4rem;}}"
    ".relay-detail{color:#3B4452;font-size:0.9rem;max-width:70ch;line-height:1.45;}"
    ".relay-flag{font-size:0.8rem;font-weight:600;margin-left:0.4rem;}"
    # briefing memo
    ".memo{font-family:'Newsreader',Georgia,serif;font-size:1.1rem;line-height:1.62;"
    f"color:{INK};background:#FFFFFF;border:1px solid {RULE};"
    "border-radius:10px;padding:1.6rem 2rem;overflow-x:auto;}"
    ".memo h1,.memo h2,.memo h3,.memo h4{font-family:'Instrument Sans',sans-serif;"
    "font-size:1.12rem;margin:1.2rem 0 0.4rem;}"
    f".memo a{{color:{PRUSSIAN};}}"
    f".memo table{{border-collapse:collapse;margin:0.6rem 0;font-size:0.98rem;}}"
    f".memo th,.memo td{{border:1px solid {RULE};padding:0.4rem 0.6rem;text-align:left;vertical-align:top;}}"
    ".badge{display:inline-block;padding:0.12rem 0.55rem;border-radius:4px;font-size:0.8rem;"
    "font-weight:600;color:#FFFFFF;}"
    # small screens
    "@media (max-width:760px){.agents,.checks{grid-template-columns:1fr;}"
    ".steps{grid-template-columns:1fr 1fr;}.memo{padding:1.1rem 1.1rem;}}"
    "@media (prefers-reduced-motion:reduce){*{transition:none !important;animation:none !important;}}"
    "</style>"
)
html(CSS)

if PAGE == "start":   # the start page has no sidebar
    html("<style>section[data-testid='stSidebar'],[data-testid='stSidebarCollapsedControl'],"
         "[data-testid='collapsedControl']{display:none !important;}</style>")

VERDICT_COLORS = {"supported": "#2E7D32", "contradicted": "#B3261E", "unverifiable": "#8A6D00"}
RISK_COLORS = {"healthy": "#2E7D32", "watch": "#B26A00", "critical": "#B3261E"}
STATUS_FLAGS = {"needs_input": ("needs input", "#B26A00"), "error": ("failed", "#B3261E")}


# ── Cached resources ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_graph():
    return build_graph()


def read_sample(name: str) -> str:
    return (SAMPLES_DIR / name).read_text()


SAMPLE_DOC = "Zenith_Consulting_Q3_Review.md"


def load_sample_memory():
    """Load the sample firm's internal review into team memory (once)."""
    store = get_store()
    path = SAMPLES_DIR / SAMPLE_DOC
    if path.exists() and path.name not in set(store.stats()["sources"]):
        n = store.add_document(path.read_text(), path.name)
        st.session_state["memory_msg"] = f"Added the sample review ({n} passages)."
    else:
        st.session_state["memory_msg"] = "The sample review is already in memory."


# ── Navigation ───────────────────────────────────────────────
def go(page: str):
    st.session_state["page"] = page


def start_with_samples():
    load_sample_memory()
    go("desk")


# ── Sample scenarios ─────────────────────────────────────────
SCENARIOS = {
    "Check client health": (
        "Which of our clients are at risk of churning, and what should we do this week?",
        "client_emails.txt"),
    "Improve a proposal": (
        "Score this proposal for Brightleaf Insurance and improve it before we send it.",
        "proposal_draft.txt"),
    "Fact-check AI text": (
        "Fact-check this AI-written paragraph before it goes into our newsletter.",
        "ai_generated_text.txt"),
    "Track competitors": (
        "We are an AI-native consulting firm in Mumbai. Track recent AI moves by Accenture "
        "and Infosys and tell us where we can win.", None),
    "Ask team memory": (
        "Which client commitments are overdue and who owns them?", None),
    "Proposal, then fact-check": (
        "Improve this Brightleaf proposal, then fact-check every claim in the improved version.",
        "proposal_draft.txt"),
}


def load_scenario(name: str):
    request, file = SCENARIOS[name]
    st.session_state["request"] = request
    st.session_state["attachment"] = read_sample(file) if file else ""
    st.session_state["route"] = "Let the supervisor decide"


# ── Text helpers ─────────────────────────────────────────────
def safe(text) -> str:
    """Streamlit renders $...$ as LaTeX maths; '$1 billion ... $2B' would turn into a formula."""
    return str(text).replace("$", "\\$")


def tidy_markdown(text: str) -> str:
    """Markdown tables need a blank line before them; LLMs often omit it."""
    out = []
    for line in text.split("\n"):
        if line.lstrip().startswith("|") and out and out[-1].strip() and \
                not out[-1].lstrip().startswith("|"):
            out.append("")
        out.append(line)
    return "\n".join(out)


# ── Relay timeline ───────────────────────────────────────────
def clean_detail(text: str) -> str:
    """Timeline lines are plain text: drop markdown marks and list bullets."""
    import re
    text = re.sub(r"[*_`#]|~~", "", str(text)).strip()
    return re.sub(r"^[-•]\s*", "", text)


def relay_html(trace: list[dict]) -> str:
    rows = []
    for ev in trace:
        node = ev["node"]
        if node == "supervisor":
            label, color = "Supervisor", SUPERVISOR_COLOR
        elif node == "synthesizer":
            label, color = "Briefing", SUPERVISOR_COLOR
        elif node == "constraint_check":
            label, color = "Constraint check", "#5F6B7A"
        else:
            label, color = AGENT_META[node]["label"], AGENT_META[node]["color"]
        flag = ""
        if ev["status"] in STATUS_FLAGS:
            text, fc = STATUS_FLAGS[ev["status"]]
            flag = f'<span class="relay-flag" style="color:{fc}">{text}</span>'
        rows.append(
            f'<div class="relay-row"><i class="dot" style="background:{color}"></i>'
            f'<div class="relay-head">{escape(label)}<span class="relay-time">{ev["seconds"]}s</span>{flag}</div>'
            f'<div class="relay-detail">{escape(clean_detail(ev["detail"]))}</div></div>')
    return '<div class="relay">' + "".join(rows) + "</div>"




# ── Shared top bar ───────────────────────────────────────────
def top_bar(button_label: str, target: str, primary: bool):
    left, right = st.columns([5, 1.3], vertical_alignment="center")
    with left:
        html('<p class="wordmark">ConsultAgent-LLM</p>'
             '<p class="wordmark-sub">Multi-agent consulting copilot</p>')
    with right:
        st.button(button_label, on_click=go, args=(target,), width="stretch",
                  type="primary" if primary else "secondary", key=f"nav_{target}")
    html('<hr class="topbar-rule">')


# ══════════════════════════════════════════════════════════════
# START PAGE
# ══════════════════════════════════════════════════════════════
AGENT_JOBS = {
    "second_brain": "Answers from your own documents, with sources, owners and dates.",
    "client_health": "Reads client emails and scores each relationship, with a reply draft.",
    "proposal": "Scores a proposal, rewrites the weak parts and removes invented claims.",
    "hallucination": "Checks each claim in a text against the web and corrects it.",
    "competitive_intel": "Tracks what competitors did in the last year and where you can win.",
}

EXAMPLE_TRACE = [
    {"node": "supervisor", "status": "ok", "seconds": 1.6,
     "detail": "Plan: Competitor Watch, then Second Brain"},
    {"node": "competitive_intel", "status": "ok", "seconds": 23.2,
     "detail": "9 dated findings on Accenture and Infosys, all with sources"},
    {"node": "second_brain", "status": "ok", "seconds": 42.2,
     "detail": "Matched the findings to our Q3 internal review"},
    {"node": "constraint_check", "status": "ok", "seconds": 29.6,
     "detail": "1 move broke the API-only rule, so it was rewritten"},
    {"node": "synthesizer", "status": "ok", "seconds": 102.9,
     "detail": "Briefing ready; 2 statements corrected against our documents"},
]

STEPS = [
    ("Plan", "The supervisor reads your request and decides which specialists work on it, "
             "in what order."),
    ("Work", "Specialists research, score and write, passing work to each other when the "
             "task needs it."),
    ("Check", "Recommendations are checked against your decisions, and figures against "
              "your documents."),
    ("Brief", "You get one briefing, with every correction listed so nothing is hidden."),
]

CHECKS = [
    "A score only counts if the model quotes the text that justifies it.",
    "Numbers about your clients must appear next to that client in your documents.",
    "Recommendations can't break your own decisions, such as price caps or hiring rules.",
    "Totals, percentages and timelines in a proposal must add up.",
    "News older than 12 months doesn't count towards a competitor's threat level.",
    "When a check can't run, the briefing tells you instead of looking checked.",
]


def render_start():
    top_bar("Open the desk", "desk", primary=False)

    hero, example = st.columns([1.15, 1], gap="large")
    with hero:
        html('<h1 class="hero-title">Ask once. Get a briefing you can check.</h1>'
             '<p class="hero-sub">A supervisor hands your request to five AI specialists, '
             "then checks their work against your firm's own documents and decisions "
             "before you read a word.</p>")
        b1, b2 = st.columns(2)
        b1.button("Start a request", type="primary", on_click=go, args=("desk",),
                  width="stretch", key="hero_start")
        b2.button("Try with sample documents", on_click=start_with_samples,
                  width="stretch", key="hero_samples")
    with example:
        html('<div class="hero-spacer"></div><div class="panel">'
             '<p class="panel-title">An example run</p>'
             + relay_html(EXAMPLE_TRACE) + "</div>")

    agents = "".join(
        f'<div class="agent"><i class="dot" style="background:{m["color"]}"></i><div>'
        f'<p class="agent-name">{escape(m["label"])}</p>'
        f'<p class="agent-job">{escape(AGENT_JOBS[k])}</p></div></div>'
        for k, m in AGENT_META.items())
    html('<div class="section"><h2 class="section-title">The specialists</h2>'
         '<p class="section-sub">Each one is a small workflow of its own. The supervisor '
         'decides who is needed for your request.</p>'
         f'<div class="agents">{agents}</div></div>')

    steps = "".join(
        f'<div><p class="step-num">{i}</p><p class="step-title">{t}</p>'
        f'<p class="step-text">{d}</p></div>' for i, (t, d) in enumerate(STEPS, 1))
    html('<div class="section"><h2 class="section-title">How a request flows</h2>'
         '<p class="section-sub">The same four steps happen on every request, and the '
         'desk shows each one live as it runs.</p>'
         f'<div class="steps">{steps}</div></div>')

    checks = "".join(f'<p class="check">{escape(c)}</p>' for c in CHECKS)
    html('<div class="section"><h2 class="section-title">What gets checked</h2>'
         '<p class="section-sub">The model writes. Code checks the parts that must be true.</p>'
         f'<div class="checks">{checks}</div></div>')

    html('<div class="footer">Built with LangGraph, Groq, FAISS and Streamlit. '
         'Companies in the sample documents are fictional.</div>')


if PAGE == "start":
    render_start()
    st.stop()


# ══════════════════════════════════════════════════════════════
# DESK
# ══════════════════════════════════════════════════════════════
ok, msg = provider_ready()

# ── Sidebar: team memory ─────────────────────────────────────
with st.sidebar:
    html('<p class="wordmark">Team memory</p>'
         '<p class="wordmark-sub">Your documents. Answers and checks use these.</p>')
    store = get_store()
    stats = store.stats()
    st.metric("Documents in memory", stats["documents"],
              help=f"{stats['chunks']} searchable passages")

    uploads = st.file_uploader("Add documents", type=["txt", "md", "pdf", "docx"],
                               accept_multiple_files=True)
    if uploads and st.button("Add to memory", type="primary", width="stretch"):
        with st.spinner("Reading and indexing…"):
            total = 0
            for f in uploads:
                try:
                    total += store.add_document(load_file_text(f.name, f.getvalue()), f.name)
                except Exception as e:
                    st.error(f"{f.name}: {e}")
        st.session_state["memory_msg"] = f"Added {total} passages."
        st.rerun()

    st.button("Load sample documents", width="stretch", on_click=load_sample_memory)
    if stats["documents"] and st.button("Clear memory", width="stretch"):
        store.clear()
        st.rerun()
    if st.session_state.get("memory_msg"):
        st.success(st.session_state.pop("memory_msg"))
    if stats["sources"]:
        with st.expander("Stored documents"):
            for s in stats["sources"]:
                st.caption(s)
    elif not stats["documents"]:
        st.info("Memory is empty. Without documents, recommendations can't be checked "
                "against your firm's decisions.")

    st.divider()
    if ok:
        st.caption(f"Model: {msg}")
    else:
        st.error(msg)
    st.caption(f"Web search: {search_mode()}")


# ── Desk header and request form ─────────────────────────────
top_bar("Back to start", "start", primary=False)
html('<h1 class="desk-title">New request</h1>'
     '<p class="desk-sub">Start from an example or write your own. Attach emails, a '
     'proposal or text to check when the task needs it.</p>')

html('<p class="label">Examples</p>')
cols = st.columns(3)
for i, name in enumerate(SCENARIOS):
    cols[i % 3].button(name, on_click=load_scenario, args=(name,), width="stretch",
                       key=f"scenario_{i}")

st.text_area("Your request", key="request", height=100,
             placeholder="e.g. Which clients need attention this week?")

with st.expander("Attach text: emails, a proposal or a draft to check",
                 expanded=bool(st.session_state.get("attachment"))):
    st.text_area("Attachment", key="attachment", height=220, label_visibility="collapsed")
    attached_file = st.file_uploader("Or attach a file", type=["txt", "md", "pdf", "docx"],
                                     key="attach_file")

route_options = ["Let the supervisor decide"] + [m["label"] for m in AGENT_META.values()]
c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
route = c1.selectbox("Who should handle it?", route_options, key="route")
run_clicked = c2.button("Run agents", type="primary", width="stretch", disabled=not ok)
if not ok:
    st.caption("Add your Groq API key to the .env file to run requests.")


# ── Run ──────────────────────────────────────────────────────
if run_clicked:
    request = (st.session_state.get("request") or "").strip()
    attachment = (st.session_state.get("attachment") or "").strip()
    if attached_file is not None:
        try:
            attachment = (attachment + "\n\n" + load_file_text(
                attached_file.name, attached_file.getvalue())).strip()
        except Exception as e:
            st.error(str(e))
    if not request:
        st.warning("Describe what you need first, or pick an example above.")
    else:
        forced = next((k for k, m in AGENT_META.items() if m["label"] == route), None)
        state = initial_state(request, attachment, forced)
        final = state
        with st.status("Supervisor is planning…", expanded=True) as status:
            live = st.empty()
            try:
                for node, merged in stream_run(get_graph(), state):
                    final = merged
                    live.markdown(relay_html(merged.get("trace", [])), unsafe_allow_html=True)
                    nxt = merged.get("plan", [])[merged.get("cursor", 0):]
                    if node in ("synthesizer", "constraint_check"):
                        status.update(label="Writing the briefing…")
                    elif nxt:
                        status.update(label=f"{AGENT_META[nxt[0]['agent']]['label']} is working…")
                    else:
                        status.update(label="Checking against internal decisions…")
                status.update(label="Done", state="complete", expanded=False)
            except Exception as e:
                status.update(label="Run failed", state="error")
                st.error(f"{type(e).__name__}: {e}")
        st.session_state["run"] = final


# ── Result renderers ─────────────────────────────────────────
def render_second_brain(r):
    st.markdown(safe(r["answer"]["answer_md"]))
    st.caption(f"Confidence: {r['answer']['confidence']}. Searched with: "
               + "; ".join(r.get("queries", [])))
    if r["answer"]["items"]:
        st.dataframe(pd.DataFrame(r["answer"]["items"]), width="stretch", hide_index=True)
    if r["answer"].get("gaps"):
        st.info("Not covered by your documents: " + "; ".join(r["answer"]["gaps"]))
    with st.expander("Source passages"):
        for s in r["sources"]:
            st.markdown(f"**[{s['label']}] {s['source']}**, similarity {s['score']}")
            st.text(s["text"])


def render_fact_check(r):
    c = st.columns(4)
    c[0].metric("Trust score", f"{r['trust_score']}/100")
    c[1].metric("Supported", r["counts"]["supported"])
    c[2].metric("Contradicted", r["counts"]["contradicted"])
    c[3].metric("Unverifiable", r["counts"]["unverifiable"])
    for chk in r["checks"]:
        with st.container(border=True):
            color = VERDICT_COLORS[chk["verdict"]]
            st.markdown(f'<span class="badge" style="background:{color}">{chk["verdict"]}</span> '
                        f'&nbsp;{escape(chk["claim"])}', unsafe_allow_html=True)
            st.markdown(safe(chk["explanation"]))
            if chk.get("correction"):
                st.markdown(f"**Correction:** {safe(chk['correction'])}")
            if chk.get("source_url"):
                st.caption(f"[Source]({chk['source_url']}) · confidence {chk['confidence']}%")
    if r.get("corrected_text"):
        with st.expander("Corrected text", expanded=True):
            st.text_area("Corrected", r["corrected_text"], height=220, label_visibility="collapsed")


def render_client_health(r):
    for cl in r["clients"]:
        with st.container(border=True):
            color = RISK_COLORS[cl["risk"]]
            st.markdown(f'### {escape(cl["client"])} &nbsp;<span class="badge" '
                        f'style="background:{color}">{cl["risk"]}</span>', unsafe_allow_html=True)
            m = st.columns(3)
            m[0].metric("Health", f"{cl['score']}/100")
            m[1].metric("Churn risk", f"{cl['churn_risk']}%")
            m[2].metric("Tone", cl["tone"])
            st.progress(cl["score"] / 100)
            if cl["hidden_concerns"]:
                st.write("**Reading between the lines:** " + "; ".join(cl["hidden_concerns"]))
            rec = cl["recommendation"]
            st.write("**Do next**")
            for a in rec["actions"]:
                st.write(f"- {a}")
            if rec.get("talking_points"):
                st.write("**Talking points:** " + " | ".join(rec["talking_points"]))
            if rec.get("suggested_reply"):
                with st.expander("Suggested reply"):
                    st.text_area("Reply", rec["suggested_reply"], height=180,
                                 key=f"reply_{cl['client']}", label_visibility="collapsed")
            with st.expander("How this score was calculated"):
                for line in cl["breakdown"]:
                    st.write(f"- {line}")
                st.write("**Evidence:** " + " / ".join(f'"{q}"' for q in cl["evidence"]))


def render_competitors(r):
    s = r["strategy"]
    c = st.columns(3)
    for col, title, items in [(c[0], "Threats", s["threats"]),
                              (c[1], "Opportunities", s["opportunities"]),
                              (c[2], "Recommended moves", s["recommended_moves"])]:
        col.markdown(f"**{title}**")
        for it in items:
            col.markdown(f"- {safe(it)}")
    adj = (st.session_state.get("run", {}).get("results", {}).get("constraint_check") or {}).get("adjustments") or []
    if adj:
        with st.expander(f"Adjusted to fit internal decisions and documents ({len(adj)})", expanded=True):
            for a in adj:
                st.markdown(f"~~{safe(a['original'])}~~  \n**Now:** {safe(a['revised'])}  \n"
                            f"_{a.get('label', 'Conflicts with')}:_ \"{a['decision']}\"")
    for rep in r["reports"]:
        with st.expander(f"{rep['competitor']} — threat {rep['threat_level']}",
                         expanded=rep["threat_level"] == "high"):
            st.markdown(safe(rep["summary"]))
            for f in rep["findings"]:
                when = f" ({f['date']})" if f.get("date") else ""
                st.markdown(f"**{safe(f['headline'])}**{when} · {f['signal']}, {f['importance']} importance  \n"
                            f"{safe(f['evidence'])}  \n_So what:_ {safe(f['implication'])} [Source]({f['source_url']})")
            if rep.get("older_findings"):
                st.caption("Older than 12 months (not used for the threat level): " + "; ".join(
                    f"{safe(o['headline'])} ({o.get('date') or 'undated'})" for o in rep["older_findings"]))


def render_proposal(r):
    first, last = r["history"][0], r["history"][-1]
    c = st.columns(3)
    c[0].metric("Win probability", f"{last['win_probability']}%",
                delta=f"{last['win_probability'] - first['win_probability']} pts")
    c[1].metric("Rubric total", last["total"], delta=round(last["total"] - first["total"], 1))
    c[2].metric("Rewrite rounds", len(r["history"]) - 1)
    if len(r["history"]) > 1:
        st.line_chart(pd.DataFrame(r["history"]).set_index("round")["win_probability"])
    st.dataframe(pd.DataFrame(r["dimensions"]), width="stretch", hide_index=True)
    st.caption("Win probability is a rubric-based estimate, not a trained prediction.")
    if r.get("improved"):
        with st.expander("Improved proposal", expanded=True):
            st.markdown(safe(tidy_markdown(r["improved"])))
            st.download_button("Download improved proposal", r["improved"],
                               file_name="proposal_improved.md")


RENDERERS = {"second_brain": render_second_brain, "hallucination": render_fact_check,
             "client_health": render_client_health, "competitive_intel": render_competitors,
             "proposal": render_proposal}


# ── Results ──────────────────────────────────────────────────
run = st.session_state.get("run")
if run and run.get("final_report"):
    html('<hr class="results-rule">')
    left, right = st.columns([1, 2.2], gap="large")
    with left:
        html('<p class="col-title">How the work was routed</p>')
        if run.get("plan_reasoning"):
            st.caption(run["plan_reasoning"])
        html(relay_html(run.get("trace", [])))
    with right:
        html('<p class="col-title">Briefing</p>')
        html(f'<div class="memo">{md.markdown(tidy_markdown(run["final_report"]), extensions=["tables"])}</div>')

    ran = [s["agent"] for s in run.get("plan", []) if s["agent"] in run.get("results", {})]
    if ran:
        html('<hr class="results-rule"><p class="col-title">Specialist detail</p>')
        tabs = st.tabs([AGENT_META[a]["label"] for a in ran])
        for tab, agent in zip(tabs, ran):
            with tab:
                res = run["results"][agent]
                if res["status"] == "ok":
                    RENDERERS[agent](res)
                else:
                    st.markdown(safe(res["summary_md"]))
elif not run:
    html('<hr class="results-rule">')
    st.caption("Your briefing appears here. Each step shows up live while the agents work.")