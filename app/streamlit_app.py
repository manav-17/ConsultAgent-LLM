"""
ConsultAgent-LLM — Streamlit front end.
Run from the project root:  streamlit run app/streamlit_app.py
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

st.set_page_config(page_title="ConsultAgent-LLM", page_icon="🛰️", layout="wide")

# ── Styling ──────────────────────────────────────────────────
# HTML strings in this file are built WITHOUT leading indentation:
# indented HTML inside st.markdown is treated as a code block.
st.markdown(
    "<style>"
    "@import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700"
    "&family=Newsreader:opsz,wght@6..72,400;6..72,500&display=swap');"
    ".stApp p,.stApp li,.stApp label,.stApp textarea,.stApp input,.stApp button p,"
    ".stApp h1,.stApp h2,.stApp h3,.stApp h4{font-family:'Instrument Sans',system-ui,sans-serif;}"
    ".stApp h1{font-weight:700;letter-spacing:-0.025em;}"
    ".block-container{max-width:1180px;padding-top:2.2rem;}"
    ".lede{font-size:1.08rem;color:#3B4452;max-width:62ch;margin:-0.4rem 0 1.2rem;}"
    ".roster{display:flex;flex-wrap:wrap;gap:0.5rem 1.4rem;margin:0 0 1.6rem;}"
    ".roster span{display:inline-flex;align-items:center;gap:0.45rem;font-size:0.92rem;color:#14181F;}"
    ".dot{width:10px;height:10px;border-radius:50%;display:inline-block;flex:none;}"
    ".relay{border-left:2px solid #C9D1DC;margin:0.4rem 0 0.4rem 0.45rem;padding-left:1.1rem;}"
    ".relay-row{position:relative;padding:0.35rem 0 0.75rem;}"
    ".relay-row .dot{position:absolute;left:-1.55rem;top:0.62rem;width:12px;height:12px;"
    "box-shadow:0 0 0 3px #F3F5F8;}"
    ".relay-head{font-weight:600;font-size:0.98rem;}"
    ".relay-time{color:#5B6575;font-weight:400;font-size:0.85rem;margin-left:0.4rem;}"
    ".relay-detail{color:#3B4452;font-size:0.9rem;max-width:70ch;}"
    ".relay-flag{font-size:0.8rem;font-weight:600;margin-left:0.4rem;}"
    ".memo{font-family:'Newsreader',Georgia,serif;font-size:1.12rem;line-height:1.62;"
    "color:#14181F;max-width:68ch;background:#FFFFFF;border:1px solid #D8DEE6;"
    "border-radius:6px;padding:1.6rem 2rem;}"
    ".memo h1,.memo h2,.memo h3{font-family:'Instrument Sans',sans-serif;font-size:1.15rem;"
    "margin:1.1rem 0 0.4rem;}"
    ".memo a{color:#10233F;}"
    ".badge{display:inline-block;padding:0.12rem 0.55rem;border-radius:4px;font-size:0.8rem;"
    "font-weight:600;color:#FFFFFF;}"
    "</style>",
    unsafe_allow_html=True,
)

VERDICT_COLORS = {"supported": "#2E7D32", "contradicted": "#B3261E", "unverifiable": "#8A6D00"}
RISK_COLORS = {"healthy": "#2E7D32", "watch": "#B26A00", "critical": "#B3261E"}
STATUS_FLAGS = {"needs_input": ("needs input", "#B26A00"), "error": ("failed", "#B3261E")}


# ── Cached resources ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_graph():
    return build_graph()


def read_sample(name: str) -> str:
    return (SAMPLES_DIR / name).read_text()


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
        "We are an AI-native consulting firm in Mumbai. Track recent AI moves by Accenture, "
        "Fractal Analytics and Tiger Analytics and tell us where we can win.", None),
    "Ask team memory": (
        "What commitments have we made to clients this year, and which are overdue?", None),
    "Proposal, then fact-check": (
        "Improve this Brightleaf proposal, then fact-check every claim in the improved version.",
        "proposal_draft.txt"),
}


def load_scenario(name: str):
    request, file = SCENARIOS[name]
    st.session_state["request"] = request
    st.session_state["attachment"] = read_sample(file) if file else ""
    st.session_state["route"] = "Let the supervisor decide"


def load_sample_memory():
    store = get_store()
    existing = set(store.stats()["sources"])
    added = 0
    for path in sorted((SAMPLES_DIR / "knowledge_base").glob("*.md")):
        if path.name not in existing:
            added += store.add_document(path.read_text(), path.name)
    st.session_state["memory_msg"] = f"Added {added} chunks from the sample documents."


# ── Sidebar: team memory ─────────────────────────────────────
with st.sidebar:
    st.markdown("## ConsultAgent-LLM")
    ok, msg = provider_ready()
    if ok:
        st.caption(f"Model: {msg}")
    else:
        st.error(msg)
    st.caption(f"Web search: {search_mode()}")

    st.markdown("### Team memory")
    st.caption("Documents the Second Brain answers from. Stored on this machine.")
    store = get_store()
    stats = store.stats()
    st.write(f"{stats['documents']} documents, {stats['chunks']} passages")

    uploads = st.file_uploader("Add documents", type=["txt", "md", "pdf", "docx"],
                               accept_multiple_files=True)
    if uploads and st.button("Add to memory", width="stretch"):
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
                st.write(s)


# ── Header ───────────────────────────────────────────────────
st.title("ConsultAgent-LLM")
st.markdown('<p class="lede">Ask once. A supervisor reads your request, assigns it to the '
            'right specialists, chains their work, and hands you one briefing.</p>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="roster">' + "".join(
        f'<span><i class="dot" style="background:{m["color"]}"></i>{escape(m["label"])}</span>'
        for m in AGENT_META.values()) + "</div>",
    unsafe_allow_html=True)

# ── Scenario buttons ─────────────────────────────────────────
st.markdown("**Try a scenario**")
cols = st.columns(3)
for i, name in enumerate(SCENARIOS):
    cols[i % 3].button(name, on_click=load_scenario, args=(name,), width="stretch")

# ── Request form ─────────────────────────────────────────────
st.text_area("What do you need?", key="request", height=90,
             placeholder="e.g. Which clients need attention this week?")

with st.expander("Attach text (emails, a proposal, a draft to check)",
                 expanded=bool(st.session_state.get("attachment"))):
    st.text_area("Attachment", key="attachment", height=220, label_visibility="collapsed")
    attached_file = st.file_uploader("…or attach a file", type=["txt", "md", "pdf", "docx"],
                                     key="attach_file")

route_options = ["Let the supervisor decide"] + [m["label"] for m in AGENT_META.values()]
c1, c2 = st.columns([3, 1])
route = c1.selectbox("Who should handle it?", route_options, key="route")
c2.write("")
run_clicked = c2.button("Run agents", type="primary", width="stretch", disabled=not ok)


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
            f'<div class="relay-detail">{escape(ev["detail"])}</div></div>')
    return '<div class="relay">' + "".join(rows) + "</div>"


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
        st.warning("Describe what you need first, or pick a scenario above.")
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
    st.divider()
    left, right = st.columns([1, 2.2], gap="large")
    with left:
        st.markdown("#### How the work was routed")
        if run.get("plan_reasoning"):
            st.caption(run["plan_reasoning"])
        st.markdown(relay_html(run.get("trace", [])), unsafe_allow_html=True)
    with right:
        st.markdown("#### Briefing")
        st.markdown(f'<div class="memo">{md.markdown(tidy_markdown(run["final_report"]), extensions=["tables"])}</div>',
                    unsafe_allow_html=True)

    ran = [s["agent"] for s in run.get("plan", []) if s["agent"] in run.get("results", {})]
    if ran:
        st.markdown("#### Specialist detail")
        tabs = st.tabs([AGENT_META[a]["label"] for a in ran])
        for tab, agent in zip(tabs, ran):
            with tab:
                res = run["results"][agent]
                if res["status"] == "ok":
                    RENDERERS[agent](res)
                else:
                    st.markdown(safe(res["summary_md"]))