"""
agents_meta.py — names, descriptions and colours for every agent.
The supervisor reads the descriptions to decide who to call.
The UI reads labels and colours. No agent code is imported here.
"""

AGENT_META = {
    "second_brain": {
        "label": "Second Brain",
        "color": "#0F7C7A",
        "description": (
            "Answers questions from the team's own uploaded documents "
            "(meeting notes, project docs, policies). Finds past decisions, "
            "who promised what, open commitments, and timelines. Use for "
            "anything about 'what did we decide / promise / discuss'."
        ),
    },
    "hallucination": {
        "label": "Fact Checker",
        "color": "#B3261E",
        "description": (
            "Extracts factual claims from a piece of text (usually AI-generated), "
            "verifies each against the web and internal documents, gives a trust "
            "score and a corrected version. Use when the user wants to verify, "
            "fact-check or audit text."
        ),
    },
    "client_health": {
        "label": "Client Health",
        "color": "#B26A00",
        "description": (
            "Reads client emails or interaction logs, detects tone, hidden concerns "
            "and engagement signals, scores each client's health 0-100, estimates "
            "churn risk and recommends actions plus a reply. Use for client "
            "relationships, churn, sentiment or account risk."
        ),
    },
    "competitive_intel": {
        "label": "Competitor Watch",
        "color": "#5B3FA8",
        "description": (
            "Searches the web for recent moves by named competitors (launches, "
            "hiring, pricing, partnerships, funding) and turns them into threats, "
            "opportunities and recommended moves. Use when competitors or market "
            "players are named."
        ),
    },
    "proposal": {
        "label": "Proposal Coach",
        "color": "#2E7D32",
        "description": (
            "Scores a business proposal against an 8-part rubric, estimates win "
            "probability, rewrites weak sections and re-scores in a loop until it "
            "improves. Use when the user shares or asks about a proposal, pitch or "
            "statement of work."
        ),
    },
}

AGENT_NAMES = list(AGENT_META.keys())

SUPERVISOR_COLOR = "#10233F"


# Every agent that reads team memory gets this, so no agent can mistake the
# user's own firm for a competitor (that once inverted a whole strategy).
OWN_DOCS_NOTE = (
    "IMPORTANT: documents in the team memory are the user's OWN firm's internal "
    "documents. 'We', 'us' and 'our firm' mean the firm those documents describe. "
    "Never treat that firm as a competitor or a third party, and never recommend "
    "anything its own decisions rule out."
)