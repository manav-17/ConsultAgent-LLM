"""
run_cli.py — run ConsultAgent-LLM from the terminal.

Examples
  python run_cli.py --ingest samples/knowledge_base
  python run_cli.py "Which clients are at risk?" --file samples/client_emails.txt
  python run_cli.py "Fact-check this" --file samples/ai_generated_text.txt --agent hallucination
"""
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from pathlib import Path

from core.agents_meta import AGENT_META, AGENT_NAMES
from core.config import provider_ready
from graph.workflow import build_graph, initial_state, stream_run
from memory.vector_store import get_store, load_file_text


def ingest(folder: str):
    store = get_store()
    for path in sorted(Path(folder).iterdir()):
        if path.suffix.lower() in {".txt", ".md", ".pdf", ".docx"}:
            n = store.add_document(load_file_text(path.name, path.read_bytes()), path.name)
            print(f"  indexed {path.name}: {n} passages")
    print(store.stats())


def main():
    p = argparse.ArgumentParser(description="ConsultAgent-LLM multi-agent runner")
    p.add_argument("request", nargs="?", help="what you need")
    p.add_argument("--file", help="text file to attach")
    p.add_argument("--agent", choices=AGENT_NAMES, help="skip the supervisor")
    p.add_argument("--ingest", help="folder of documents to add to the Second Brain")
    args = p.parse_args()

    if args.ingest:
        ingest(args.ingest)
        if not args.request:
            return

    if not args.request:
        p.error("give a request, or use --ingest")

    ok, msg = provider_ready()
    if not ok:
        raise SystemExit(msg)

    attachment = ""
    if args.file:
        path = Path(args.file)
        attachment = load_file_text(path.name, path.read_bytes())

    graph = build_graph()
    final = {}
    for node, state in stream_run(graph, initial_state(args.request, attachment, args.agent)):
        ev = state["trace"][-1]
        label = AGENT_META.get(node, {}).get("label", node.title())
        print(f"[{ev['seconds']:>5}s] {label:<17} {ev['status']:<11} {ev['detail']}")
        final = state

    print("\n" + "=" * 70 + "\n")
    print(final.get("final_report", ""))


if __name__ == "__main__":
    main()
