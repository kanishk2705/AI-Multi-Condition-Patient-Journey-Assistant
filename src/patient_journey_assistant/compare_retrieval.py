"""Compare baseline and condition/stage-aware retrieval for one query."""
from __future__ import annotations

import argparse
import sys

from .manifest import ALLOWED_CONDITIONS, ALLOWED_STAGES
from .retrieval import RetrievedChunk, create_persistent_retriever
from .vector_index import DEFAULT_COLLECTION_NAME, DEFAULT_INDEX_DIR


def _print_results(label: str, results: list[RetrievedChunk]) -> None:
    print(f"\n=== {label} ({len(results)} results) ===")
    if not results:
        print("No eligible evidence was found.")
        return

    for rank, item in enumerate(results, start=1):
        metadata = item.metadata
        title = metadata.get("title", "Untitled source")
        url = metadata.get("source_url", "")
        stages = ", ".join(metadata.get("stage_scope", []))
        excerpt = " ".join(item.text.split())
        if len(excerpt) > 320:
            excerpt = excerpt[:317].rstrip() + "..."
        print(f"\n{rank}. {title} | cosine similarity={item.similarity:.4f}")
        print(f"   Chunk ID: {item.chunk_id}")
        print(f"   Condition: {metadata.get('condition', 'unknown')}")
        print(f"   Stage scope: {stages or 'not recorded'}")
        print(f"   Source: {url}")
        print(f"   Evidence: {excerpt}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True, help="General educational question to search.")
    parser.add_argument("--condition", required=True, choices=sorted(ALLOWED_CONDITIONS))
    parser.add_argument("--stage", required=True, choices=sorted(ALLOWED_STAGES))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME)
    args = parser.parse_args()

    if args.k < 1:
        parser.error("--k must be at least 1.")

    try:
        retriever = create_persistent_retriever(
            args.index_dir,
            args.collection,
        )
        baseline = retriever.retrieve_baseline(args.query, k=args.k)
        contextual = retriever.retrieve_contextual(
            args.query,
            condition=args.condition,
            stage=args.stage,
            k=args.k,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Retrieval failed: {exc}", file=sys.stderr)
        return 1

    print(f"Query: {args.query}")
    print(f"Selected context: {args.condition} / {args.stage}")
    print("No answer generation or external LLM API is used in this command.")
    _print_results("Method A — baseline semantic retrieval", baseline)
    _print_results("Method B — condition-and-stage-aware retrieval", contextual)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
