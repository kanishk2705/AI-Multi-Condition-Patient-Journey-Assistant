"""Command-line entry point to build the local Chroma index."""
from __future__ import annotations

import argparse
import json
import sys

from .embeddings import DEFAULT_EMBEDDING_MODEL
from .vector_index import (
    DEFAULT_CHUNKS_PATH,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_INDEX_DIR,
    build_persistent_index,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", default=str(DEFAULT_CHUNKS_PATH))
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--model-name", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    try:
        report = build_persistent_index(
            args.chunks,
            args.index_dir,
            model_name=args.model_name,
            collection_name=args.collection,
            batch_size=args.batch_size,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Index build failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
