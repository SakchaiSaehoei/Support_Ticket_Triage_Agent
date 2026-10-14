"""Prepare the Chroma knowledge database or inspect existing search results."""

import argparse
import asyncio
from pathlib import Path

from openai import APIConnectionError, APIStatusError, AuthenticationError, RateLimitError

from support_ticket_triage.retrieval import configured_retriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Index internal support references in Chroma")
    parser.add_argument("--source-dir", type=Path, help="Markdown source directory")
    parser.add_argument("--persist-dir", type=Path, help="Chroma persistence directory")
    parser.add_argument("--model", help="Override OPENAI_EMBEDDING_MODEL")
    parser.add_argument("--rebuild", action="store_true", help="Recreate the knowledge collection")
    parser.add_argument("--query", help="Search an existing index instead of building it")
    parser.add_argument("--doc-type", choices=["support_guide", "handbook"], default="support_guide")
    args = parser.parse_args()
    if args.query is not None and args.rebuild:
        parser.error("Use --rebuild for indexing, or --query for searching.")
    try:
        retriever = configured_retriever(
            source_path=args.source_dir, index_path=args.persist_dir, model=args.model
        )
        if args.query is None:
            count = asyncio.run(retriever.build_index(rebuild=args.rebuild))
            print(f"Chroma knowledge index is ready: {count} sections.")
        else:
            for article in asyncio.run(retriever.search(args.query, doc_type=args.doc_type)):
                print(article.model_dump_json(indent=2))
    except AuthenticationError:
        parser.exit(2, "OpenAI rejected the configured credentials. Update the local API key.\n")
    except RateLimitError:
        parser.exit(2, "OpenAI rejected the embedding request due to quota or rate limits.\n")
    except APIConnectionError:
        parser.exit(2, "Could not reach OpenAI for embeddings. Check network access.\n")
    except APIStatusError as exc:
        parser.exit(2, f"OpenAI embedding request failed with HTTP status {exc.status_code}.\n")
    except (OSError, ValueError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
