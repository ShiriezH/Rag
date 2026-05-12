#!/usr/bin/env python3

import argparse
import json

from semantic_search import embed_query_text
from semantic_search import embed_text
from semantic_search import SemanticSearch
from semantic_search import verify_embeddings
from semantic_search import verify_model


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser(
        "verify",
        help="Verify semantic search model"
    )

    embed_text_parser = subparsers.add_parser(
        "embed_text",
        help="Generate embedding for text"
    )

    embed_text_parser.add_argument(
        "text",
        type=str,
        help="Text to embed"
    )

    subparsers.add_parser(
        "verify_embeddings",
        help="Verify movie embeddings"
    )

    embed_query_parser = subparsers.add_parser(
        "embed_query",
        help="Generate embedding for query text"
    )

    embed_query_parser.add_argument(
        "query",
        type=str,
        help="Query text"
    )

    search_parser = subparsers.add_parser(
        "search",
        help="Semantic movie search"
    )

    search_parser.add_argument(
        "query",
        type=str,
        help="Search query"
    )

    search_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of results"
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()

        case "embed_text":
            embed_text(args.text)

        case "verify_embeddings":
            verify_embeddings()

        case "embed_query":
            embed_query_text(args.query)

        case "search":
            semantic_search = SemanticSearch()

            with open("data/movies.json", "r") as file:
                data = json.load(file)

            documents = data["movies"]

            semantic_search.load_or_create_embeddings(
                documents
            )

            results = semantic_search.search(
                args.query,
                args.limit
            )

            for i, result in enumerate(results, start=1):
                print(
                    f"{i}. {result['title']} (score: {result['score']:.4f})"
                )

                print(f"  {result['description'][:100]}...")
                print()

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()