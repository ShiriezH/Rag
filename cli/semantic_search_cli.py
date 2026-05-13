#!/usr/bin/env python3

import argparse
import json
import re

from semantic_search import embed_query_text
from semantic_search import embed_text
from semantic_search import SemanticSearch
from semantic_search import ChunkedSemanticSearch
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

    chunk_parser = subparsers.add_parser(
        "chunk",
        help="Chunk text into fixed-size word groups"
    )

    chunk_parser.add_argument(
        "text",
        type=str,
        help="Text to chunk"
    )

    chunk_parser.add_argument(
        "--chunk-size",
        type=int,
        default=200,
        help="Number of words per chunk"
    )

    chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Number of overlapping words between chunks"
    )

    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk",
        help="Chunk text by sentence boundaries"
    )

    semantic_chunk_parser.add_argument(
        "text",
        type=str,
        help="Text to semantically chunk"
    )

    semantic_chunk_parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=4,
        help="Maximum number of sentences per chunk"
    )

    semantic_chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Number of overlapping sentences"
    )

    subparsers.add_parser(
        "embed_chunks",
        help="Generate chunk embeddings"
    )

    search_chunked_parser = subparsers.add_parser(
        "search_chunked",
        help="Search using chunk embeddings"
    )

    search_chunked_parser.add_argument(
        "query",
        type=str,
        help="Search query"
    )

    search_chunked_parser.add_argument(
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

        case "chunk":
            words = args.text.split()

            chunks = []

            start = 0

            while start < len(words):
                end = start + args.chunk_size

                chunk_words = words[start:end]

                chunks.append(" ".join(chunk_words))

                if end >= len(words):
                    break

                start += args.chunk_size - args.overlap

            print(f"Chunking {len(args.text)} characters")

            for i, chunk in enumerate(chunks, start=1):
                print(f"{i}. {chunk}")

        case "semantic_chunk":
            text = args.text.strip()

            if not text:
                chunks = []

            else:
                sentences = re.split(
                    r"(?<=[.!?])\s+",
                    text
                )

                if (
                    len(sentences) == 1
                    and not re.search(r"[.!?]$", sentences[0])
                ):
                    sentences = [text]

                chunks = []

                start = 0

                while start < len(sentences):
                    end = start + args.max_chunk_size

                    chunk_sentences = []

                    for sentence in sentences[start:end]:
                        cleaned = sentence.strip()

                        if cleaned:
                            chunk_sentences.append(cleaned)

                    chunk = " ".join(chunk_sentences).strip()

                    if chunk:
                        chunks.append(chunk)

                    if end >= len(sentences):
                        break

                    start += (
                        args.max_chunk_size - args.overlap
                    )

            print(
                f"Semantically chunking {len(args.text)} characters"
            )

            for i, chunk in enumerate(chunks, start=1):
                print(f"{i}. {chunk}")

        case "embed_chunks":
            with open("data/movies.json", "r") as file:
                data = json.load(file)

            documents = data["movies"]

            semantic_search = ChunkedSemanticSearch()

            embeddings = semantic_search.load_or_create_chunk_embeddings(
                documents
            )

            print(
                f"Generated {len(embeddings)} chunked embeddings"
            )

        case "search_chunked":
            with open("data/movies.json", "r") as file:
                data = json.load(file)

            documents = data["movies"]

            semantic_search = ChunkedSemanticSearch()

            semantic_search.load_or_create_chunk_embeddings(
                documents
            )

            results = semantic_search.search_chunks(
                args.query,
                args.limit
            )

            for i, result in enumerate(results, start=1):
                print(
                    f"\n{i}. {result['title']} (score: {result['score']:.4f})"
                )

                print(
                    f"   {result['document']}..."
                )

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()