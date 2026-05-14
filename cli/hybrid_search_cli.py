import argparse
import json

from hybrid_search import HybridSearch


def normalize_scores(scores):
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score:
        return [1.0 for _ in scores]

    normalized = []

    for score in scores:
        normalized_score = (
            (score - min_score)
            / (max_score - min_score)
        )

        normalized.append(normalized_score)

    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hybrid Search CLI"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands"
    )

    # Normalize command
    normalize_parser = subparsers.add_parser(
        "normalize",
        help="Normalize scores"
    )

    normalize_parser.add_argument(
        "scores",
        type=float,
        nargs="*",
        help="Scores to normalize"
    )

    # Weighted search command
    weighted_parser = subparsers.add_parser(
        "weighted-search",
        help="Weighted hybrid search"
    )

    weighted_parser.add_argument(
        "query",
        type=str,
        help="Search query"
    )

    weighted_parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Keyword weighting"
    )

    weighted_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum results"
    )

    # RRF search command
    rrf_parser = subparsers.add_parser(
        "rrf-search",
        help="RRF hybrid search"
    )

    rrf_parser.add_argument(
        "query",
        type=str,
        help="Search query"
    )

    rrf_parser.add_argument(
        "-k",
        type=int,
        default=60,
        help="RRF k value"
    )

    rrf_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum results"
    )

    args = parser.parse_args()

    match args.command:

        case "normalize":

            normalized = normalize_scores(
                args.scores
            )

            for score in normalized:
                print(f"* {score:.4f}")

        case "weighted-search":

            with open(
                "data/movies.json",
                "r"
            ) as file:
                data = json.load(file)

            documents = data["movies"]

            hybrid = HybridSearch(
                documents
            )

            results = hybrid.weighted_search(
                args.query,
                args.alpha,
                args.limit
            )

            for i, result in enumerate(
                results,
                start=1
            ):

                print(
                    f"{i}. {result['title']}"
                )

                print(
                    f"  Hybrid Score: {result['hybrid']:.3f}"
                )

                print(
                    f"  BM25: {result['bm25']:.3f}, Semantic: {result['semantic']:.3f}"
                )

                print(
                    f"  {result['document'][:100]}..."
                )

        case "rrf-search":

            with open(
                "data/movies.json",
                "r"
            ) as file:
                data = json.load(file)

            documents = data["movies"]

            hybrid = HybridSearch(
                documents
            )

            results = hybrid.rrf_search(
                args.query,
                args.k,
                args.limit
            )

            for i, result in enumerate(
                results,
                start=1
            ):

                print(
                    f"{i}. {result['title']}"
                )

                print(
                    f"  RRF Score: {result['rrf_score']:.3f}"
                )

                print(
                    f"  BM25 Rank: {result['bm25_rank']}, Semantic Rank: {result['semantic_rank']}"
                )

                print(
                    f"  {result['document'][:100]}..."
                )

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()