import argparse
import math

from inverted_index import InvertedIndex
from inverted_index import process_text
from constants import BM25_B
from constants import BM25_K1


def calculate_idf(index, term):
    processed = process_text(term)

    if len(processed) != 1:
        raise Exception("Expected exactly one token")

    term = processed[0]

    total_doc_count = len(index.docmap)
    term_match_doc_count = len(index.get_documents(term))

    return math.log((total_doc_count + 1) / (term_match_doc_count + 1))


def bm25_idf_command(term):
    index = InvertedIndex()

    index.load()

    return index.get_bm25_idf(term)


def bm25_tf_command(doc_id, term, k1=BM25_K1, b=BM25_B):
    index = InvertedIndex()

    index.load()

    return index.get_bm25_tf(doc_id, term, k1, b)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build inverted index")

    search_parser = subparsers.add_parser(
        "search",
        help="Search movies using BM25"
    )
    search_parser.add_argument(
        "query",
        type=str,
        help="Search query"
    )

    tf_parser = subparsers.add_parser(
        "tf",
        help="Get term frequency"
    )
    tf_parser.add_argument(
        "doc_id",
        type=int,
        help="Document ID"
    )
    tf_parser.add_argument(
        "term",
        type=str,
        help="Search term"
    )

    idf_parser = subparsers.add_parser(
        "idf",
        help="Get inverse document frequency"
    )
    idf_parser.add_argument(
        "term",
        type=str,
        help="Search term"
    )

    tfidf_parser = subparsers.add_parser(
        "tfidf",
        help="Get TF-IDF score"
    )
    tfidf_parser.add_argument(
        "doc_id",
        type=int,
        help="Document ID"
    )
    tfidf_parser.add_argument(
        "term",
        type=str,
        help="Search term"
    )

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf",
        help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument(
        "term",
        type=str,
        help="Term to get BM25 IDF score for"
    )

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf",
        help="Get BM25 TF score for a given document ID and term"
    )

    bm25_tf_parser.add_argument(
        "doc_id",
        type=int,
        help="Document ID"
    )

    bm25_tf_parser.add_argument(
        "term",
        type=str,
        help="Term to get BM25 TF score for"
    )

    bm25_tf_parser.add_argument(
        "k1",
        type=float,
        nargs="?",
        default=BM25_K1,
        help="Tunable BM25 K1 parameter"
    )

    bm25_tf_parser.add_argument(
        "b",
        type=float,
        nargs="?",
        default=BM25_B,
        help="Tunable BM25 b parameter"
    )

    bm25search_parser = subparsers.add_parser(
        "bm25search",
        help="Search movies using full BM25 scoring"
    )

    bm25search_parser.add_argument(
        "query",
        type=str,
        help="Search query"
    )

    bm25search_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of results"
    )

    args = parser.parse_args()

    match args.command:
        case "build":
            index = InvertedIndex()

            index.build()
            index.save()

            print("Index built successfully")

        case "search":
            index = InvertedIndex()

            try:
                index.load()
            except FileNotFoundError:
                print("Index files not found. Please run build first.")
                return

            print(f"Searching for: {args.query}")

            query_tokens = process_text(args.query)

            results = []

            for token in query_tokens:
                docs = index.get_documents(token)

                for doc_id in docs:
                    if doc_id not in results:
                        results.append(doc_id)

                    if len(results) >= 5:
                        break

                if len(results) >= 5:
                    break

            for doc_id in results:
                movie = index.docmap[doc_id]
                print(f"{movie['id']}. {movie['title']}")

        case "tf":
            index = InvertedIndex()

            index.load()

            tf = index.get_tf(args.doc_id, args.term)

            print(tf)

        case "idf":
            index = InvertedIndex()

            index.load()

            idf = calculate_idf(index, args.term)

            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case "tfidf":
            index = InvertedIndex()

            index.load()

            tf = index.get_tf(args.doc_id, args.term)
            idf = calculate_idf(index, args.term)

            tf_idf = tf * idf

            print(
                f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}"
            )

        case "bm25idf":
            bm25idf = bm25_idf_command(args.term)

            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")

        case "bm25tf":
            bm25tf = bm25_tf_command(
                args.doc_id,
                args.term,
                args.k1,
                args.b
            )

            print(
                f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}"
            )

        case "bm25search":
            index = InvertedIndex()

            index.load()

            results = index.bm25_search(
                args.query,
                args.limit
            )

            for i, (doc_id, score) in enumerate(results, start=1):
                movie = index.docmap[doc_id]

                print(
                    f"{i}. ({doc_id}) {movie['title']} - Score: {score:.2f}"
                )

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()