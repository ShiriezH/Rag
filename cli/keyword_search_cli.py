import argparse
import json
import string

from nltk.stem import PorterStemmer


stemmer = PorterStemmer()


def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))


def tokenize(text):
    return [token for token in text.split() if token]


def remove_stopwords(tokens, stopwords):
    return [token for token in tokens if token not in stopwords]


def stem_tokens(tokens):
    return [stemmer.stem(token) for token in tokens]


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")

            with open("data/movies.json", "r") as file:
                data = json.load(file)

            with open("data/stopwords.txt", "r") as file:
                stopwords = file.read().splitlines()

            results = []

            cleaned_query = remove_punctuation(args.query.lower())
            query_tokens = tokenize(cleaned_query)
            query_tokens = remove_stopwords(query_tokens, stopwords)
            query_tokens = stem_tokens(query_tokens)

            for movie in data["movies"]:
                cleaned_title = remove_punctuation(movie["title"].lower())
                title_tokens = tokenize(cleaned_title)
                title_tokens = remove_stopwords(title_tokens, stopwords)
                title_tokens = stem_tokens(title_tokens)

                found_match = False

                for query_token in query_tokens:
                    for title_token in title_tokens:
                        if query_token in title_token:
                            found_match = True

                if found_match:
                    results.append(movie)

            results = results[:5]

            for i, movie in enumerate(results, start=1):
                print(f"{i}. {movie['title']}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()