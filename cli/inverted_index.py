import json
import math
import pickle
import string
from collections import Counter
from pathlib import Path

from nltk.stem import PorterStemmer

from constants import BM25_B
from constants import BM25_K1


stemmer = PorterStemmer()


def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))


def tokenize(text):
    return [token for token in text.split() if token]


def remove_stopwords(tokens, stopwords):
    return [token for token in tokens if token not in stopwords]


def stem_tokens(tokens):
    return [stemmer.stem(token) for token in tokens]


def process_text(text):
    with open("data/stopwords.txt", "r") as file:
        stopwords = file.read().splitlines()

    text = remove_punctuation(text.lower())
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens, stopwords)
    tokens = stem_tokens(tokens)

    return tokens


class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.term_frequencies = {}
        self.doc_lengths = {}

    def __add_document(self, doc_id, text):
        tokens = process_text(text)

        self.term_frequencies[doc_id] = Counter()
        self.doc_lengths[doc_id] = len(tokens)

        for token in tokens:
            if token not in self.index:
                self.index[token] = set()

            self.index[token].add(doc_id)

            self.term_frequencies[doc_id][token] += 1

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0

        total = sum(self.doc_lengths.values())

        return total / len(self.doc_lengths)

    def get_documents(self, term):
        tokens = process_text(term)

        if len(tokens) != 1:
            raise Exception("Expected exactly one token")

        term = tokens[0]

        docs = self.index.get(term, set())

        return sorted(list(docs))

    def get_tf(self, doc_id, term):
        tokens = process_text(term)

        if len(tokens) != 1:
            raise Exception("Expected exactly one token")

        term = tokens[0]

        return self.term_frequencies.get(doc_id, Counter()).get(term, 0)

    def get_bm25_idf(self, term: str) -> float:
        tokens = process_text(term)

        if len(tokens) != 1:
            raise Exception("Expected exactly one token")

        term = tokens[0]

        N = len(self.docmap)
        df = len(self.get_documents(term))

        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    def get_bm25_tf(
        self,
        doc_id,
        term,
        k1=BM25_K1,
        b=BM25_B
    ):
        tf = self.get_tf(doc_id, term)

        doc_length = self.doc_lengths.get(doc_id, 0)
        avg_doc_length = self.__get_avg_doc_length()

        if avg_doc_length == 0:
            return 0.0

        length_norm = 1 - b + b * (doc_length / avg_doc_length)

        return (tf * (k1 + 1)) / (tf + k1 * length_norm)
    
    def bm25(self, doc_id, term):
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)

        return bm25_tf * bm25_idf

    def bm25_search(self, query, limit=5):
        query_tokens = process_text(query)

        scores = {}

        for doc_id in self.docmap:
            total_score = 0

            for token in query_tokens:
                total_score += self.bm25(doc_id, token)

            scores[doc_id] = total_score

        sorted_scores = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True
        )

        return sorted_scores[:limit]

    def build(self):
        with open("data/movies.json", "r") as file:
            data = json.load(file)

        for movie in data["movies"]:
            doc_id = movie["id"]

            self.docmap[doc_id] = movie

            text = f"{movie['title']} {movie['description']}"

            self.__add_document(doc_id, text)

    def save(self):
        Path("cache").mkdir(exist_ok=True)

        with open("cache/index.pkl", "wb") as file:
            pickle.dump(self.index, file)

        with open("cache/docmap.pkl", "wb") as file:
            pickle.dump(self.docmap, file)

        with open("cache/term_frequencies.pkl", "wb") as file:
            pickle.dump(self.term_frequencies, file)

        with open("cache/doc_lengths.pkl", "wb") as file:
            pickle.dump(self.doc_lengths, file)

    def load(self):
        with open("cache/index.pkl", "rb") as file:
            self.index = pickle.load(file)

        with open("cache/docmap.pkl", "rb") as file:
            self.docmap = pickle.load(file)

        with open("cache/term_frequencies.pkl", "rb") as file:
            self.term_frequencies = pickle.load(file)

        with open("cache/doc_lengths.pkl", "rb") as file:
            self.doc_lengths = pickle.load(file)