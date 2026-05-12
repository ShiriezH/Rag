import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text):
        if not text or text.isspace():
            raise ValueError("Text cannot be empty")

        embedding = self.model.encode([text])

        return embedding[0]

    def build_embeddings(self, documents):
        self.documents = documents

        for doc in documents:
            self.document_map[doc["id"]] = doc

        movie_strings = []

        for doc in documents:
            movie_strings.append(
                f"{doc['title']}: {doc['description']}"
            )

        self.embeddings = self.model.encode(
            movie_strings,
            show_progress_bar=True
        )

        Path("cache").mkdir(exist_ok=True)

        np.save(
            "cache/movie_embeddings.npy",
            self.embeddings
        )

        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents

        for doc in documents:
            self.document_map[doc["id"]] = doc

        cache_path = Path("cache/movie_embeddings.npy")

        if cache_path.exists():
            self.embeddings = np.load(cache_path)

            if len(self.embeddings) == len(documents):
                return self.embeddings

        return self.build_embeddings(documents)

    def search(self, query, limit=5):
        if self.embeddings is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)

        results = []

        for i, embedding in enumerate(self.embeddings):
            similarity = cosine_similarity(
                query_embedding,
                embedding
            )

            document = self.documents[i]

            results.append(
                (
                    similarity,
                    document
                )
            )

        results.sort(
            key=lambda x: x[0],
            reverse=True
        )

        final_results = []

        for similarity, document in results[:limit]:
            final_results.append(
                {
                    "score": similarity,
                    "title": document["title"],
                    "description": document["description"]
                }
            )

        return final_results


def verify_model():
    semantic_search = SemanticSearch()

    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")


def embed_text(text):
    semantic_search = SemanticSearch()

    embedding = semantic_search.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    semantic_search = SemanticSearch()

    with open("data/movies.json", "r") as file:
        data = json.load(file)

    documents = data["movies"]

    embeddings = semantic_search.load_or_create_embeddings(
        documents
    )

    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_query_text(query):
    semantic_search = SemanticSearch()

    embedding = semantic_search.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")