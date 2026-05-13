import json
import os
import re
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
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

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

        self.document_map = {}

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

        self.document_map = {}

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


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)

        self.chunk_embeddings = None
        self.chunk_metadata = None

    def semantic_chunk_text(
        self,
        text,
        max_chunk_size=4,
        overlap=1
    ):
        sentences = re.split(
            r"(?<=[.!?])\s+",
            text
        )

        chunks = []

        start = 0

        while start < len(sentences):
            end = start + max_chunk_size

            chunk_sentences = sentences[start:end]

            chunks.append(
                " ".join(chunk_sentences)
            )

            if end >= len(sentences):
                break

            start += max_chunk_size - overlap

        return chunks

    def build_chunk_embeddings(self, documents):
        self.documents = documents

        self.document_map = {}

        for doc in documents:
            self.document_map[doc["id"]] = doc

        all_chunks = []

        chunk_metadata = []

        for movie_idx, doc in enumerate(documents):
            description = doc.get("description", "")

            if not description.strip():
                continue

            chunks = self.semantic_chunk_text(
                description,
                max_chunk_size=4,
                overlap=1
            )

            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)

                chunk_metadata.append({
                    "movie_idx": movie_idx,
                    "chunk_idx": chunk_idx,
                    "total_chunks": len(chunks)
                })

        self.chunk_embeddings = self.model.encode(
            all_chunks,
            show_progress_bar=True
        )

        self.chunk_metadata = chunk_metadata

        os.makedirs("cache", exist_ok=True)

        np.save(
            "cache/chunk_embeddings.npy",
            self.chunk_embeddings
        )

        with open(
            "cache/chunk_metadata.json",
            "w"
        ) as f:
            json.dump(
                {
                    "chunks": chunk_metadata,
                    "total_chunks": len(all_chunks)
                },
                f,
                indent=2
            )

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(
        self,
        documents
    ):
        self.documents = documents

        self.document_map = {}

        for doc in documents:
            self.document_map[doc["id"]] = doc

        embeddings_path = "cache/chunk_embeddings.npy"

        metadata_path = "cache/chunk_metadata.json"

        if (
            os.path.exists(embeddings_path)
            and os.path.exists(metadata_path)
        ):
            self.chunk_embeddings = np.load(
                embeddings_path
            )

            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            self.chunk_metadata = metadata["chunks"]

            return self.chunk_embeddings

        return self.build_chunk_embeddings(
            documents
        )

    def search_chunks(self, query: str, limit: int = 10):
        query_embedding = self.generate_embedding(query)

        chunk_scores = []

        for i, chunk_embedding in enumerate(self.chunk_embeddings):
            similarity = cosine_similarity(
                query_embedding,
                chunk_embedding
            )

            metadata = self.chunk_metadata[i]

            chunk_scores.append({
                "chunk_idx": metadata["chunk_idx"],
                "movie_idx": metadata["movie_idx"],
                "score": similarity
            })

        movie_scores = {}

        for chunk_score in chunk_scores:
            movie_idx = chunk_score["movie_idx"]
            score = chunk_score["score"]

            if (
                movie_idx not in movie_scores
                or score > movie_scores[movie_idx]
            ):
                movie_scores[movie_idx] = score

        sorted_movies = sorted(
            movie_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        results = []

        for movie_idx, score in sorted_movies[:limit]:
            movie = self.documents[movie_idx]

            results.append({
                "id": movie["id"],
                "title": movie["title"],
                "document": movie["description"][:100],
                "score": round(score, 4),
                "metadata": {}
            })

        return results


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