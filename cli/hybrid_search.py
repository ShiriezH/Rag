import os

from inverted_index import InvertedIndex
from semantic_search import ChunkedSemanticSearch


def normalize_scores(scores):
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score:
        return [1.0 for _ in scores]

    normalized = []

    for score in scores:
        normalized.append(
            (score - min_score)
            / (max_score - min_score)
        )

    return normalized


def hybrid_score(
    bm25_score,
    semantic_score,
    alpha=0.5
):
    return (
        alpha * bm25_score
        + (1 - alpha) * semantic_score
    )


def rrf_score(rank, k=60):
    return 1 / (k + rank)


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents

        self.document_map = {}

        for doc in documents:
            self.document_map[doc["id"]] = doc

        self.semantic_search = ChunkedSemanticSearch()

        self.semantic_search.load_or_create_chunk_embeddings(
            documents
        )

        self.idx = InvertedIndex()

        if not os.path.exists("cache/index.pkl"):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()

        return self.idx.bm25_search(
            query,
            limit
        )

    def weighted_search(
        self,
        query,
        alpha,
        limit=5
    ):
        bm25_results = self._bm25_search(
            query,
            limit * 500
        )

        semantic_results = self.semantic_search.search_chunks(
            query,
            limit * 500
        )

        # BM25 format:
        # (doc_id, score)

        bm25_scores = [
            result[1]
            for result in bm25_results
        ]

        semantic_scores = [
            result["score"]
            for result in semantic_results
        ]

        normalized_bm25 = normalize_scores(
            bm25_scores
        )

        normalized_semantic = normalize_scores(
            semantic_scores
        )

        combined = {}

        # Process BM25 results
        for i, result in enumerate(bm25_results):

            doc_id = int(result[0])

            document = self.document_map[doc_id]

            combined[doc_id] = {
                "id": doc_id,
                "title": document["title"],
                "document": document["description"],
                "bm25": normalized_bm25[i],
                "semantic": 0.0
            }

        # Process semantic results
        for i, result in enumerate(semantic_results):

            doc_id = result["id"]

            if doc_id not in combined:

                combined[doc_id] = {
                    "id": doc_id,
                    "title": result["title"],
                    "document": result["document"],
                    "bm25": 0.0,
                    "semantic": normalized_semantic[i]
                }

            else:

                combined[doc_id]["semantic"] = (
                    normalized_semantic[i]
                )

        final_results = []

        for result in combined.values():

            result["hybrid"] = hybrid_score(
                result["bm25"],
                result["semantic"],
                alpha
            )

            final_results.append(result)

        final_results.sort(
            key=lambda x: x["hybrid"],
            reverse=True
        )

        return final_results[:limit]

    def rrf_search(
        self,
        query,
        k=60,
        limit=10
    ):
        bm25_results = self._bm25_search(
            query,
            limit * 500
        )

        semantic_results = self.semantic_search.search_chunks(
            query,
            limit * 500
        )

        combined = {}

        # BM25 rankings
        for rank, result in enumerate(
            bm25_results,
            start=1
        ):

            doc_id = int(result[0])

            document = self.document_map[doc_id]

            if doc_id not in combined:

                combined[doc_id] = {
                    "id": doc_id,
                    "title": document["title"],
                    "document": document["description"],
                    "bm25_rank": rank,
                    "semantic_rank": None,
                    "rrf_score": 0.0
                }

            combined[doc_id]["rrf_score"] += (
                rrf_score(rank, k)
            )

        # Semantic rankings
        for rank, result in enumerate(
            semantic_results,
            start=1
        ):

            doc_id = result["id"]

            if doc_id not in combined:

                combined[doc_id] = {
                    "id": doc_id,
                    "title": result["title"],
                    "document": result["document"],
                    "bm25_rank": None,
                    "semantic_rank": rank,
                    "rrf_score": 0.0
                }

            else:

                combined[doc_id]["semantic_rank"] = rank

            combined[doc_id]["rrf_score"] += (
                rrf_score(rank, k)
            )

        final_results = list(
            combined.values()
        )

        final_results.sort(
            key=lambda x: x["rrf_score"],
            reverse=True
        )

        return final_results[:limit]