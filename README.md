# RAG Hybrid Search

A Python-based Retrieval-Augmented Generation (RAG) search system that combines:

- BM25 keyword search
- Semantic vector search
- Hybrid weighted ranking
- Reciprocal Rank Fusion (RRF)

This project was built to explore modern search techniques used in AI-powered applications and retrieval systems.

---

## Features

### Keyword Search (BM25)
Traditional keyword-based search using an inverted index.

### Semantic Search
Uses embeddings to search based on meaning instead of exact words.

### Weighted Hybrid Search
Combines BM25 and semantic search scores using configurable weighting.

Example:

```python
hybrid_score = (
    alpha * bm25_score
    + (1 - alpha) * semantic_score
)
```

## Reciprocal Rank Fusion (RRF)

Combines rankings instead of raw scores for more stable hybrid results.

```python
rrf_score = 1 / (k + rank)
```

## Project Structure
```
Rag/
│
├── cache/
├── cli/
│   ├── hybrid_search.py
│   ├── hybrid_search_cli.py
│   ├── inverted_index.py
│   ├── semantic_search.py
│   └── constants.py
│
├── data/
│   └── movies.json
│
├── main.py
├── pyproject.toml
└── README.md
```

## Installation

Clone the repository:

```Bash
git clone https://github.com/YOUR_USERNAME/Rag.git
cd Rag
```
Install dependencies:
```Bash
uv sync
```

Or with pip:
```Bash
pip install -r requirements.txt
```

## Running the CLI
Normalize Scores

```Bash
uv run cli/hybrid_search_cli.py normalize 0.5 2.3 1.2 0.5 0.1
```

## Example output:

* 0.1818
* 1.0000
* 0.5000
* 0.1818
* 0.0000

## Weighted Hybrid Search

```Bash
uv run cli/hybrid_search_cli.py weighted-search "British Bear" --alpha 0.5 --limit 5
```

## Arguments:

| Argument | Description                                   |
| -------- | --------------------------------------------- |
| query    | Search query                                  |
| --alpha  | Weighting between keyword and semantic search |
| --limit  | Maximum results                               |

## RRF Search

```Bash
uv run cli/hybrid_search_cli.py rrf-search "family fighting movie" --limit 5
```

## Arguments:

| Argument | Description     |
| -------- | --------------- |
| query    | Search query    |
| --k      | RRF constant    |
| --limit  | Maximum results |

# Search Techniques
## BM25 Search
- Exact keyword matching
- Fast retrieval using inverted indexes
- Great for titles and exact phrases
- 
## Semantic Search
- Embedding-based retrieval
- Finds conceptually related documents
- Better for natural language queries
  
## Hybrid Search
Combines both approaches to improve relevance and flexibility.

## Technologies Used
- Python
- Sentence Transformers
- NumPy
- BM25
- Vector Embeddings
- CLI Tools

## Future Improvements
- Add reranking models
- Web interface
- Database storage
- Streaming search responses
- OpenAI integration
- Advanced chunking strategies
