# 🔍 Semantic Search for Twitter API Documentation

A beginner-friendly Python project to search Twitter API documentation using semantic embeddings and FAISS vector search.

## 📋 Features

- **Document Chunking**: Splits markdown files into meaningful chunks
- **Embeddings**: Uses SentenceTransformers for semantic embeddings
- **Vector Search**: FAISS-based similarity search
- **CLI Interface**: Command-line query interface
- **Metadata Tracking**: Keeps track of document sources

## 🛠 Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone/Navigate to project**
```bash
cd semantic-search-twitter
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
# OR
venv\Scripts\activate     # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Add your data**
```bash
# Place Twitter API markdown files in:
data/raw_docs/
```

## 🚀 Usage

### Basic Search
```bash
python semantic_search.py --query "how to fetch tweets"
```

### Advanced Options
```bash
python semantic_search.py --query "authentication" --top-k 5 --rebuild-index
```

### Command-line Arguments
- `--query TEXT`: Search query (required)
- `--top-k INT`: Number of results to return (default: 3)
- `--rebuild-index`: Force rebuild of embeddings index

## 📁 Project Structure

```
semantic-search-twitter/
├── data/
│   └── raw_docs/              # Place markdown files here
├── embeddings/
│   ├── index.faiss            # Vector index (auto-generated)
│   └── metadata.json          # Chunk metadata (auto-generated)
├── src/
│   ├── __init__.py
│   ├── chunker.py             # Document chunking logic
│   ├── embedder.py            # Embedding generation
│   ├── indexer.py             # FAISS index management
│   ├── search.py              # Search implementation
│   └── utils.py               # Helper functions
├── semantic_search.py         # Main CLI entry point
├── requirements.txt           # Project dependencies
├── README.md                  # This file
└── .gitignore                 # Git ignore rules
```

## 📚 Project Files Explanation

| File | Purpose |
|------|---------|
| `src/chunker.py` | Loads markdown files and splits them into chunks |
| `src/embedder.py` | Converts text chunks into vector embeddings |
| `src/indexer.py` | Creates and manages FAISS vector index |
| `src/search.py` | Performs semantic similarity search |
| `src/utils.py` | Utility functions (file I/O, logging, etc.) |
| `semantic_search.py` | Main entry point for CLI application |

## 🔄 Workflow

1. **Load Documents** → `src/chunker.py` reads all markdown files
2. **Create Chunks** → Documents split into meaningful segments
3. **Generate Embeddings** → `src/embedder.py` creates vector representations
4. **Build Index** → `src/indexer.py` stores in FAISS
5. **Search** → User queries matched against index
6. **Return Results** → Top-K similar chunks ranked by similarity

## 🎯 Next Steps

1. Add your Twitter API documentation markdown files to `data/raw_docs/`
2. Run `python semantic_search.py --query "test"` to build the index
3. Try different queries to test the search
4. Iterate on the code to improve results

## 📝 Notes for Beginners

- Start simple: build one module at a time
- Test each piece independently
- Use print statements to debug
- Read the docstrings in each module
- Gradually add complexity (caching, filtering, etc.)

## 🤝 Common Issues & Solutions

**Issue**: ImportError for faiss or sentence-transformers
```bash
# Solution: Make sure virtual environment is activated
source venv/bin/activate
pip install -r requirements.txt
```

**Issue**: No documents found in `data/raw_docs/`
```bash
# Solution: Add markdown files to the folder first
ls data/raw_docs/  # Check if files exist
```

**Issue**: Out of memory with large documents
```bash
# Solution: Adjust chunk size in src/chunker.py
```

## 📖 Learning Resources

- [FAISS Documentation](https://faiss.ai/)
- [Sentence-Transformers](https://www.sbert.net/)
- [Semantic Search Concepts](https://huggingface.co/docs/transformers/tasks/semantic_similarity)

## ⚖️ License

MIT License
