# LocalRAG

[![Tests](https://github.com/Nithinchary06/local-rag-toolkit/actions/workflows/tests.yml/badge.svg)](https://github.com/Nithinchary06/local-rag-toolkit/actions/workflows/tests.yml) [![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**A small, local-first Python toolkit for searching your documents and building grounded RAG applications.**

LocalRAG loads TXT, Markdown, and PDF files, splits them into source-aware chunks, retrieves relevant passages with TF-IDF, and can optionally generate answers with a locally running [Ollama](https://ollama.com/) model.

LocalRAG offers a lightweight TF-IDF baseline and an optional semantic retriever powered by Sentence Transformers.

## Features

- Load `.txt`, `.md`, `.markdown`, and text-based `.pdf` files.
- Process directories recursively.
- Preserve source paths and PDF page numbers for traceability.
- Configurable word-based chunking and overlap.
- Retrieve passages using TF-IDF or semantic sentence embeddings.
- Keep semantic embedding dependencies optional; models run locally.
- Optional grounded answer generation through Ollama.
- Command-line interface and Python API.
- No hosted API key required.

## Quickstart

### Requirements

- Python 3.10 or newer.
- Ollama is optional and only needed for generated answers.

### Install

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e .

# Optional: local semantic embedding support
pip install -e '.[semantic]'
```

### Choose a retrieval backend

TF-IDF is the lightweight default. To use semantic retrieval, install the optional extra above and run:

```bash
localrag search ./knowledge "How can I travel by automobile?" --retriever semantic

# Choose where semantic embeddings are cached (default: .localrag/index.sqlite3)
localrag search ./knowledge "How can I travel by automobile?" --retriever semantic --index-path ./.localrag/index.sqlite3
```

The default model is `sentence-transformers/all-MiniLM-L6-v2`. It is downloaded on first use and then runs locally. Select another Sentence Transformers model or local model directory with `--embedding-model`.

Semantic document embeddings are cached in a local SQLite database at `.localrag/index.sqlite3` by default. Repeated runs reuse the cached vectors when the source content, chunk settings, and embedding model are unchanged. If documents change, LocalRAG rebuilds the affected index automatically. The `.localrag/` directory is excluded from Git.

### Add your documents

```bash
mkdir knowledge
```

Place your `.txt`, `.md`, or `.pdf` documents inside the `knowledge` directory.

### Search documents

```bash
localrag search ./knowledge "What does the documentation say about installation?"
```

### Generate an answer with Ollama

Install Ollama, start its local service, and download a model:

```bash
ollama pull llama3.2
```

Then run:

```bash
localrag ask ./knowledge "Summarize the installation steps." --model llama3.2
```

The generator is optional. Retrieval works without Ollama.

## Python API

```python
from localrag import LocalRAG

rag = LocalRAG.from_path("./knowledge")
results = rag.retrieve("How do I install the package?", top_k=3)

for result in results:
    print(result.chunk.source, result.score)
    print(result.chunk.text)

from localrag import OllamaGenerator
response = rag.ask(
    "How do I install the package?",
    OllamaGenerator(model="llama3.2"),
    top_k=3,
)
print(response.answer)
for source in response.sources:
    print(source.chunk.source)
```

## How it works

```text
Documents (TXT / Markdown / PDF)
              |
              v
        Document loader
              |
              v
      Word-based chunking
              |
              v
       TF-IDF index
              |
       User question
              |
              v
     Similarity retrieval
          /       \\
         v         v
   Inspect hits   Ollama (optional)
                      |
                      v
              Answer + sources
```

## Current scope and limitations

- TF-IDF is lexical retrieval and may miss conceptually related passages that use different words. Use the optional semantic retriever for meaning-based matching.
- Scanned PDFs need OCR before their contents can be indexed.
- Semantic embeddings are persisted in a local SQLite cache. TF-IDF indexes are still built in memory for each process.
- Ollama must be installed and running locally for answer generation.
- Generated answers can still be incorrect. Review the cited source passages for important decisions.

## Development

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
```

## Roadmap

- [x] Add optional embedding-based semantic retrieval.
- [x] Add persistent SQLite storage for semantic embeddings.
- [ ] Add incremental indexing and reusable indexes across chunking/model migrations.
- [ ] Add chunking strategies and metadata filters.
- [ ] Improve PDF handling and add OCR integration as an optional extra.
- [ ] Add evaluation datasets and retrieval metrics.
- [ ] Publish package releases to PyPI.
- [ ] Add more examples and documentation.

Feature requests and bug reports are welcome in [GitHub Issues](https://github.com/Nithinchary06/local-rag-toolkit/issues).

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md), follow the [Code of Conduct](CODE_OF_CONDUCT.md), and open an issue before large changes.

## Security

Please see [SECURITY.md](SECURITY.md) for vulnerability reporting guidance.

## License

MIT. See [LICENSE](LICENSE).
