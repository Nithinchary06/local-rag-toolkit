"""Run with: python examples/basic_usage.py ./knowledge"""

import sys

from localrag import LocalRAG

if len(sys.argv) != 2:
    raise SystemExit("Usage: python examples/basic_usage.py PATH_TO_DOCUMENTS")

rag = LocalRAG.from_path(sys.argv[1])
for result in rag.retrieve("What are the main ideas?", top_k=3):
    print(f"\nScore: {result.score:.3f} | Source: {result.chunk.source}")
    print(result.chunk.text)
