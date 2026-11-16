"""Command-line interface for LocalRAG."""

import argparse
import sys
from .generators import OllamaGenerator
from .pipeline import LocalRAG


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="localrag", description="Search local documents or ask them with a local Ollama model.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    search_parser = subparsers.add_parser("search", help="retrieve relevant document passages")
    search_parser.add_argument("path", help="file or directory containing .txt, .md, or .pdf files")
    search_parser.add_argument("question", help="search query")
    search_parser.add_argument("--top-k", type=int, default=4)
    ask_parser = subparsers.add_parser("ask", help="generate an answer using Ollama")
    ask_parser.add_argument("path", help="file or directory containing .txt, .md, or .pdf files")
    ask_parser.add_argument("question", help="question to answer")
    ask_parser.add_argument("--model", default="llama3.2")
    ask_parser.add_argument("--top-k", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        rag = LocalRAG.from_path(args.path)
        if args.command == "search":
            results = rag.retrieve(args.question, top_k=args.top_k)
            if not results:
                print("No relevant passages found.")
                return
            for number, result in enumerate(results, start=1):
                page = result.chunk.metadata.get("page")
                suffix = f" (page {page})" if page is not None else ""
                print(f"\n[{number}] {result.chunk.source}{suffix} | score={result.score:.3f}")
                print(result.chunk.text)
            return
        response = rag.ask(args.question, OllamaGenerator(model=args.model), top_k=args.top_k)
        print(response.answer)
        if response.sources:
            print("\nRetrieved sources:")
            for number, result in enumerate(response.sources, start=1):
                page = result.chunk.metadata.get("page")
                suffix = f", page {page}" if page is not None else ""
                print(f"[{number}] {result.chunk.source}{suffix}")
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
