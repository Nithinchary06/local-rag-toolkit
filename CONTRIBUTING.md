# Contributing to LocalRAG

Thank you for considering a contribution. Small, focused contributions are especially welcome.

## Development setup

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## Before opening a pull request

1. Search existing issues and pull requests to avoid duplicate work.
2. For substantial changes, open an issue to discuss the approach first.
3. Add or update tests for behavior changes.
4. Run `pytest` and `ruff check .` locally.
5. Update the README or docs if user-facing behavior changes.
6. Keep pull requests focused and explain the problem, solution, and tests.

## Good first contributions

- Add tests for edge cases in loading and chunking.
- Improve error messages and documentation.
- Add small sample documents or runnable examples.
- Benchmark chunk sizes and retrieval quality on a documented sample set.
- Propose a new loader or retriever through an issue before implementing it.

## Code style

- Target Python 3.10+.
- Add type hints to public functions.
- Prefer small functions and clear names.
- Do not commit API keys, private documents, model weights, or virtual environments.
