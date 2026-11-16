"""Optional local text-generation adapters."""

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .models import SearchResult


def build_prompt(question: str, results: list[SearchResult]) -> str:
    """Build a grounded prompt with numbered source references."""
    evidence = []
    for number, result in enumerate(results, start=1):
        page = result.chunk.metadata.get("page")
        page_note = f", page {page}" if page is not None else ""
        evidence.append(f"[{number}] Source: {result.chunk.source}{page_note}\n{result.chunk.text}")
    context = "\n\n".join(evidence) or "No relevant passages were retrieved."
    return ("Answer the question using only the context below. When you use a fact, cite its source "
            "with the supplied label, such as [1]. If the context does not contain the answer, "
            "say that you do not know.\n\n" f"Context:\n{context}\n\nQuestion: {question}\nAnswer:")


class OllamaGenerator:
    """Generate answers through a locally running Ollama server."""
    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434", timeout: int = 120) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        payload = json.dumps({"model": self.model, "prompt": prompt, "stream": False}).encode("utf-8")
        request = Request(f"{self.base_url}/api/generate", data=payload, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (URLError, HTTPError, TimeoutError) as exc:
            raise RuntimeError("Could not reach Ollama. Start Ollama, then run `ollama pull llama3.2` or choose a model you already have.") from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RuntimeError("Ollama returned an invalid response") from exc
        answer = data.get("response")
        if not isinstance(answer, str):
            raise RuntimeError("Ollama response did not contain a text 'response' field")
        return answer.strip()
