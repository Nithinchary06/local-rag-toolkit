from localrag import Chunk, LocalRAG, SearchResult
from localrag.generators import build_prompt
from localrag.retrieval import TfidfRetriever


def test_build_prompt_includes_question_and_numbered_source_passages():
    results = [
        SearchResult(
            Chunk("Install dependencies with pip.", "setup.md", 0),
            0.92,
        ),
        SearchResult(
            Chunk("Pin package versions in requirements.txt.", "deploy.md", 1),
            0.71,
        ),
    ]

    prompt = build_prompt("How do I install dependencies?", results)

    assert "Question: How do I install dependencies?" in prompt
    assert "[1] Source: setup.md" in prompt
    assert "Install dependencies with pip." in prompt
    assert "[2] Source: deploy.md" in prompt
    assert "Pin package versions in requirements.txt." in prompt


def test_build_prompt_includes_pdf_page_number_in_source_reference():
    results = [
        SearchResult(
            Chunk(
                "The installation steps are listed here.",
                "manual.pdf",
                0,
                {"page": 4},
            ),
            0.88,
        )
    ]

    prompt = build_prompt("Where are installation steps?", results)

    assert "[1] Source: manual.pdf, page 4" in prompt
    assert "The installation steps are listed here." in prompt


def test_build_prompt_explains_when_no_passages_were_retrieved():
    prompt = build_prompt("What is in the document?", [])

    assert "No relevant passages were retrieved." in prompt
    assert "Question: What is in the document?" in prompt
    assert "say that you do not know" in prompt


def test_local_rag_ask_sends_grounded_prompt_to_generator():
    class RecordingGenerator:
        def __init__(self):
            self.prompt = None

        def generate(self, prompt):
            self.prompt = prompt
            return "Install dependencies with pip. [1]"

    chunk = Chunk("Install dependencies with pip.", "setup.md", 0)
    rag = LocalRAG(TfidfRetriever([chunk]))
    generator = RecordingGenerator()

    response = rag.ask("How do I install dependencies?", generator)

    assert response.answer == "Install dependencies with pip. [1]"
    assert generator.prompt is not None
    assert "Question: How do I install dependencies?" in generator.prompt
    assert "[1] Source: setup.md" in generator.prompt
    assert response.sources[0].chunk.source == "setup.md"
