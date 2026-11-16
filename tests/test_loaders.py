from localrag import load_documents


def test_load_text_file(tmp_path):
    source = tmp_path / "guide.md"
    source.write_text("LocalRAG loads Markdown.", encoding="utf-8")
    docs = load_documents(source)
    assert len(docs) == 1
    assert docs[0].text == "LocalRAG loads Markdown."
    assert docs[0].source == str(source)


def test_load_directory_ignores_unsupported_files(tmp_path):
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"not really an image")
    docs = load_documents(tmp_path)
    assert len(docs) == 1
    assert docs[0].text == "alpha"
