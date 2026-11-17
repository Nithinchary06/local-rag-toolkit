from pathlib import Path

from localrag import load_documents


def _write_pdf(path: Path, page_texts: list[str]) -> None:
    """Write a tiny text PDF fixture without external tools or network access."""
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        (
            f"<< /Type /Pages /Kids [{' '.join(f'{4 + 2 * i} 0 R' for i in range(len(page_texts)))}] "
            f"/Count {len(page_texts)} >>"
        ).encode("ascii"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    for index, text in enumerate(page_texts):
        page_object = 4 + 2 * index
        content_object = page_object + 1
        escaped_text = (
            text.replace("\\", "\\\\")
            .replace("(", "\\(")
            .replace(")", "\\)")
        )
        stream = (
            f"BT /F1 12 Tf 72 720 Td ({escaped_text}) Tj ET".encode("ascii")
            if text
            else b""
        )
        page = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> "
            f"/Contents {content_object} 0 R >>"
        ).encode("ascii")
        content = (
            f"<< /Length {len(stream)} >>\nstream\n".encode("ascii")
            + stream
            + b"\nendstream"
        )
        assert page_object == len(objects) + 1
        objects.extend([page, content])

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for object_number, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{object_number} 0 obj\n".encode("ascii"))
        pdf.extend(body)
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    path.write_bytes(pdf)


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


def test_load_pdf_pages_separately_with_one_based_page_metadata(tmp_path):
    source = tmp_path / "guide.pdf"
    _write_pdf(source, ["First page instructions.", "Second page details."])

    docs = load_documents(source)

    assert len(docs) == 2
    assert [doc.text for doc in docs] == [
        "First page instructions.",
        "Second page details.",
    ]
    assert [doc.metadata["page"] for doc in docs] == [1, 2]
    assert all(doc.source == str(source) for doc in docs)


def test_load_pdf_skips_empty_pages_and_preserves_original_page_numbers(tmp_path):
    source = tmp_path / "with-empty-page.pdf"
    _write_pdf(source, ["Page one content.", "", "Page three content."])

    docs = load_documents(source)

    assert [doc.text for doc in docs] == [
        "Page one content.",
        "Page three content.",
    ]
    assert [doc.metadata["page"] for doc in docs] == [1, 3]
    assert all(doc.source == str(source) for doc in docs)
