import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO
from xml.etree import ElementTree


SUPPORTED_DOCUMENT_EXTENSIONS = {".txt", ".md", ".markdown", ".docx", ".pdf"}
MAX_EXTRACTED_CHARS = 120_000


@dataclass
class ParsedDocument:
    filename: str
    extension: str
    text: str

    @property
    def char_count(self) -> int:
        return len(self.text)


def parse_uploaded_document(file_obj: BinaryIO, filename: str) -> ParsedDocument:
    extension = Path(filename).suffix.lower()
    raw = file_obj.read()

    if extension not in SUPPORTED_DOCUMENT_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_DOCUMENT_EXTENSIONS))
        raise ValueError(f"暂不支持 {extension or 'unknown'} 文件，支持格式：{supported}")

    if extension in {".txt", ".md", ".markdown"}:
        text = _decode_text(raw)
    elif extension == ".docx":
        text = _extract_docx_text(raw)
    else:
        text = _extract_pdf_text(raw)

    text = _normalize_text(text)
    if not text:
        raise ValueError("文档解析后没有可用文本")

    return ParsedDocument(
        filename=filename,
        extension=extension,
        text=text[:MAX_EXTRACTED_CHARS],
    )


def build_document_context(documents, max_total_chars: int = 14_000) -> str:
    chunks = []
    total_chars = 0

    for index, document in enumerate(documents, 1):
        text = getattr(document, "extracted_text", "") or ""
        if not text:
            continue

        budget = max_total_chars - total_chars
        if budget <= 0:
            break

        header = (
            f"【上传文档{index}】\n"
            f"Document ID: {getattr(document, 'id', 'N/A')}\n"
            f"Filename: {getattr(document, 'filename', 'unknown')}\n"
            f"Content Type: {getattr(document, 'content_type', '') or 'unknown'}\n"
            "Parsed Text:\n"
        )
        available = max(0, budget - len(header) - 20)
        snippet = text[:available]
        if len(text) > len(snippet):
            snippet += "\n...[文档内容已截断]"

        chunk = header + snippet
        chunks.append(chunk)
        total_chars += len(chunk)

    return "\n\n".join(chunks).strip()


def _decode_text(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def _extract_docx_text(raw: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(raw)) as docx_zip:
        with docx_zip.open("word/document.xml") as document_xml:
            root = ElementTree.fromstring(document_xml.read())

    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs = []

    for paragraph in root.findall(".//w:p", namespace):
        parts = [
            node.text
            for node in paragraph.findall(".//w:t", namespace)
            if node.text
        ]
        if parts:
            paragraphs.append("".join(parts))

    return "\n".join(paragraphs)


def _extract_pdf_text(raw: bytes) -> str:
    try:
        from pypdf import PdfReader
    except Exception as exc:
        raise ValueError("PDF 解析需要安装 pypdf 依赖") from exc

    reader = PdfReader(io.BytesIO(raw))
    pages = []

    for page in reader.pages:
        pages.append(page.extract_text() or "")

    return "\n".join(pages)


def _normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
