import io
import json
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.apps.chat.document_parser import build_document_context, parse_uploaded_document
from figure_agent.agent.mcp_tools import MCPToolRegistry


class _Doc:
    id = 1
    filename = "notes.md"
    content_type = "text/markdown"
    extracted_text = "Transformer uses self-attention."


def test_parse_markdown_document():
    parsed = parse_uploaded_document(
        io.BytesIO("标题\n\nTransformer 注意力机制".encode("utf-8")),
        "notes.md",
    )

    assert parsed.extension == ".md"
    assert "Transformer" in parsed.text
    assert parsed.char_count > 0


def test_parse_docx_document_with_stdlib_zip_reader():
    buffer = io.BytesIO()
    xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body><w:p><w:r><w:t>Document context works</w:t></w:r></w:p></w:body>"
        "</w:document>"
    )
    with zipfile.ZipFile(buffer, "w") as docx_zip:
        docx_zip.writestr("word/document.xml", xml)

    parsed = parse_uploaded_document(io.BytesIO(buffer.getvalue()), "case.docx")

    assert parsed.extension == ".docx"
    assert "Document context works" in parsed.text


def test_build_document_context_formats_source_metadata():
    context = build_document_context([_Doc()])

    assert "上传文档1" in context
    assert "notes.md" in context
    assert "Transformer uses self-attention" in context


def test_mcp_registry_loads_manifest(tmp_path, monkeypatch):
    manifest = tmp_path / "mcp_tools.json"
    manifest.write_text(
        json.dumps(
            {
                "tools": [
                    {
                        "name": "paper_meta_search",
                        "description": "Search external paper metadata",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MCP_TOOL_MANIFEST_PATH", str(manifest))

    registry = MCPToolRegistry.from_env()
    specs = registry.list_tool_specs()

    assert specs[0].name == "mcp:paper_meta_search"
    assert "MCP Server" in specs[0].description
