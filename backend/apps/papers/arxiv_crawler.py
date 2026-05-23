import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Iterable, List


ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


@dataclass
class CrawledPaper:
    title: str
    summary: str
    keywords: List[str]
    year: str
    source: str


def fetch_arxiv_papers(
    query: str,
    max_results: int = 20,
    start: int = 0,
    sort_by: str = "submittedDate",
    sort_order: str = "descending",
    batch_size: int = 100,
    delay_seconds: float = 3.0,
    timeout: int = 30,
    raw_query: bool = False,
) -> List[CrawledPaper]:
    """
    Fetch paper metadata from the official arXiv Atom API.

    The returned records are ready to be written into PaperRecord:
    - MySQL stores readable metadata/text.
    - FAISS can later rebuild vectors from these text records.
    """
    query = query.strip()
    if not query:
        return []

    max_results = max(1, max_results)
    batch_size = max(1, min(batch_size, 100))
    papers: List[CrawledPaper] = []

    while len(papers) < max_results:
        current_batch = min(batch_size, max_results - len(papers))
        raw_xml = fetch_arxiv_page(
            query=query,
            start=start + len(papers),
            max_results=current_batch,
            sort_by=sort_by,
            sort_order=sort_order,
            timeout=timeout,
            raw_query=raw_query,
        )
        page_papers = parse_arxiv_feed(raw_xml)

        if not page_papers:
            break

        papers.extend(page_papers)

        if len(papers) >= max_results or len(page_papers) < current_batch:
            break

        time.sleep(delay_seconds)

    return papers[:max_results]


def fetch_arxiv_page(
    query: str,
    start: int,
    max_results: int,
    sort_by: str,
    sort_order: str,
    timeout: int,
    raw_query: bool = False,
) -> bytes:
    search_query = query if raw_query else f"all:{query}"
    params = {
        "search_query": search_query,
        "start": str(start),
        "max_results": str(max_results),
        "sortBy": sort_by,
        "sortOrder": sort_order,
    }
    url = f"{ARXIV_API_URL}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "PicAgentPaperCrawler/1.0 (metadata ingestion)",
        },
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def parse_arxiv_feed(raw_xml: bytes | str) -> List[CrawledPaper]:
    root = ET.fromstring(raw_xml)
    papers = []

    for entry in root.findall("atom:entry", ATOM_NS):
        title = _text(entry, "atom:title")
        abstract = _text(entry, "atom:summary")
        published = _text(entry, "atom:published")
        arxiv_url = _text(entry, "atom:id")
        arxiv_id = arxiv_url.rstrip("/").split("/")[-1] if arxiv_url else ""
        authors = [
            _text(author, "atom:name")
            for author in entry.findall("atom:author", ATOM_NS)
        ]
        authors = [author for author in authors if author]
        categories = [
            category.attrib.get("term", "").strip()
            for category in entry.findall("atom:category", ATOM_NS)
        ]
        categories = [category for category in categories if category]
        year = published[:4] if len(published) >= 4 else ""
        source = f"arxiv:{arxiv_id}"[:100] if arxiv_id else "arxiv"
        summary = build_summary(
            abstract=abstract,
            authors=authors,
            arxiv_id=arxiv_id,
            arxiv_url=arxiv_url,
        )

        if title and summary:
            papers.append(
                CrawledPaper(
                    title=title,
                    summary=summary,
                    keywords=categories,
                    year=year,
                    source=source,
                )
            )

    return papers


def build_summary(
    abstract: str,
    authors: Iterable[str],
    arxiv_id: str,
    arxiv_url: str,
) -> str:
    authors_text = ", ".join(list(authors)[:8])
    parts = []

    if arxiv_id:
        parts.append(f"ArXiv ID: {arxiv_id}")

    if arxiv_url:
        parts.append(f"URL: {arxiv_url}")

    if authors_text:
        parts.append(f"Authors: {authors_text}")

    if abstract:
        parts.append(f"Abstract: {_normalize_space(abstract)}")

    return "\n".join(parts)


def _text(node: ET.Element, path: str) -> str:
    found = node.find(path, ATOM_NS)
    if found is None or found.text is None:
        return ""

    return _normalize_space(found.text)


def _normalize_space(text: str) -> str:
    return " ".join((text or "").split())
