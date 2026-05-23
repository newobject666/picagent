import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from apps.papers.arxiv_crawler import parse_arxiv_feed


ARXIV_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.12345v1</id>
    <updated>2024-01-20T00:00:00Z</updated>
    <published>2024-01-18T00:00:00Z</published>
    <title> Retrieval Augmented Generation for Reliable Agents </title>
    <summary>
      This paper studies retrieval augmented generation for reliable agents.
    </summary>
    <author><name>Alice Zhang</name></author>
    <author><name>Bob Li</name></author>
    <category term="cs.CL" />
    <category term="cs.AI" />
  </entry>
</feed>
"""


def test_parse_arxiv_feed_to_paper_records():
    papers = parse_arxiv_feed(ARXIV_SAMPLE)

    assert len(papers) == 1

    paper = papers[0]
    assert paper.title == "Retrieval Augmented Generation for Reliable Agents"
    assert paper.year == "2024"
    assert paper.keywords == ["cs.CL", "cs.AI"]
    assert paper.source == "arxiv:2401.12345v1"
    assert "Alice Zhang, Bob Li" in paper.summary
    assert "Abstract: This paper studies retrieval augmented generation" in paper.summary


if __name__ == "__main__":
    test_parse_arxiv_feed_to_paper_records()
    print("arXiv crawler tests passed.")
