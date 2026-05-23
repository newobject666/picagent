import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from figure_agent.agent.research_agent import ResearchAgent


def test_lightweight_greeting_skips_dynamic_planner(tmp_path):
    agent = ResearchAgent.__new__(ResearchAgent)
    agent.runtime_documents = []

    assert agent._should_use_lightweight_reply("你好")
    assert agent._build_lightweight_reply("你好") == "你好，我在。"


def test_research_question_does_not_use_lightweight_reply():
    agent = ResearchAgent.__new__(ResearchAgent)
    agent.runtime_documents = []

    assert not agent._should_use_lightweight_reply("帮我总结 Transformer 论文")


def test_uploaded_document_disables_lightweight_reply():
    agent = ResearchAgent.__new__(ResearchAgent)
    agent.runtime_documents = [
        {
            "filename": "note.md",
            "text": "some document",
        }
    ]

    assert not agent._should_use_lightweight_reply("你好")
