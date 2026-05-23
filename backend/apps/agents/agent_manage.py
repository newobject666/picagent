# backend/apps/agents/agent_manager.py

import logging
import os
from typing import Dict, List

from figure_agent.agent.research_agent import ResearchAgent
from apps.chat.models import ChatMessage
from apps.papers.models import PaperRecord as DBPaperRecord

logger = logging.getLogger("picagent")


class AgentManager:
    """
    维护每个 session 对应的 ResearchAgent 实例。

    Web 版中：
    - MySQL 是论文库主数据源
    - Agent 创建时自动从 MySQL 加载 RAG 论文
    """

    def __init__(self):
        self._agents: Dict[int, ResearchAgent] = {}
        self.default_corpus_id = os.environ.get("RAG_CORPUS_ID", "default")
        self.paper_page_size = int(os.environ.get("RAG_PAPER_PAGE_SIZE", "500"))

    def get_agent(self, session_id: int) -> ResearchAgent:
        if session_id not in self._agents:
            logger.info(f"创建新的 ResearchAgent, session_id={session_id}")

            agent = ResearchAgent(
                skill_dir="Skill",
                max_recent_rounds=3,
                max_tokens=1500,
            )

            records = self._load_mysql_papers(corpus_id=self.default_corpus_id)
            agent.reload_rag_from_records(records)
            self._restore_short_term_memory(agent, session_id)

            logger.info(
                f"新 Agent 已从 MySQL 分页加载 RAG 论文数量: {len(records)}, "
                f"corpus_id={self.default_corpus_id}"
            )

            self._agents[session_id] = agent

        return self._agents[session_id]

    def _load_mysql_papers(
        self,
        corpus_id: str = None,
        page_size: int = None,
    ) -> List[DBPaperRecord]:
        """
        从 MySQL 分页加载指定 corpus 下的 active 论文记录。
        """
        corpus_id = (corpus_id or self.default_corpus_id).strip() or self.default_corpus_id
        page_size = page_size or self.paper_page_size
        base_query = (
            DBPaperRecord.objects
            .filter(corpus_id=corpus_id, is_active=True)
            .exclude(title="")
            .exclude(summary="")
            .order_by("-id")
        )
        total = base_query.count()
        records: List[DBPaperRecord] = []
        offset = 0

        while offset < total:
            page = list(base_query[offset: offset + page_size])
            records.extend(page)
            offset += page_size

        logger.info(
            "MySQL paper_record 分页加载完成: corpus_id=%s, active_count=%s, page_size=%s",
            corpus_id,
            len(records),
            page_size,
        )

        return records

    def _restore_short_term_memory(
        self,
        agent: ResearchAgent,
        session_id: int,
    ) -> None:
        messages = list(
            ChatMessage.objects
            .filter(session_id=session_id, role__in=["user", "assistant"])
            .order_by("created_at")
            .values("role", "content")
        )

        if not messages:
            return

        agent.load_short_term_messages(messages)
        logger.info(
            f"session_id={session_id} 已恢复短期记忆消息数量: {len(messages)}"
        )

    def reload_all_rag_from_mysql(self, corpus_id: str = None) -> int:
        """
        将 MySQL 论文库重新加载到所有已创建 Agent 的 RAG 中。
        """
        corpus_id = (corpus_id or self.default_corpus_id).strip() or self.default_corpus_id
        records = self._load_mysql_papers(corpus_id=corpus_id)

        for session_id, agent in self._agents.items():
            agent.reload_rag_from_records(records)
            logger.info(
                f"session_id={session_id} 的 Agent 已重新加载 RAG, "
                f"corpus_id={corpus_id}, 论文数量={len(records)}"
            )

        return len(records)

    def clear_agent(self, session_id: int) -> None:
        agent = self._agents.get(session_id)
        if agent:
            agent.clear_memory()

    def remove_agent(self, session_id: int) -> None:
        if session_id in self._agents:
            del self._agents[session_id]


agent_manager = AgentManager()
