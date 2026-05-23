# figure_agent/agent/auto_memory.py

import hashlib
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class AutoMemoryNote:
    path: Path
    summary: str
    topic: str
    body: str
    updated_at: str


class AutoMemoryStore:
    """
    File-based long-term memory.

    Layout:
    - memory/MEMORY.md: compact index, never above max_index_lines.
    - memory/<topic>_*.md: topic notes. First line is a <=150 chars summary.

    The store records agent turns automatically, merges overlapping fragments,
    incrementally updates FTS5 entries for changed notes, and keeps full FTS
    rebuilds as a startup or repair path.
    """

    INDEX_FILENAME = "MEMORY.md"
    FTS_DB_FILENAME = "memory_fts.sqlite3"

    def __init__(
        self,
        root_dir: str | Path = "memory",
        max_index_lines: int = 200,
        max_summary_chars: int = 150,
        overlap_threshold: float = 0.62,
    ):
        self.root_dir = Path(root_dir)
        self.index_path = self.root_dir / self.INDEX_FILENAME
        self.fts_db_path = self.root_dir / self.FTS_DB_FILENAME
        self.max_index_lines = max(1, max_index_lines)
        self.max_summary_chars = max(40, max_summary_chars)
        self.overlap_threshold = overlap_threshold
        self.fts_available = False

        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._init_fts()

        if not self.index_path.exists():
            self.rebuild_index()
        else:
            self.rebuild_fts_index()

    def record_turn(
        self,
        user_input: str,
        assistant_answer: str,
        metadata: Optional[Dict[str, str]] = None,
        summary_client=None,
    ) -> Optional[Path]:
        user_input = (user_input or "").strip()
        assistant_answer = (assistant_answer or "").strip()

        if not self._is_memorable(user_input, assistant_answer):
            return None

        metadata = metadata or {}
        topic = self._detect_topic(user_input + "\n" + assistant_answer)
        summary = self._build_summary(
            user_input=user_input,
            assistant_answer=assistant_answer,
            summary_client=summary_client,
        )
        path = self._unique_note_path(topic, summary)
        body = self._build_note_body(
            summary=summary,
            topic=topic,
            user_input=user_input,
            assistant_answer=assistant_answer,
            metadata=metadata,
        )

        path.write_text(body, encoding="utf-8")
        final_path = self._merge_overlapping_fragments(path)
        self.rebuild_index(rebuild_fts=False)
        self._sync_fts_after_memory_write(final_path)
        return final_path

    def search(self, query: str, limit: int = 10) -> List[str]:
        if self.fts_available:
            try:
                fts_hits = self._search_fts(query=query, limit=limit)
                if fts_hits:
                    return fts_hits
            except Exception:
                pass

        notes = self._load_notes()
        if not notes:
            return []

        query = (query or "").strip()
        scored = []

        for index, note in enumerate(notes):
            if query:
                score = self._overlap_score(query, f"{note.summary}\n{note.body}")
            else:
                score = 0.0

            # Keep a tiny recency signal for ties and empty queries.
            score += max(0, len(notes) - index) * 0.0001
            scored.append((score, note))

        scored.sort(key=lambda item: item[0], reverse=True)

        return [
            self._format_search_hit(note, query)
            for score, note in scored[:limit]
            if score > 0
        ]

    def rebuild_index(self, rebuild_fts: bool = True) -> None:
        notes = self._load_notes()
        lines = ["# Auto Memory Index"]

        for note in notes:
            if len(lines) >= self.max_index_lines:
                break

            lines.append(
                f"- {note.updated_at} | {note.topic} | "
                f"[{note.summary}]({note.path.name})"
            )

        self.index_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        if rebuild_fts:
            self.rebuild_fts_index(notes)

    def rebuild_fts_index(self, notes: Optional[List[AutoMemoryNote]] = None) -> None:
        if not self.fts_available:
            return

        notes = notes if notes is not None else self._load_notes()

        with self._connect_fts() as conn:
            conn.execute("DELETE FROM memory_fts")
            for note in notes:
                self._insert_fts_note(conn, note)

    def _merge_overlapping_fragments(self, new_note_path: Path) -> Path:
        notes = self._load_notes()
        new_note = next((note for note in notes if note.path == new_note_path), None)

        if new_note is None:
            return new_note_path

        overlapping = []
        new_text = f"{new_note.summary}\n{new_note.body}"

        for note in notes:
            note_text = f"{note.summary}\n{note.body}"

            if note.path == new_note.path:
                overlapping.append(note)
                continue

            if note.topic != new_note.topic:
                continue

            if self._overlap_score(new_text, note_text) >= self.overlap_threshold:
                overlapping.append(note)

        if len(overlapping) < 2:
            return new_note_path

        merged_summary = self._build_merged_summary(overlapping)
        merged_topic = new_note.topic
        merged_path = self._unique_note_path(merged_topic, merged_summary, prefix="merged")
        merged_body = self._build_merged_body(
            summary=merged_summary,
            topic=merged_topic,
            notes=overlapping,
        )

        merged_path.write_text(merged_body, encoding="utf-8")

        for note in overlapping:
            if note.path.exists() and note.path != merged_path:
                note.path.unlink()

        return merged_path

    def _load_notes(self) -> List[AutoMemoryNote]:
        notes = []

        for path in self.root_dir.glob("*.md"):
            note = self._load_note(path)
            if note is None:
                continue

            notes.append(note)

        notes.sort(key=lambda note: note.path.stat().st_mtime, reverse=True)
        return notes

    def _load_note(self, path: Path) -> Optional[AutoMemoryNote]:
        if path.name == self.INDEX_FILENAME or not path.exists() or path.suffix.lower() != ".md":
            return None

        try:
            body = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return None

        summary = self._extract_summary(body)
        topic = self._extract_field(body, "主题") or self._topic_from_filename(path)
        updated_at = self._extract_field(body, "更新时间") or self._mtime_text(path)
        return AutoMemoryNote(
            path=path,
            summary=summary,
            topic=topic,
            body=body,
            updated_at=updated_at,
        )

    def _build_note_body(
        self,
        summary: str,
        topic: str,
        user_input: str,
        assistant_answer: str,
        metadata: Dict[str, str],
    ) -> str:
        now = self._now_text()
        metadata_lines = [
            f"- {key}: {value}"
            for key, value in metadata.items()
            if key and value
        ]
        metadata_text = "\n".join(metadata_lines) if metadata_lines else "- source: agent-session"

        return (
            f"摘要: {summary}\n"
            f"更新时间: {now}\n"
            f"主题: {topic}\n"
            "来源: auto-memory\n\n"
            "## 元数据\n"
            f"{metadata_text}\n\n"
            "## 用户输入\n"
            f"{self._truncate(user_input, 1600)}\n\n"
            "## 助手回答摘要\n"
            f"{self._truncate(assistant_answer, 2400)}\n"
        )

    def _build_merged_body(
        self,
        summary: str,
        topic: str,
        notes: List[AutoMemoryNote],
    ) -> str:
        now = self._now_text()
        summary_lines = "\n".join(
            f"- {note.summary}"
            for note in notes
        )
        source_lines = "\n".join(
            f"- {note.path.name}"
            for note in notes
        )
        fragment_lines = []

        for note in notes:
            fragment_lines.append(
                f"### {note.path.name}\n"
                f"{self._truncate(self._strip_summary_header(note.body), 1200)}"
            )

        return (
            f"摘要: {summary}\n"
            f"更新时间: {now}\n"
            f"主题: {topic}\n"
            "来源: auto-memory-merge\n\n"
            "## 合并概括\n"
            f"{summary_lines}\n\n"
            "## 被替代文件\n"
            f"{source_lines}\n\n"
            "## 原始片段摘要\n"
            f"{'\n\n'.join(fragment_lines)}\n"
        )

    def _build_summary(
        self,
        user_input: str,
        assistant_answer: str,
        summary_client=None,
    ) -> str:
        if summary_client is not None:
            llm_summary = self._build_llm_summary(
                client=summary_client,
                user_input=user_input,
                assistant_answer=assistant_answer,
            )
            if llm_summary:
                return self._trim_summary(llm_summary)

        user_text = self._clean_inline(user_input)
        answer_text = self._clean_inline(assistant_answer)

        if user_text:
            summary = f"用户关注：{user_text}"
        else:
            summary = f"会话记录：{answer_text}"

        return self._trim_summary(summary)

    def _build_llm_summary(
        self,
        client,
        user_input: str,
        assistant_answer: str,
    ) -> str:
        prompt = (
            "请为下面这轮对话生成一句长期记忆摘要，要求：\n"
            "1. 只输出一句话，不要解释。\n"
            "2. 不超过 150 个字符。\n"
            "3. 保留用户偏好、项目决策、关键实现细节或长期上下文。\n"
            "4. 不要保存临时寒暄。\n\n"
            f"用户：{self._truncate(user_input, 1200)}\n\n"
            f"助手：{self._truncate(assistant_answer, 1800)}"
        )
        messages = [
            {
                "role": "system",
                "content": "你是长期记忆摘要器，只输出一句中文摘要。",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        try:
            response = client.chat_with_messages(
                messages=messages,
                stream=False,
                max_tokens=120,
            )
            content = response.choices[0].message.content.strip()
        except Exception:
            return ""

        first_line = content.splitlines()[0].strip() if content else ""
        first_line = re.sub(r"^[-*#\d.、\s]+", "", first_line).strip()
        return first_line.strip("\"'“”")

    def _build_merged_summary(self, notes: List[AutoMemoryNote]) -> str:
        if not notes:
            return "合并后的长期记忆"

        topic = notes[0].topic
        first = notes[0].summary
        if len(notes) == 2:
            summary = f"{topic} 相关重叠记忆合并：{first}"
        else:
            summary = f"{topic} 相关重叠记忆合并：{first}；另含 {len(notes) - 1} 条片段"

        return self._trim_summary(summary)

    def _trim_summary(self, text: str) -> str:
        text = self._clean_inline(text)
        if len(text) <= self.max_summary_chars:
            return text

        return text[: self.max_summary_chars - 1].rstrip() + "…"

    def _unique_note_path(
        self,
        topic: str,
        summary: str,
        prefix: str = "note",
    ) -> Path:
        now = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        digest = hashlib.md5(summary.encode("utf-8")).hexdigest()[:8]
        slug = self._slug(topic)
        path = self.root_dir / f"{slug}_{prefix}_{now}_{digest}.md"

        counter = 1
        while path.exists():
            path = self.root_dir / f"{slug}_{prefix}_{now}_{digest}_{counter}.md"
            counter += 1

        return path

    def _detect_topic(self, text: str) -> str:
        lower_text = text.lower()
        topic_keywords = [
            ("rag", ("rag", "检索", "召回", "幻觉", "证据", "rerank", "embedding", "bm25", "mrr")),
            ("memory", ("memory", "记忆", "上下文", "context", "summary")),
            ("safety", ("hook", "安全", "拒答", "权限", "policy")),
            ("frontend", ("前端", "上传", "react", "页面", "组件", "ui")),
            ("testing", ("测试", "pytest", "评估", "指标", "recall", "hallucination")),
            ("paper", ("论文", "paper", "文献", "模型", "transformer", "cnn", "lstm", "rnn")),
        ]

        for topic, keywords in topic_keywords:
            if any(keyword in lower_text for keyword in keywords):
                return topic

        return "general"

    def _is_memorable(self, user_input: str, assistant_answer: str) -> bool:
        text = f"{user_input}\n{assistant_answer}".strip()
        if len(text) < 20:
            return False

        noise_patterns = (
            "你好",
            "hello",
            "谢谢",
            "ok",
            "好的",
        )
        lower_text = text.lower()
        if len(text) < 80 and any(pattern in lower_text for pattern in noise_patterns):
            return False

        return True

    def _overlap_score(self, left: str, right: str) -> float:
        left_terms = set(self._terms(left))
        right_terms = set(self._terms(right))

        if not left_terms or not right_terms:
            return 0.0

        return len(left_terms & right_terms) / len(left_terms | right_terms)

    def _init_fts(self) -> None:
        try:
            with self._connect_fts() as conn:
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts
                    USING fts5(
                        path UNINDEXED,
                        summary,
                        topic,
                        body,
                        updated_at UNINDEXED,
                        terms
                    )
                    """
                )
            self.fts_available = True
        except sqlite3.Error:
            self.fts_available = False

    def _connect_fts(self):
        return sqlite3.connect(self.fts_db_path)

    def _sync_fts_after_memory_write(self, final_path: Path) -> None:
        if not self.fts_available:
            return

        note = self._load_note(final_path)
        current_path_names = {
            path.name
            for path in self.root_dir.glob("*.md")
            if path.name != self.INDEX_FILENAME
        }

        with self._connect_fts() as conn:
            self._delete_missing_fts_paths(conn, current_path_names)
            if note is not None:
                self._upsert_fts_note(conn, note)

    def _delete_missing_fts_paths(self, conn, current_path_names: set[str]) -> None:
        rows = conn.execute("SELECT path FROM memory_fts").fetchall()
        stale_paths = [
            (row[0],)
            for row in rows
            if row[0] not in current_path_names
        ]
        if stale_paths:
            conn.executemany("DELETE FROM memory_fts WHERE path = ?", stale_paths)

    def _upsert_fts_note(self, conn, note: AutoMemoryNote) -> None:
        conn.execute("DELETE FROM memory_fts WHERE path = ?", (note.path.name,))
        self._insert_fts_note(conn, note)

    def _insert_fts_note(self, conn, note: AutoMemoryNote) -> None:
        searchable_terms = " ".join(
            self._terms(f"{note.summary}\n{note.topic}\n{note.body}")
        )
        conn.execute(
            """
            INSERT INTO memory_fts(path, summary, topic, body, updated_at, terms)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                note.path.name,
                note.summary,
                note.topic,
                note.body,
                note.updated_at,
                searchable_terms,
            ),
        )

    def _search_fts(self, query: str, limit: int) -> List[str]:
        query = (query or "").strip()
        if not query:
            return []

        fts_query = self._build_fts_query(query)
        if not fts_query:
            return []

        with self._connect_fts() as conn:
            rows = conn.execute(
                """
                SELECT path, summary, topic, body, updated_at, bm25(memory_fts) AS rank
                FROM memory_fts
                WHERE memory_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, limit),
            ).fetchall()

        return [
            self._format_search_hit(
                AutoMemoryNote(
                    path=self.root_dir / row[0],
                    summary=row[1],
                    topic=row[2],
                    body=row[3],
                    updated_at=row[4],
                ),
                query,
            )
            for row in rows
        ]

    def _build_fts_query(self, query: str) -> str:
        terms = []

        for term in self._terms(query):
            clean_term = re.sub(r'["]', "", term).strip()
            if not clean_term or clean_term in terms:
                continue

            terms.append(clean_term)

            if len(terms) >= 32:
                break

        return " OR ".join(f'"{term}"' for term in terms)

    def _format_search_hit(self, note: AutoMemoryNote, query: str) -> str:
        snippet = self._build_snippet(note.body, query)

        if snippet:
            return f"{note.summary} [{note.path.name}]\n  片段: {snippet}"

        return f"{note.summary} [{note.path.name}]"

    def _build_snippet(self, body: str, query: str, max_chars: int = 260) -> str:
        body = self._strip_summary_header(body)
        query_terms = self._terms(query)

        best_line = ""
        best_score = 0

        for raw_line in body.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("- source:"):
                continue

            line_terms = set(self._terms(line))
            score = sum(1 for term in query_terms if term in line_terms)

            if score > best_score:
                best_line = line
                best_score = score

        if not best_line:
            best_line = self._clean_inline(body)

        return self._truncate(self._clean_inline(best_line), max_chars)

    @staticmethod
    def _terms(text: str) -> List[str]:
        terms = []
        terms.extend(re.findall(r"[a-zA-Z0-9]+", text.lower()))

        chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
        chinese_text = "".join(chinese_chars)
        for size in (2, 3, 4):
            for index in range(0, max(0, len(chinese_text) - size + 1)):
                terms.append(chinese_text[index:index + size])

        return [
            term
            for term in terms
            if len(term) >= 2
        ]

    @staticmethod
    def _extract_summary(body: str) -> str:
        first_line = body.splitlines()[0].strip() if body.splitlines() else ""
        if first_line.startswith("摘要:"):
            return first_line.split(":", 1)[1].strip()

        return first_line[:150] if first_line else "未命名记忆"

    @staticmethod
    def _extract_field(body: str, field_name: str) -> str:
        pattern = re.compile(rf"^{re.escape(field_name)}:\s*(.+)$", re.M)
        match = pattern.search(body)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _strip_summary_header(body: str) -> str:
        lines = body.splitlines()
        if lines and lines[0].startswith("摘要:"):
            return "\n".join(lines[1:]).strip()

        return body.strip()

    @staticmethod
    def _clean_inline(text: str) -> str:
        return re.sub(r"\s+", " ", text or "").strip()

    @staticmethod
    def _slug(text: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
        return slug or "memory"

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        text = text or ""
        if len(text) <= max_chars:
            return text

        return text[: max_chars - 16].rstrip() + "\n...[truncated]"

    @staticmethod
    def _topic_from_filename(path: Path) -> str:
        return path.stem.split("_", 1)[0] or "general"

    @staticmethod
    def _mtime_text(path: Path) -> str:
        return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _now_text() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
