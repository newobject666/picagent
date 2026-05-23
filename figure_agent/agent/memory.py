# figure_agent/agent/memory.py

import re
from typing import Dict, List, Optional

from figure_agent.agent.prompt import get_summary_system_prompt


class ConversationMemory:
    """
    三层上下文管理：

    1. 常规历史管理：
       - system prompt 永远保留，不压缩。
       - 最近 max_recent_rounds 轮完整保留。
       - 工具返回结果进入 tool_summaries，大文件自动提取关键片段。

    2. 自动压缩：
       - 接近 200K 上下文窗口时，旧对话压缩成摘要。
       - 大工具结果只保留关键片段。
       - 子 agent 执行结果只保留摘要，不保留完整过程。

    3. Context reset：
       - 压缩后仍超预算时，丢弃旧窗口，只保留 system prompt 和当前问题。
       - 工程状态应外化到文件系统/数据库，新窗口通过文件恢复进度。
    """

    def __init__(
        self,
        system_prompt: str,
        max_recent_rounds: int = 3,
        context_window_tokens: int = 200_000,
        compression_trigger_ratio: float = 0.8,
        hard_reset_ratio: float = 0.98,
        large_tool_result_tokens: int = 8_000,
        tool_excerpt_tokens: int = 1_200,
        summary_max_tokens: int = 2_000,
    ):
        self.system_prompt = system_prompt
        self.max_recent_rounds = max_recent_rounds
        self.context_window_tokens = context_window_tokens
        self.compression_trigger_tokens = int(
            context_window_tokens * compression_trigger_ratio
        )
        self.hard_reset_tokens = int(context_window_tokens * hard_reset_ratio)
        self.large_tool_result_tokens = large_tool_result_tokens
        self.tool_excerpt_tokens = tool_excerpt_tokens
        self.summary_max_tokens = summary_max_tokens

        self.summary: str = ""
        self.recent_messages: List[Dict[str, str]] = []
        self.tool_summaries: List[Dict[str, str]] = []
        self.context_reset_count = 0

    def add_user_message(self, content: str) -> None:
        self.recent_messages.append(
            {
                "role": "user",
                "content": content,
            }
        )

    def add_assistant_message(self, content: str) -> None:
        self.recent_messages.append(
            {
                "role": "assistant",
                "content": content,
            }
        )

    def add_tool_result(
        self,
        tool_name: str,
        content: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        保存工具返回结果。大文件或长输出自动只保留关键片段。
        """
        metadata = metadata or {}
        compact_content = self._compact_tool_content(content)

        self.tool_summaries.append(
            {
                "tool_name": tool_name,
                "content": compact_content,
                "metadata": str(metadata),
            }
        )

    def add_subagent_result(self, agent_name: str, result_summary: str) -> None:
        """
        子 agent 暂未接入。后续多 agent 化时，只保存最终摘要，不保存完整过程。
        """
        self.tool_summaries.append(
            {
                "tool_name": f"subagent:{agent_name}",
                "content": self._truncate_to_token_budget(
                    result_summary,
                    self.tool_excerpt_tokens,
                ),
                "metadata": "subagent summary only",
            }
        )

    def load_messages(self, messages: List[Dict[str, str]]) -> None:
        self.recent_messages = [
            {
                "role": msg["role"],
                "content": msg["content"],
            }
            for msg in messages
            if msg.get("role") in {"user", "assistant"} and msg.get("content")
        ]

    def get_messages(self) -> List[Dict[str, str]]:
        """
        获取发送给 LLM 的 messages。

        必须注入：
        - system prompt
        - 当前问题，已保存在 recent_messages 末尾

        按需注入：
        - 历史摘要
        - 工具结果关键片段
        - 最近 3 轮完整对话
        """
        messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": self.system_prompt,
            }
        ]

        if self.context_reset_count:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        f"上下文窗口已重置 {self.context_reset_count} 次。"
                        "旧执行状态应以项目文件、数据库记录和日志为准。"
                    ),
                }
            )

        if self.summary:
            messages.append(
                {
                    "role": "system",
                    "content": f"以下是此前对话压缩摘要，请按需参考：\n{self.summary}",
                }
            )

        if self.tool_summaries:
            tool_text = self._format_tool_summaries()
            messages.append(
                {
                    "role": "system",
                    "content": f"以下是工具返回结果的关键片段，请按当前任务需要参考：\n{tool_text}",
                }
            )

        messages.extend(self._recent_window_messages())

        return messages

    def clear(self) -> None:
        self.summary = ""
        self.recent_messages = []
        self.tool_summaries = []
        self.context_reset_count = 0

    def remove_last_user_message(self) -> None:
        for index in range(len(self.recent_messages) - 1, -1, -1):
            if self.recent_messages[index]["role"] == "user":
                self.recent_messages.pop(index)
                break

    def compress_if_needed(self, client) -> None:
        """
        执行三层上下文管理。
        """
        self._compact_large_tool_results()
        self._compress_old_dialogue_to_summary(client)

        if self._estimate_messages_tokens(self.get_messages()) < self.compression_trigger_tokens:
            return

        self._compress_summary_if_needed(client)
        self._shrink_tool_summaries()

        if self._estimate_messages_tokens(self.get_messages()) >= self.hard_reset_tokens:
            self._context_reset()

    def _compress_old_dialogue_to_summary(self, client) -> None:
        preserve_count = self.max_recent_rounds * 2

        if self.recent_messages and self.recent_messages[-1]["role"] == "user":
            preserve_count += 1

        if len(self.recent_messages) <= preserve_count:
            return

        old_messages = self.recent_messages[:-preserve_count]
        self.recent_messages = self.recent_messages[-preserve_count:]

        old_text = self._format_messages(old_messages)

        if not old_text.strip():
            return

        new_summary = self._summarize_old_messages(
            client=client,
            old_text=old_text,
        )

        if self.summary:
            self.summary = self._merge_summary(
                client=client,
                old_summary=self.summary,
                new_summary=new_summary,
            )
        else:
            self.summary = new_summary

    def _compress_summary_if_needed(self, client) -> None:
        if not self.summary:
            return

        if self._estimate_tokens(self.summary) <= self.summary_max_tokens:
            return

        try:
            self.summary = self._summarize_text(
                client=client,
                text=self.summary,
                max_tokens=self.summary_max_tokens,
            )
        except Exception:
            self.summary = self._truncate_to_token_budget(
                self.summary,
                self.summary_max_tokens,
            )

    def _compact_large_tool_results(self) -> None:
        for item in self.tool_summaries:
            content = item.get("content", "")
            if self._estimate_tokens(content) > self.large_tool_result_tokens:
                item["content"] = self._compact_tool_content(content)

    def _shrink_tool_summaries(self) -> None:
        self.tool_summaries = self.tool_summaries[-8:]

        for item in self.tool_summaries:
            item["content"] = self._truncate_to_token_budget(
                item.get("content", ""),
                self.tool_excerpt_tokens,
            )

    def _context_reset(self) -> None:
        current_user_message = None

        for message in reversed(self.recent_messages):
            if message.get("role") == "user":
                current_user_message = message
                break

        self.summary = ""
        self.tool_summaries = []
        self.recent_messages = [current_user_message] if current_user_message else []
        self.context_reset_count += 1

    def _recent_window_messages(self) -> List[Dict[str, str]]:
        preserve_count = self.max_recent_rounds * 2

        if self.recent_messages and self.recent_messages[-1]["role"] == "user":
            preserve_count += 1

        return self.recent_messages[-preserve_count:]

    def _summarize_old_messages(self, client, old_text: str) -> str:
        return self._summarize_text(
            client=client,
            text=f"请压缩以下旧对话：\n\n{old_text}",
            max_tokens=800,
        )

    def _merge_summary(
        self,
        client,
        old_summary: str,
        new_summary: str,
    ) -> str:
        text = (
            "请将以下两段历史摘要合并为一段更简洁的摘要。\n\n"
            f"旧摘要：\n{old_summary}\n\n"
            f"新增摘要：\n{new_summary}"
        )

        return self._summarize_text(
            client=client,
            text=text,
            max_tokens=800,
        )

    @staticmethod
    def _summarize_text(client, text: str, max_tokens: int) -> str:
        messages = [
            {
                "role": "system",
                "content": get_summary_system_prompt(),
            },
            {
                "role": "user",
                "content": text,
            },
        ]

        response = client.chat_with_messages(
            messages=messages,
            stream=False,
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content.strip()

    def _compact_tool_content(self, content: str) -> str:
        if self._estimate_tokens(content) <= self.large_tool_result_tokens:
            return content

        key_lines = self._extract_key_lines(content)
        head = self._truncate_to_token_budget(content[:4000], self.tool_excerpt_tokens // 2)
        tail = self._truncate_to_token_budget(content[-4000:], self.tool_excerpt_tokens // 3)

        return (
            "[工具结果过大，已自动保留关键片段]\n\n"
            "【开头片段】\n"
            f"{head}\n\n"
            "【关键行】\n"
            f"{key_lines}\n\n"
            "【结尾片段】\n"
            f"{tail}"
        ).strip()

    def _extract_key_lines(self, content: str) -> str:
        patterns = (
            "error",
            "failed",
            "failure",
            "exception",
            "warning",
            "success",
            "passed",
            "created",
            "updated",
            "deleted",
            "summary",
            "result",
            "todo",
            "文件",
            "路径",
            "错误",
            "失败",
            "成功",
            "结论",
            "摘要",
        )
        lines = []

        for line in content.splitlines():
            lower_line = line.lower()
            if any(pattern in lower_line for pattern in patterns):
                lines.append(line)

            if len(lines) >= 80:
                break

        if not lines:
            return "未提取到明显关键行。"

        return self._truncate_to_token_budget(
            "\n".join(lines),
            self.tool_excerpt_tokens // 2,
        )

    def _format_tool_summaries(self) -> str:
        lines = []

        for index, item in enumerate(self.tool_summaries[-8:], 1):
            lines.append(
                f"【工具结果 {index}: {item.get('tool_name', 'unknown')}】\n"
                f"{item.get('content', '')}"
            )

        return "\n\n".join(lines)

    @staticmethod
    def _format_messages(messages: List[Dict[str, str]]) -> str:
        lines = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "user":
                lines.append(f"用户：{content}")
            elif role == "assistant":
                lines.append(f"助手：{content}")
            else:
                lines.append(f"{role}：{content}")

        return "\n".join(lines)

    def _estimate_messages_tokens(self, messages: List[Dict[str, str]]) -> int:
        return sum(self._estimate_tokens(message.get("content", "")) for message in messages)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        if not text:
            return 0

        ascii_chars = len(re.findall(r"[\x00-\x7F]", text))
        non_ascii_chars = len(text) - ascii_chars

        return max(1, ascii_chars // 4 + non_ascii_chars)

    def _truncate_to_token_budget(self, text: str, token_budget: int) -> str:
        if self._estimate_tokens(text) <= token_budget:
            return text

        char_budget = max(200, token_budget * 2)
        return text[:char_budget].rstrip() + "\n...[truncated]"
