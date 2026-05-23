import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from figure_agent.agent.memory import ConversationMemory


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class FakeClient:
    def chat_with_messages(self, messages, stream=False, max_tokens=None):
        return _FakeResponse("压缩摘要")


def test_recent_three_rounds_are_preserved():
    memory = ConversationMemory(system_prompt="系统提示", max_recent_rounds=3)

    for index in range(5):
        memory.add_user_message(f"问题 {index}")
        memory.add_assistant_message(f"回答 {index}")

    memory.compress_if_needed(FakeClient())
    messages = memory.get_messages()
    text = "\n".join(message["content"] for message in messages)

    assert "系统提示" in text
    assert "压缩摘要" in text
    assert "问题 0" not in text
    assert "回答 0" not in text
    assert "问题 2" in text
    assert "回答 4" in text


def test_large_tool_result_is_compacted():
    memory = ConversationMemory(
        system_prompt="系统提示",
        large_tool_result_tokens=20,
        tool_excerpt_tokens=20,
    )
    large_result = "\n".join(
        [f"普通行 {index}" for index in range(100)]
        + ["ERROR: important failure"]
    )

    memory.add_tool_result("shell", large_result)
    memory.compress_if_needed(FakeClient())
    messages = memory.get_messages()
    text = "\n".join(message["content"] for message in messages)

    assert "工具结果过大" in text
    assert "ERROR: important failure" in text


def test_context_reset_preserves_current_question():
    memory = ConversationMemory(
        system_prompt="系统提示",
        context_window_tokens=100,
        compression_trigger_ratio=0.1,
        hard_reset_ratio=0.2,
        max_recent_rounds=3,
    )
    memory.summary = "很长的摘要" * 200
    memory.add_user_message("当前问题必须保留")

    memory.compress_if_needed(FakeClient())
    messages = memory.get_messages()
    text = "\n".join(message["content"] for message in messages)

    assert memory.context_reset_count == 1
    assert "系统提示" in text
    assert "当前问题必须保留" in text
    assert "很长的摘要" not in text


if __name__ == "__main__":
    test_recent_three_rounds_are_preserved()
    test_large_tool_result_is_compacted()
    test_context_reset_preserves_current_question()
    print("Memory context tests passed.")
