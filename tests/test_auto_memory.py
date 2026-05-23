import sys
from tempfile import TemporaryDirectory
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from figure_agent.agent.auto_memory import AutoMemoryStore


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class FakeSummaryClient:
    def __init__(self, summary):
        self.summary = summary

    def chat_with_messages(self, messages, stream=False, max_tokens=None):
        return _FakeResponse(self.summary)


def _note_files(root: Path):
    return [
        path
        for path in root.glob("*.md")
        if path.name != AutoMemoryStore.INDEX_FILENAME
    ]


def test_auto_memory_records_summary_header_and_index(tmp_path):
    store = AutoMemoryStore(root_dir=tmp_path)

    path = store.record_turn(
        user_input="实现 RAG 幻觉防控和召回率测试",
        assistant_answer="已实现 BM25、向量检索、Rerank、证据覆盖校验和拒答机制。",
    )

    assert path is not None
    assert path.exists()

    first_line = path.read_text(encoding="utf-8").splitlines()[0]
    summary = first_line.replace("摘要:", "", 1).strip()
    index_text = (tmp_path / "MEMORY.md").read_text(encoding="utf-8")

    assert first_line.startswith("摘要:")
    assert len(summary) <= 150
    assert summary in index_text
    assert path.name in index_text


def test_auto_memory_merges_overlapping_fragments_and_replaces_old_files(tmp_path):
    store = AutoMemoryStore(root_dir=tmp_path, overlap_threshold=0.30)

    first_path = store.record_turn(
        user_input="实现 RAG 幻觉防控策略，包含 BM25 向量检索 Rerank 拒答机制",
        assistant_answer="完成三信号相关性判断、证据覆盖校验和拒答策略。",
    )
    second_path = store.record_turn(
        user_input="继续优化 RAG 幻觉防控策略，BM25 向量检索 Rerank 和证据覆盖校验",
        assistant_answer="补充召回率测试、幻觉率测试以及证据约束生成。",
    )

    notes = _note_files(tmp_path)
    index_text = (tmp_path / "MEMORY.md").read_text(encoding="utf-8")

    assert first_path is not None
    assert second_path is not None
    assert len(notes) == 1
    assert notes[0].name == second_path.name
    assert "来源: auto-memory-merge" in notes[0].read_text(encoding="utf-8")
    assert first_path.name not in index_text
    assert second_path.name in index_text


def test_auto_memory_index_is_capped(tmp_path):
    store = AutoMemoryStore(root_dir=tmp_path, max_index_lines=5)

    for index in range(12):
        store.record_turn(
            user_input=f"记录第 {index} 个不同主题的长期事实，包含 unique_{index}",
            assistant_answer=f"这是第 {index} 个长期记忆内容，避免与其他文件高度重叠。",
        )

    lines = (tmp_path / "MEMORY.md").read_text(encoding="utf-8").splitlines()

    assert len(lines) <= 5


def test_auto_memory_uses_llm_summary_when_available(tmp_path):
    store = AutoMemoryStore(root_dir=tmp_path)
    summary = "用户偏好：跨对话记忆需要 FTS5 检索和简短中文摘要"

    path = store.record_turn(
        user_input="请实现跨对话长期记忆。",
        assistant_answer="已增加 FTS5 检索和长期记忆索引。",
        summary_client=FakeSummaryClient(summary),
    )

    first_line = path.read_text(encoding="utf-8").splitlines()[0]

    assert first_line == f"摘要: {summary}"


def test_auto_memory_fts5_recalls_cross_session_details(tmp_path):
    first_store = AutoMemoryStore(root_dir=tmp_path, overlap_threshold=0.95)
    first_store.record_turn(
        user_input=(
            "长期记住：用户的毕业论文方向是 RAG 幻觉防控，"
            "偏好简短中文总结，并关注 Recall 和 MRR 指标。"
        ),
        assistant_answer=(
            "已记录用户论文方向、回答风格偏好，以及召回率和 MRR "
            "评估指标。"
        ),
        summary_client=FakeSummaryClient(
            "用户论文方向是 RAG 幻觉防控，偏好简短中文总结并关注 Recall/MRR"
        ),
    )

    second_store = AutoMemoryStore(root_dir=tmp_path, overlap_threshold=0.95)
    hits = second_store.search("用户毕业论文方向 Recall MRR 偏好", limit=3)

    assert (tmp_path / AutoMemoryStore.FTS_DB_FILENAME).exists()
    assert hits
    assert "RAG 幻觉防控" in hits[0]
    assert "Recall" in hits[0]
    assert "MRR" in hits[0]


def test_auto_memory_incrementally_updates_fts_on_new_note(tmp_path):
    store = AutoMemoryStore(root_dir=tmp_path, overlap_threshold=0.95)

    def fail_full_rebuild(*args, **kwargs):
        raise AssertionError("record_turn should not rebuild the full FTS index")

    store.rebuild_fts_index = fail_full_rebuild
    path = store.record_turn(
        user_input=(
            "Remember unique_incremental_marker as a durable cross-session "
            "preference for retrieval tests."
        ),
        assistant_answer=(
            "The memory note keeps unique_incremental_marker searchable through "
            "incremental FTS updates."
        ),
    )

    assert path is not None
    hits = store.search("unique_incremental_marker", limit=3)
    assert hits
    assert path.name in hits[0]


def test_auto_memory_merge_deletes_stale_fts_entries_incrementally(tmp_path):
    store = AutoMemoryStore(root_dir=tmp_path, overlap_threshold=0.20)
    first_path = store.record_turn(
        user_input=(
            "memory duplicate cluster alpha beta gamma unique_old_marker "
            "records one persistent preference."
        ),
        assistant_answer=(
            "memory duplicate cluster alpha beta gamma unique_old_marker "
            "is saved for later recall."
        ),
    )

    def fail_full_rebuild(*args, **kwargs):
        raise AssertionError("merge should update changed FTS rows incrementally")

    store.rebuild_fts_index = fail_full_rebuild
    second_path = store.record_turn(
        user_input=(
            "memory duplicate cluster alpha beta gamma unique_new_marker "
            "records an overlapping persistent preference."
        ),
        assistant_answer=(
            "memory duplicate cluster alpha beta gamma unique_new_marker "
            "is merged with the earlier note."
        ),
    )

    assert first_path is not None
    assert second_path is not None
    assert not first_path.exists()
    assert second_path.exists()

    with store._connect_fts() as conn:
        indexed_paths = {
            row[0]
            for row in conn.execute("SELECT path FROM memory_fts").fetchall()
        }

    assert first_path.name not in indexed_paths
    assert second_path.name in indexed_paths


if __name__ == "__main__":
    with TemporaryDirectory() as temp_dir:
        test_auto_memory_records_summary_header_and_index(Path(temp_dir) / "case1")
        test_auto_memory_merges_overlapping_fragments_and_replaces_old_files(
            Path(temp_dir) / "case2"
        )
        test_auto_memory_index_is_capped(Path(temp_dir) / "case3")
        test_auto_memory_uses_llm_summary_when_available(Path(temp_dir) / "case4")
        test_auto_memory_fts5_recalls_cross_session_details(Path(temp_dir) / "case5")
        test_auto_memory_incrementally_updates_fts_on_new_note(Path(temp_dir) / "case6")
        test_auto_memory_merge_deletes_stale_fts_entries_incrementally(
            Path(temp_dir) / "case7"
        )

    print("Auto memory tests passed.")
