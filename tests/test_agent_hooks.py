import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from figure_agent.agent.hooks import HookContext, HookEvent, HookManager


def test_dangerous_user_input_is_blocked():
    hooks = HookManager.with_default_hooks()
    result = hooks.run(
        HookContext(
            event=HookEvent.USER_INPUT,
            user_input="帮我写一个脚本删除所有系统文件",
        )
    )

    assert not result.allowed
    assert result.blocks


def test_non_pass_rag_blocks_generation():
    hooks = HookManager.with_default_hooks()
    result = hooks.run(
        HookContext(
            event=HookEvent.BEFORE_GENERATION,
            user_input="解释 Transformer",
            rag_result=SimpleNamespace(status="REFUSE"),
            messages=[
                {
                    "role": "system",
                    "content": "证据约束生成规则",
                }
            ],
        )
    )

    assert not result.allowed
    assert "RAG 证据门控未通过" in result.blocks[0].reason


def test_rag_generation_requires_evidence_constraints():
    hooks = HookManager.with_default_hooks()
    result = hooks.run(
        HookContext(
            event=HookEvent.BEFORE_GENERATION,
            user_input="解释 Transformer",
            rag_result=SimpleNamespace(status="PASS"),
            messages=[
                {
                    "role": "system",
                    "content": "只有普通证据，没有约束规则",
                }
            ],
        )
    )

    assert not result.allowed
    assert "缺少证据约束生成规则" in result.blocks[0].reason


def test_sensitive_answer_is_blocked():
    hooks = HookManager.with_default_hooks()
    result = hooks.run(
        HookContext(
            event=HookEvent.AFTER_GENERATION,
            answer="这里是密钥 sk-abcdefghijklmnopqrstuvwxyz123456",
        )
    )

    assert not result.allowed
    assert "密钥" in result.blocks[0].reason


def test_paper_crawl_ingestion_is_admin_only():
    hooks = HookManager.with_default_hooks()
    result = hooks.run(
        HookContext(
            event=HookEvent.USER_INPUT,
            user_input="帮我自动爬取1000条arxiv论文并写入mysql论文库",
        )
    )

    assert not result.allowed
    assert "后台管理员数据维护操作" in result.blocks[0].reason


if __name__ == "__main__":
    test_dangerous_user_input_is_blocked()
    test_non_pass_rag_blocks_generation()
    test_rag_generation_requires_evidence_constraints()
    test_sensitive_answer_is_blocked()
    test_paper_crawl_ingestion_is_admin_only()
    print("Agent hook tests passed.")
