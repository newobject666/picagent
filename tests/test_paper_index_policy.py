import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from figure_agent.rag.paper_index_policy import choose_paper_index_update_mode


def test_small_append_batch_uses_incremental_update():
    decision = choose_paper_index_update_mode(
        added_count=100,
        incremental_limit=100,
        existing_index_ready=True,
    )

    assert decision.mode == "incremental"


def test_large_append_batch_retrains_cluster_index():
    decision = choose_paper_index_update_mode(
        added_count=101,
        incremental_limit=100,
        existing_index_ready=True,
    )

    assert decision.mode == "retrain"


def test_updates_and_deactivations_retrain_because_vectors_are_not_append_only():
    updated = choose_paper_index_update_mode(
        added_count=1,
        updated_count=1,
        existing_index_ready=True,
    )
    deactivated = choose_paper_index_update_mode(
        added_count=1,
        deactivated_count=1,
        existing_index_ready=True,
    )

    assert updated.mode == "retrain"
    assert deactivated.mode == "retrain"
