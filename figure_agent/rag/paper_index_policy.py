from dataclasses import dataclass


@dataclass(frozen=True)
class PaperIndexUpdateDecision:
    mode: str
    reason: str


def choose_paper_index_update_mode(
    added_count: int,
    updated_count: int = 0,
    deactivated_count: int = 0,
    clear: bool = False,
    incremental_limit: int = 100,
    existing_index_ready: bool = True,
) -> PaperIndexUpdateDecision:
    """
    Decide how to refresh the clustered paper vector index after a write batch.

    Small pure append batches can reuse existing IVF centroids. Replacements and
    removals need a full rebuild because a plain IVF index cannot safely update
    or delete old vectors in place.
    """
    added_count = max(0, added_count)
    updated_count = max(0, updated_count)
    deactivated_count = max(0, deactivated_count)
    incremental_limit = max(1, incremental_limit)

    if clear:
        return PaperIndexUpdateDecision("retrain", "corpus was cleared before insert")

    if deactivated_count > 0:
        return PaperIndexUpdateDecision("retrain", "inactive papers require vector removal")

    if updated_count > 0:
        return PaperIndexUpdateDecision("retrain", "updated papers require vector replacement")

    if added_count == 0:
        return PaperIndexUpdateDecision("skip", "no new papers were added")

    if not existing_index_ready:
        return PaperIndexUpdateDecision("retrain", "no trained index is available")

    if added_count <= incremental_limit:
        return PaperIndexUpdateDecision(
            "incremental",
            f"append batch size {added_count} <= incremental limit {incremental_limit}",
        )

    return PaperIndexUpdateDecision(
        "retrain",
        f"append batch size {added_count} > incremental limit {incremental_limit}",
    )
