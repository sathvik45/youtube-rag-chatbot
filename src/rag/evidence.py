"""Pure, shared selection policy for scored retrieval evidence.

Both production retrieval and the offline calibration replay use this module.
Keeping the score gate and playlist cap in one place prevents an evaluation
from approving a policy that behaves differently for a real chat request.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Generic, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class ScoredCandidate(Generic[T]):
    """One already-ranked retrieval candidate and its provider score."""

    value: T
    score: float
    video_id: str


@dataclass(frozen=True)
class EvidenceSelection(Generic[T]):
    """The candidates that qualify by score and the final context selection."""

    qualifying: tuple[ScoredCandidate[T], ...]
    selected: tuple[ScoredCandidate[T], ...]


def candidate_fetch_limit(
    *,
    k: int,
    scope_size: int,
    per_video_cap: int | None,
) -> int:
    """Return the ranked pool size needed before evidence selection.

    A playlist needs extra pre-cap candidates so the round-robin policy can
    find evidence from more than one video. This is shared with evaluator
    replay so metrics at a given ``k`` match a production request at that ``k``.
    """
    if k <= 0:
        raise ValueError("k must be positive.")
    if scope_size <= 0:
        raise ValueError("scope_size must be positive.")
    if per_video_cap is not None and per_video_cap <= 0:
        raise ValueError("per_video_cap must be positive when configured.")

    if per_video_cap is None or scope_size == 1:
        return k
    return min(k * 4, k + 20 * scope_size)


def select_evidence(
    candidates: list[ScoredCandidate[T]],
    *,
    k: int,
    per_video_cap: int | None,
    single_video: bool,
    score_threshold: float | None,
) -> EvidenceSelection[T]:
    """Apply the production evidence policy to an already-ranked candidate list.

    Scores equal to the threshold are retained. The threshold deliberately
    runs *before* the playlist cap: a weak chunk is not evidence merely
    because its video has unused room in the final context.
    """
    # Reuse the same value validation as the provider-fetch path. Selection
    # does not need the actual scope size, so use a harmless one-video value.
    candidate_fetch_limit(k=k, scope_size=1, per_video_cap=per_video_cap)
    if score_threshold is not None and not isfinite(score_threshold):
        raise ValueError("score_threshold must be finite when configured.")

    for candidate in candidates:
        if not isfinite(candidate.score):
            raise ValueError("retrieval returned a non-finite score.")

    qualifying = tuple(
        candidate
        for candidate in candidates
        if score_threshold is None or candidate.score >= score_threshold
    )

    if per_video_cap is None or single_video:
        return EvidenceSelection(
            qualifying=qualifying,
            selected=qualifying[:k],
        )

    # Candidates are already ranked by the provider. Round-robin prevents one
    # long video from consuming every context slot, then sorting the chosen
    # indexes restores that relevance order for the generator.
    by_video: dict[str, list[int]] = {}
    for index, candidate in enumerate(qualifying):
        by_video.setdefault(candidate.video_id, []).append(index)

    picked_indexes: list[int] = []
    for rank in range(per_video_cap):
        for indexes in by_video.values():
            if rank < len(indexes):
                picked_indexes.append(indexes[rank])
        if len(picked_indexes) >= k:
            break

    selected = tuple(
        qualifying[index] for index in sorted(picked_indexes)[:k]
    )
    return EvidenceSelection(qualifying=qualifying, selected=selected)
