"""Clip-level alignment quality metrics.

Extracted from notebooks/foreign_whispers_pipeline.ipynb (M8-align).
Imports from foreign_whispers.alignment — no other dependencies.
"""
import statistics as _stats

from foreign_whispers.alignment import (
    AlignAction,
    AlignedSegment,
    SegmentMetrics,
    decide_action,
)


def clip_evaluation_report(
    metrics: list[SegmentMetrics],
    aligned: list[AlignedSegment],
) -> dict:
    """Return a summary dict of alignment quality metrics for one clip.

    Keys:
        mean_abs_duration_error_s: Mean |predicted_tts_s - source_duration_s| per segment.
        pct_severe_stretch: % of aligned segments with stretch_factor > 1.4.
        n_gap_shifts: Number of segments resolved via gap-shift.
        n_translation_retries: Number of segments that required re-ranking.
        total_cumulative_drift_s: End-to-end drift introduced by gap-shifts.
    """
    if not metrics:
        return {
            "mean_abs_duration_error_s": 0.0,
            "pct_severe_stretch":        0.0,
            "n_gap_shifts":              0,
            "n_translation_retries":     0,
            "total_cumulative_drift_s":  0.0,
        }

    errors    = [abs(m.predicted_tts_s - m.source_duration_s) for m in metrics]
    n_severe  = sum(1 for a in aligned if a.stretch_factor > 1.4)
    n_shifted = sum(1 for a in aligned if a.action == AlignAction.GAP_SHIFT)
    n_retry   = sum(1 for m in metrics if decide_action(m) == AlignAction.REQUEST_SHORTER)
    drift     = (
        aligned[-1].scheduled_end - aligned[-1].original_end
        if aligned else 0.0
    )

    return {
        "mean_abs_duration_error_s": round(_stats.mean(errors), 3),
        "pct_severe_stretch":        round(100 * n_severe / max(len(metrics), 1), 1),
        "n_gap_shifts":              n_shifted,
        "n_translation_retries":     n_retry,
        "total_cumulative_drift_s":  round(drift, 3),
    }


def dubbing_scorecard(
    metrics:          list[SegmentMetrics],
    aligned_segments: list[AlignedSegment],
    align_report:     dict,
) -> dict:
    """Multi-dimensional quality scorecard for a dubbed clip.

    Returns five float scores in [0, 1]:

    timing_accuracy
        Penalises mean absolute duration error and the proportion of segments
        requiring severe time-stretch (> 1.4×).
    naturalness
        Low variance in per-segment stretch factors indicates a consistent
        speaking rate — high variance sounds unnatural.
    semantic_fidelity
        Character-count ratio between target and source text serves as a
        lightweight proxy for meaning preservation.  Spanish is typically
        10-20 % longer than English, so the score peaks at a ratio of 1.15.
    coverage
        Proportion of segments resolved without fallback (i.e. not
        ``REQUEST_SHORTER`` or ``FAIL``).
    overall
        Unweighted mean of the four dimension scores.

    Args:
        metrics: Per-segment timing metrics from ``compute_segment_metrics``.
        aligned_segments: Scheduled segments from ``global_align`` or
            ``global_align_dp``.
        align_report: Dict returned by ``clip_evaluation_report``.

    Returns:
        Dict with keys: timing_accuracy, naturalness, semantic_fidelity,
        coverage, overall.
    """
    if not metrics or not aligned_segments:
        return {
            "timing_accuracy":    0.0,
            "naturalness":        0.0,
            "semantic_fidelity":  0.0,
            "coverage":           0.0,
            "overall":            0.0,
        }

    # 1. Timing accuracy ─────────────────────────────────────────────────
    mean_err   = align_report.get("mean_abs_duration_error_s", 0.0)
    pct_severe = align_report.get("pct_severe_stretch", 0.0)
    timing_accuracy = (
        max(0.0, 1.0 - mean_err / 3.0)
        * max(0.0, 1.0 - pct_severe / 100.0)
    )

    # 2. Naturalness: low stretch-factor variance → consistent speaking rate
    stretch_factors = [a.stretch_factor for a in aligned_segments if a.stretch_factor > 0]
    if len(stretch_factors) >= 2:
        mean_sf  = sum(stretch_factors) / len(stretch_factors)
        variance = sum((sf - mean_sf) ** 2 for sf in stretch_factors) / len(stretch_factors)
        naturalness = max(0.0, 1.0 - variance / 0.25)
    else:
        naturalness = 1.0

    # 3. Semantic fidelity: char-count ratio target/source ≈ 1.15 for ES/EN
    ratios = [m.tgt_char_count / max(m.src_char_count, 1) for m in metrics]
    mean_ratio = sum(ratios) / len(ratios)
    semantic_fidelity = max(0.0, 1.0 - abs(mean_ratio - 1.15) / 0.85)

    # 4. Coverage: segments handled without fallback
    bad_actions = {AlignAction.FAIL, AlignAction.REQUEST_SHORTER}
    coverage = (
        sum(1 for a in aligned_segments if a.action not in bad_actions)
        / len(aligned_segments)
    )

    overall = (timing_accuracy + naturalness + semantic_fidelity + coverage) / 4.0

    return {
        "timing_accuracy":   round(timing_accuracy, 3),
        "naturalness":       round(naturalness, 3),
        "semantic_fidelity": round(semantic_fidelity, 3),
        "coverage":          round(coverage, 3),
        "overall":           round(overall, 3),
    }
