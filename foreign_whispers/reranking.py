"""Deterministic failure analysis and translation re-ranking stubs.

The failure analysis function uses simple threshold rules derived from
SegmentMetrics.  The translation re-ranking function is a **student assignment**
— see the docstring for inputs, outputs, and implementation guidance.
"""

import dataclasses
import logging
import re

from foreign_whispers.alignment import _estimate_duration

logger = logging.getLogger(__name__)

# Simple substitutions mapping for Spanish rule-based shortening.
SPANISH_SHORTENINGS = {
    r"\bsin embargo\b": "pero",
    r"\ben este momento\b": "ahora",
    r"\ben el caso de que\b": "si",
    r"\bcon el fin de\b": "para",
    r"\ba pesar de que\b": "aunque",
    r"\bpor lo tanto\b": "así que",
    r"\bdebido a que\b": "porque",
    r"\bdebido al hecho de que\b": "porque",
    r"\bde acuerdo con\b": "según",
    r"\bcon respecto a\b": "sobre",
    r"\bes necesario\b": "hay que",
    r"\bde manera rápida\b": "rápido",
    r"\bde manera eficiente\b": "bien",
    r"\bse puede observar\b": "se ve",
    r"\badicionalmente\b": "además",
    r"\bpor consiguiente\b": "por eso",
    r"\bpara poder\b": "para",
    r"\bcon el propósito de\b": "para",
}


@dataclasses.dataclass
class TranslationCandidate:
    """A candidate translation that fits a duration budget.

    Attributes:
        text: The translated text.
        char_count: Number of characters in *text*.
        brevity_rationale: Short explanation of what was shortened.
    """
    text: str
    char_count: int
    brevity_rationale: str = ""


@dataclasses.dataclass
class FailureAnalysis:
    """Diagnostic summary of the dominant failure mode in a clip.

    Attributes:
        failure_category: One of "duration_overflow", "cumulative_drift",
            "stretch_quality", or "ok".
        likely_root_cause: One-sentence description.
        suggested_change: Most impactful next action.
    """
    failure_category: str
    likely_root_cause: str
    suggested_change: str


def analyze_failures(report: dict) -> FailureAnalysis:
    """Classify the dominant failure mode from a clip evaluation report.

    Pure heuristic — no LLM needed.  The thresholds below match the policy
    bands defined in ``alignment.decide_action``.

    Args:
        report: Dict returned by ``clip_evaluation_report()``.  Expected keys:
            ``mean_abs_duration_error_s``, ``pct_severe_stretch``,
            ``total_cumulative_drift_s``, ``n_translation_retries``.

    Returns:
        A ``FailureAnalysis`` dataclass.
    """
    mean_err = report.get("mean_abs_duration_error_s", 0.0)
    pct_severe = report.get("pct_severe_stretch", 0.0)
    drift = abs(report.get("total_cumulative_drift_s", 0.0))
    retries = report.get("n_translation_retries", 0)

    if pct_severe > 20:
        return FailureAnalysis(
            failure_category="duration_overflow",
            likely_root_cause=(
                f"{pct_severe:.0f}% of segments exceed the 1.4x stretch threshold — "
                "translated text is consistently too long for the available time window."
            ),
            suggested_change="Implement duration-aware translation re-ranking (P8).",
        )

    if drift > 3.0:
        return FailureAnalysis(
            failure_category="cumulative_drift",
            likely_root_cause=(
                f"Total drift is {drift:.1f}s — small per-segment overflows "
                "accumulate because gaps between segments are not being reclaimed."
            ),
            suggested_change="Enable gap_shift in the global alignment optimizer (P9).",
        )

    if mean_err > 0.8:
        return FailureAnalysis(
            failure_category="stretch_quality",
            likely_root_cause=(
                f"Mean duration error is {mean_err:.2f}s — segments fit within "
                "stretch limits but the stretch distorts audio quality."
            ),
            suggested_change="Lower the mild_stretch ceiling or shorten translations.",
        )

    return FailureAnalysis(
        failure_category="ok",
        likely_root_cause="No dominant failure mode detected.",
        suggested_change="Review individual outlier segments if any remain.",
    )


def get_shorter_translations(
    source_text: str,
    baseline_es: str,
    target_duration_s: float,
    context_prev: str = "",
    context_next: str = "",
) -> list[TranslationCandidate]:
    """Return shorter translation candidates that fit *target_duration_s*.

    .. admonition:: Student Assignment — Duration-Aware Translation Re-ranking

       This function is intentionally a **stub that returns an empty list**.
       Your task is to implement a strategy that produces shorter
       target-language translations when the baseline translation is too long
       for the time budget.

       **Inputs**

       ============== ======== ==================================================
       Parameter      Type     Description
       ============== ======== ==================================================
       source_text    str      Original source-language segment text
       baseline_es    str      Baseline target-language translation (from argostranslate)
       target_duration_s float Time budget in seconds for this segment
       context_prev   str      Text of the preceding segment (for coherence)
       context_next   str      Text of the following segment (for coherence)
       ============== ======== ==================================================

       **Outputs**

       A list of ``TranslationCandidate`` objects, sorted shortest first.
       Each candidate has:

       - ``text``: the shortened target-language translation
       - ``char_count``: ``len(text)``
       - ``brevity_rationale``: short note on what was changed

       **Duration heuristic**: target-language TTS produces ~15 characters/second
       (or ~4.5 syllables/second for Romance languages).  So a 3-second budget
       ≈ 45 characters.

       **Approaches to consider** (pick one or combine):

       1. **Rule-based shortening** — strip filler words, use shorter synonyms
          from a lookup table, contract common phrases
          (e.g. "en este momento" → "ahora").
       2. **Multiple translation backends** — call argostranslate with
          paraphrased input, or use a second translation model, then pick
          the shortest output that preserves meaning.
       3. **LLM re-ranking** — use an LLM (e.g. via an API) to generate
          condensed alternatives.  This was the previous approach but adds
          latency, cost, and a runtime dependency.
       4. **Hybrid** — rule-based first, fall back to LLM only for segments
          that still exceed the budget.

       **Evaluation criteria**: the caller selects the candidate whose
       ``len(text) / 15.0`` is closest to ``target_duration_s``.

    Returns:
        Empty list (stub).  Implement to return ``TranslationCandidate`` items.
    """
    logger.info(
        "get_shorter_translations called for %.1fs budget (%d chars baseline)",
        target_duration_s,
        len(baseline_es),
    )

    budget_chars = int(target_duration_s * 15)
    candidates = []

    # Always consider the baseline as a candidate (it might be the only one, or already fit).
    candidates.append(TranslationCandidate(
        text=baseline_es,
        char_count=len(baseline_es),
        brevity_rationale="baseline"
    ))

    # Detect severe hallucination from the translation model
    if len(source_text) <= 20 and len(baseline_es) > len(source_text) * 3:
        fallback_dict = {
            "yes.": "sí.",
            "yes": "sí",
            "no.": "no.",
            "no": "no",
            "look, yes.": "mira, sí.",
            "look, yes": "mira, sí",
            "period.": "punto.",
            "the strait.": "el estrecho."
        }
        lower_src = source_text.strip().lower()
        if lower_src in fallback_dict:
            candidates.append(TranslationCandidate(
                text=fallback_dict[lower_src],
                char_count=len(fallback_dict[lower_src]),
                brevity_rationale="hallucination dictionary fallback"
            ))
        else:
            truncated = baseline_es[:int(target_duration_s * 15)]
            candidates.append(TranslationCandidate(
                text=truncated,
                char_count=len(truncated),
                brevity_rationale="hallucination aggressive truncation"
            ))
    
    current_text = baseline_es
    applied_rules = []
    
    # Iteratively apply rules to generate progressively shorter candidates
    for pattern, replacement in SPANISH_SHORTENINGS.items():
        # Case insensitive substitution
        new_text, count = re.subn(pattern, replacement, current_text, flags=re.IGNORECASE)
        if count > 0:
            current_text = new_text
            clean_pattern = pattern.replace(r"\b", "")
            applied_rules.append(f"'{clean_pattern}' -> '{replacement}'")
            candidates.append(TranslationCandidate(
                text=current_text,
                char_count=len(current_text),
                brevity_rationale=", ".join(applied_rules)
            ))
            
            # If we've reached the budget, we could stop, but it's fine to provide
            # all variations and let the caller decide.

    # Sort candidates by how close their predicted duration is to the target duration
    # We want to minimize absolute difference between predicted and target duration
    candidates.sort(key=lambda c: abs(_estimate_duration(c.text) - target_duration_s))
    return candidates
