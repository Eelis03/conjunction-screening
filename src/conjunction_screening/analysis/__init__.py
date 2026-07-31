"""Analysis layer: ranking, the dilution study, method comparison, and figures.

Figure generation lives in :mod:`conjunction_screening.analysis.figures` and is
imported on demand, so importing this package does not pull in matplotlib.
"""

from __future__ import annotations

from conjunction_screening.analysis.comparison import (
    MethodComparison,
    compare_methods,
    format_comparison_table,
    relative_disagreement,
    worst_pairwise_disagreement,
)
from conjunction_screening.analysis.dilution import (
    DilutionCurve,
    dilution_curve,
    format_dilution_summary,
)
from conjunction_screening.analysis.ranking import (
    ACTION_THRESHOLD,
    MONITOR_THRESHOLD,
    ActionClass,
    ActionThresholds,
    RankedConjunction,
    format_ranking_table,
    rank_events,
    rank_report,
)

__all__ = [
    "ACTION_THRESHOLD",
    "MONITOR_THRESHOLD",
    "ActionClass",
    "ActionThresholds",
    "DilutionCurve",
    "MethodComparison",
    "RankedConjunction",
    "compare_methods",
    "dilution_curve",
    "format_comparison_table",
    "format_dilution_summary",
    "format_ranking_table",
    "rank_events",
    "rank_report",
    "relative_disagreement",
    "worst_pairwise_disagreement",
]
