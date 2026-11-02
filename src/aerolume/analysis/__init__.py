"""Post-processing of solver output: statistics, transients, Strouhal, convergence,
polar and finite-wing figures of merit, baseline comparisons, grid-convergence
uncertainty (:mod:`.gci`) and the heat-transfer / fan outputs (:mod:`.physics`).
numpy + stdlib only."""
from .compare import (
    Metric,
    compare_rows,
    comparison_groups,
    comparison_warnings,
    conditions_differ,
    result_dimension,
    reynolds_mismatch,
)
from .gci import (
    GCIResult,
    GridLevel,
    GridStudy,
    find_grid_studies,
    gci_study,
    gci_two_grid,
    grid_study,
    observed_order,
    richardson,
)
from .physics import (
    fan_entries,
    has_fans,
    has_thermal,
    heat_balance,
    heat_rate_stat,
    momentum_theory,
    nusselt_stat,
)
from .polar import METRIC_KEYS, polar_metrics
from .stats import (
    HISTORY_KEYS_3D,
    auto_batch_size,
    autocorrelation,
    batch_means,
    batch_means_stderr,
    coefficient_stats,
    convergence_state,
    detect_steady,
    dominant_frequency,
    effective_sample_size,
    integrated_autocorr_time,
    stats,
    strouhal,
    summarize_history,
    summarize_history3d,
    transient_end,
)
from .wing import WING_METRIC_KEYS, wing_metrics

__all__ = [
    "HISTORY_KEYS_3D", "METRIC_KEYS", "WING_METRIC_KEYS", "GCIResult", "GridLevel", "GridStudy", "Metric",
    "auto_batch_size", "autocorrelation", "batch_means", "batch_means_stderr", "coefficient_stats", "compare_rows",
    "comparison_groups", "comparison_warnings", "conditions_differ", "convergence_state", "detect_steady",
    "dominant_frequency", "effective_sample_size", "fan_entries", "find_grid_studies", "gci_study", "gci_two_grid",
    "grid_study", "has_fans", "has_thermal", "heat_balance", "heat_rate_stat", "integrated_autocorr_time",
    "momentum_theory", "nusselt_stat", "observed_order", "polar_metrics", "result_dimension", "reynolds_mismatch",
    "richardson", "stats", "strouhal", "summarize_history", "summarize_history3d", "transient_end", "wing_metrics",
]
