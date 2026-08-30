"""Step 5: why max-score ≠ good study probability. Breakpoints in aggregation.py."""
from __future__ import annotations

from fracture.inference.aggregation import StudyAggregator, aggregation_features, longest_run

# Toy positive-like: consecutive mid confidences
pos_scores = [0.02, 0.05, 0.22, 0.31, 0.28, 0.04, 0.01]
# Toy negative-like: one spike (suture / vessel FP)
neg_scores = [0.01, 0.40, 0.02, 0.01, 0.01]

for name, scores in [("pos_like", pos_scores), ("neg_spike", neg_scores)]:
    feats = aggregation_features(scores, thresholds=(0.05, 0.1, 0.3))
    max_agg = StudyAggregator("max").predict(scores)
    print(
        name,
        {
            "max_agg": max_agg,
            "longest_run_ge_0_1": longest_run(scores, 0.1),
            "count_ge_0_1": feats["count_ge_0_1"],
            "top3_mean": feats["top3_mean"],
            "detected_slices": feats["detected_slices"],
        },
    )

print("Insight: max treats a single FP spike like a real multi-slice fracture.")
print("OK: aggregation")
