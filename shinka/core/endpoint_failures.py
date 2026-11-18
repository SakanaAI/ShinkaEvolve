"""How a run reacts when an LLM endpoint refuses every request.

A provider answers a wrong key, a missing model or a closed account on the
first attempt. Recording each such proposal as a failed generation would spend
the whole generation budget in seconds, so the run tries another model of the
pool and otherwise stops, leaving the generation for a resumed run.
"""

import math
from typing import Collection, List, Optional, Sequence

# A pool of several models tolerates this many proposals in a row that found
# no working endpoint; a single-model pool stops on the first.
ENDPOINT_FAILURES_BEFORE_STOP = 3


def endpoint_failures_before_stop(pool_size: int) -> int:
    """Return how many consecutive endpoint-failed proposals stop the run."""
    return 1 if pool_size <= 1 else ENDPOINT_FAILURES_BEFORE_STOP


def probabilities_without(
    model_names: Sequence[str],
    failed_models: Collection[str],
    weights: Optional[Sequence[float]] = None,
) -> Optional[List[float]]:
    """Return sampling probabilities over the models that have not failed.

    The given weights are kept in proportion. Returns None when no model is left.
    """
    usable = [name not in failed_models for name in model_names]
    if not any(usable):
        return None
    if weights is None or len(weights) != len(model_names):
        weights = [1.0] * len(model_names)
    kept = [float(weight) if ok else 0.0 for weight, ok in zip(weights, usable)]
    total = sum(kept)
    if not (math.isfinite(total) and total > 0.0):
        # The weights left nothing for the remaining models (a one-hot choice).
        kept = [1.0 if ok else 0.0 for ok in usable]
        total = sum(kept)
    return [weight / total for weight in kept]
