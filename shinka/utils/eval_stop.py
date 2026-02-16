"""
Early stopping methods for stochastic program evaluation.

These methods allow stopping evaluation early when we're confident that
the program's mean score won't beat (or will beat) a target threshold.
"""

import numpy as np
from scipy import stats as sp_stats
from dataclasses import dataclass
from typing import List, Optional, Literal
from abc import ABC, abstractmethod


@dataclass
class EarlyStopDecision:
    """Result of an early stopping check."""

    should_stop: bool
    prediction: Literal["beats", "fails", "uncertain"]
    confidence: float  # probability or confidence level
    trials_used: int
    reason: Optional[str] = None


class EarlyStopMethod(ABC):
    """Base class for early stopping methods."""

    @abstractmethod
    def check(self, scores: List[float], threshold: float) -> EarlyStopDecision:
        """
        Check if we should stop evaluation early.

        Args:
            scores: List of scores observed so far.
            threshold: Target threshold to beat.

        Returns:
            EarlyStopDecision with stop decision and metadata.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this method."""
        pass

    def reset(self) -> None:
        """Reset any internal state (called at start of new evaluation)."""
        pass


class NoEarlyStop(EarlyStopMethod):
    """Baseline: never stop early, always run all trials."""

    @property
    def name(self) -> str:
        return "no_early_stop"

    def check(self, scores: List[float], threshold: float) -> EarlyStopDecision:
        return EarlyStopDecision(
            should_stop=False,
            prediction="uncertain",
            confidence=0.5,
            trials_used=len(scores),
            reason="early stopping disabled",
        )


class BayesianEarlyStop(EarlyStopMethod):
    """
    Bayesian early stopping using Normal model.

    Computes P(μ > threshold | data) and stops if this probability
    is below `prob_cutoff` (fails) or above `1 - prob_cutoff` (beats).
    """

    def __init__(self, prob_cutoff: float = 0.05, min_trials: int = 2):
        """
        Args:
            prob_cutoff: Stop if P(beats) < cutoff or P(beats) > 1-cutoff.
            min_trials: Minimum trials before considering early stop.
        """
        self.prob_cutoff = prob_cutoff
        self.min_trials = min_trials

    @property
    def name(self) -> str:
        return f"bayesian_p{self.prob_cutoff}"

    def check(self, scores: List[float], threshold: float) -> EarlyStopDecision:
        n = len(scores)

        if n < self.min_trials:
            return EarlyStopDecision(
                should_stop=False,
                prediction="uncertain",
                confidence=0.5,
                trials_used=n,
                reason=f"need at least {self.min_trials} trials",
            )

        mean = np.mean(scores)
        std = np.std(scores, ddof=1) if n > 1 else 1.0

        # Posterior standard error (using empirical variance)
        posterior_se = std / np.sqrt(n)

        # Avoid division by zero
        if posterior_se < 1e-10:
            # All scores are identical
            if mean > threshold:
                return EarlyStopDecision(
                    should_stop=True,
                    prediction="beats",
                    confidence=1.0,
                    trials_used=n,
                    reason="all scores identical and above threshold",
                )
            else:
                return EarlyStopDecision(
                    should_stop=True,
                    prediction="fails",
                    confidence=1.0,
                    trials_used=n,
                    reason="all scores identical and below threshold",
                )

        # P(μ > threshold | data) using normal approximation
        prob_beats = 1 - sp_stats.norm.cdf(threshold, loc=mean, scale=posterior_se)

        if prob_beats < self.prob_cutoff:
            return EarlyStopDecision(
                should_stop=True,
                prediction="fails",
                confidence=1 - prob_beats,
                trials_used=n,
                reason=f"P(beats)={prob_beats:.4f} < {self.prob_cutoff}",
            )
        elif prob_beats > (1 - self.prob_cutoff):
            return EarlyStopDecision(
                should_stop=True,
                prediction="beats",
                confidence=prob_beats,
                trials_used=n,
                reason=f"P(beats)={prob_beats:.4f} > {1 - self.prob_cutoff}",
            )

        return EarlyStopDecision(
            should_stop=False,
            prediction="uncertain",
            confidence=prob_beats,
            trials_used=n,
            reason=f"P(beats)={prob_beats:.4f}, continuing",
        )


class ConfidenceIntervalEarlyStop(EarlyStopMethod):
    """
    Early stopping based on confidence intervals.

    Stops when the CI for the mean is entirely above or below threshold.
    """

    def __init__(self, confidence: float = 0.90, min_trials: int = 3):
        """
        Args:
            confidence: Confidence level for the interval (e.g., 0.90 for 90% CI).
            min_trials: Minimum trials before considering early stop.
        """
        self.confidence = confidence
        self.min_trials = min_trials

    @property
    def name(self) -> str:
        return f"ci_{int(self.confidence * 100)}"

    def check(self, scores: List[float], threshold: float) -> EarlyStopDecision:
        n = len(scores)

        if n < self.min_trials:
            return EarlyStopDecision(
                should_stop=False,
                prediction="uncertain",
                confidence=0.5,
                trials_used=n,
                reason=f"need at least {self.min_trials} trials",
            )

        mean = np.mean(scores)
        se = sp_stats.sem(scores)

        # Handle zero standard error
        if se < 1e-10:
            if mean > threshold:
                return EarlyStopDecision(
                    should_stop=True,
                    prediction="beats",
                    confidence=1.0,
                    trials_used=n,
                    reason="all scores identical and above threshold",
                )
            else:
                return EarlyStopDecision(
                    should_stop=True,
                    prediction="fails",
                    confidence=1.0,
                    trials_used=n,
                    reason="all scores identical and below threshold",
                )

        # Two-sided confidence interval
        alpha = 1 - self.confidence
        t_crit = sp_stats.t.ppf(1 - alpha / 2, df=n - 1)
        lower = mean - t_crit * se
        upper = mean + t_crit * se

        if upper < threshold:
            return EarlyStopDecision(
                should_stop=True,
                prediction="fails",
                confidence=self.confidence,
                trials_used=n,
                reason=f"CI upper bound {upper:.4f} < threshold {threshold}",
            )
        elif lower > threshold:
            return EarlyStopDecision(
                should_stop=True,
                prediction="beats",
                confidence=self.confidence,
                trials_used=n,
                reason=f"CI lower bound {lower:.4f} > threshold {threshold}",
            )

        return EarlyStopDecision(
            should_stop=False,
            prediction="uncertain",
            confidence=self.confidence,
            trials_used=n,
            reason=f"CI [{lower:.4f}, {upper:.4f}] contains threshold {threshold}",
        )


class HybridEarlyStop(EarlyStopMethod):
    """
    Hybrid early stopping combining Bayesian probability and CI bounds.

    Stops if either:
    - The CI upper bound is below threshold (certain fail)
    - The Bayesian P(beats) is below prob_cutoff (likely fail)
    - The Bayesian P(beats) is above 1-prob_cutoff (likely beats)
    """

    def __init__(
        self,
        prob_cutoff: float = 0.05,
        ci_confidence: float = 0.90,
        min_trials: int = 3,
    ):
        """
        Args:
            prob_cutoff: Bayesian probability cutoff for stopping.
            ci_confidence: Confidence level for CI check.
            min_trials: Minimum trials before considering early stop.
        """
        self.prob_cutoff = prob_cutoff
        self.ci_confidence = ci_confidence
        self.min_trials = min_trials

    @property
    def name(self) -> str:
        return "hybrid"

    def check(self, scores: List[float], threshold: float) -> EarlyStopDecision:
        n = len(scores)

        if n < self.min_trials:
            return EarlyStopDecision(
                should_stop=False,
                prediction="uncertain",
                confidence=0.5,
                trials_used=n,
                reason=f"need at least {self.min_trials} trials",
            )

        mean = np.mean(scores)
        std = np.std(scores, ddof=1)
        se = std / np.sqrt(n)

        # Handle zero standard error
        if se < 1e-10:
            if mean > threshold:
                return EarlyStopDecision(
                    should_stop=True,
                    prediction="beats",
                    confidence=1.0,
                    trials_used=n,
                    reason="all scores identical and above threshold",
                )
            else:
                return EarlyStopDecision(
                    should_stop=True,
                    prediction="fails",
                    confidence=1.0,
                    trials_used=n,
                    reason="all scores identical and below threshold",
                )

        # Bayesian probability
        prob_beats = 1 - sp_stats.norm.cdf(threshold, loc=mean, scale=se)

        # CI upper bound
        alpha = 1 - self.ci_confidence
        t_crit = sp_stats.t.ppf(1 - alpha / 2, df=n - 1)
        upper_bound = mean + t_crit * se

        # Check CI first (stronger signal)
        if upper_bound < threshold:
            return EarlyStopDecision(
                should_stop=True,
                prediction="fails",
                confidence=self.ci_confidence,
                trials_used=n,
                reason=f"CI upper {upper_bound:.4f} < threshold (P(beats)={prob_beats:.4f})",
            )

        # Then check Bayesian probability
        if prob_beats < self.prob_cutoff:
            return EarlyStopDecision(
                should_stop=True,
                prediction="fails",
                confidence=1 - prob_beats,
                trials_used=n,
                reason=f"P(beats)={prob_beats:.4f} < {self.prob_cutoff}",
            )
        elif prob_beats > (1 - self.prob_cutoff):
            return EarlyStopDecision(
                should_stop=True,
                prediction="beats",
                confidence=prob_beats,
                trials_used=n,
                reason=f"P(beats)={prob_beats:.4f} > {1 - self.prob_cutoff}",
            )

        return EarlyStopDecision(
            should_stop=False,
            prediction="uncertain",
            confidence=prob_beats,
            trials_used=n,
            reason=f"P(beats)={prob_beats:.4f}, upper={upper_bound:.4f}, continuing",
        )


def create_early_stop_method(
    method: str = "none",
    prob_cutoff: float = 0.05,
    ci_confidence: float = 0.90,
    min_trials: int = 3,
) -> EarlyStopMethod:
    """
    Factory function to create an early stopping method.

    Args:
        method: One of "none", "bayesian", "ci", "hybrid".
        prob_cutoff: Probability cutoff for Bayesian/Hybrid methods.
        ci_confidence: Confidence level for CI/Hybrid methods.
        min_trials: Minimum trials before considering early stop.

    Returns:
        An EarlyStopMethod instance.
    """
    method = method.lower()

    if method == "none" or method == "no_early_stop":
        return NoEarlyStop()
    elif method == "bayesian":
        return BayesianEarlyStop(prob_cutoff=prob_cutoff, min_trials=min_trials)
    elif method == "ci" or method == "confidence_interval":
        return ConfidenceIntervalEarlyStop(
            confidence=ci_confidence, min_trials=min_trials
        )
    elif method == "hybrid":
        return HybridEarlyStop(
            prob_cutoff=prob_cutoff,
            ci_confidence=ci_confidence,
            min_trials=min_trials,
        )
    else:
        raise ValueError(
            f"Unknown early stop method: {method}. "
            f"Choose from: none, bayesian, ci, hybrid"
        )
