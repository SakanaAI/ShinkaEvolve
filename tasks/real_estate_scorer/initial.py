# EVOLVE-BLOCK-START
"""Scoring function for Buenos Aires apartment investment quality."""


def score_listing(listing: dict) -> float:
    """Score an apartment listing for investment quality.

    Takes a listing dict with keys: price_usd, size_m2, neighborhood,
    location_score (1-5), amenity_count (0-10), floor, age_years.

    Returns a float investment score (higher = better investment signal).
    """
    price_per_m2 = listing["price_usd"] / listing["size_m2"]
    location = listing["location_score"]
    amenities = listing["amenity_count"]

    # Simple weighted sum with basic normalization
    score = (
        0.5 * (price_per_m2 / 5000.0)
        + 0.3 * (location / 5.0)
        + 0.2 * (amenities / 10.0)
    )
    return score


# EVOLVE-BLOCK-END

# --- Fixed evaluation harness below (not evolved) ---

import json  # noqa: E402
import os  # noqa: E402


def run_scoring() -> dict:
    """Run the scoring function on the test set and return results.

    Returns dict with keys: scores, listings, ground_truth_ranks.
    """
    data_dir = os.path.dirname(os.path.abspath(__file__))
    test_path = os.path.join(data_dir, "listings_test.json")

    with open(test_path) as f:
        test_listings = json.load(f)

    scores = [score_listing(listing) for listing in test_listings]
    ground_truth_ranks = [entry["ground_truth_rank"] for entry in test_listings]

    return {
        "scores": scores,
        "listings": test_listings,
        "ground_truth_ranks": ground_truth_ranks,
    }
