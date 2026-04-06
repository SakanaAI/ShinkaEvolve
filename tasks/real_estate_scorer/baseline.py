"""
Baseline scorer: simple linear weighted sum.

score = 0.5*(price_per_m2 normalized) + 0.3*(location_score normalized)
        + 0.2*(amenity_count normalized)

Run: python -m tasks.real_estate_scorer.baseline
"""

import json
import os

from scipy.stats import spearmanr


def baseline_score(listing: dict) -> float:
    """Compute a simple weighted investment score."""
    price_per_m2 = listing["price_usd"] / listing["size_m2"]
    location = listing["location_score"]
    amenities = listing["amenity_count"]

    score = (
        0.5 * (price_per_m2 / 5000.0)
        + 0.3 * (location / 5.0)
        + 0.2 * (amenities / 10.0)
    )
    return score


def main():
    data_dir = os.path.dirname(os.path.abspath(__file__))
    test_path = os.path.join(data_dir, "listings_test.json")

    if not os.path.exists(test_path):
        print("Test data not found. Generating dataset...")
        from tasks.real_estate_scorer.data import save_dataset
        save_dataset(data_dir)

    with open(test_path) as f:
        test_listings = json.load(f)

    scores = [baseline_score(listing) for listing in test_listings]
    gt_ranks = [listing["ground_truth_rank"] for listing in test_listings]

    # Higher score should map to rank 1 (best), so negate ranks
    correlation, p_value = spearmanr(scores, [-r for r in gt_ranks])

    print("=" * 50)
    print("Baseline Real Estate Scorer Results")
    print("=" * 50)
    print(f"Test listings:        {len(test_listings)}")
    print(f"Spearman correlation: {correlation:.4f}")
    print(f"p-value:              {p_value:.6f}")
    print("=" * 50)

    print("\nPer-listing breakdown:")
    print(f"{'#':<3} {'Neighborhood':<14} {'Price/m2':>10} {'Score':>8} {'GT Rank':>8}")
    print("-" * 50)
    for i, (listing, score) in enumerate(zip(test_listings, scores)):
        print(
            f"{i+1:<3} {listing['neighborhood']:<14} "
            f"{listing['price_per_m2']:>10.2f} "
            f"{score:>8.4f} "
            f"{listing['ground_truth_rank']:>8}"
        )

    return correlation


if __name__ == "__main__":
    main()
