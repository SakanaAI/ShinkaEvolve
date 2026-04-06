"""
Generate synthetic Buenos Aires apartment listings for the real estate scorer task.

Uses seeded numpy RNG (seed=42) for reproducibility.
Saves listings_train.json (40 listings) and listings_test.json (10 listings).
"""

import json
import os
import numpy as np

# Neighborhood desirability scores (1-5)
NEIGHBORHOODS = {
    "Palermo": 5,
    "Recoleta": 5,
    "Belgrano": 4,
    "Almagro": 3,
    "Villa Crespo": 3,
    "Boedo": 2,
    "Mataderos": 1,
}

NEIGHBORHOOD_NAMES = list(NEIGHBORHOODS.keys())
NEIGHBORHOOD_SCORES = [NEIGHBORHOODS[n] for n in NEIGHBORHOOD_NAMES]


def generate_listings(n: int = 50, seed: int = 42) -> list[dict]:
    """Generate n synthetic apartment listings with ground-truth ranking."""
    rng = np.random.default_rng(seed)

    listings = []
    for _ in range(n):
        idx = rng.integers(0, len(NEIGHBORHOOD_NAMES))
        neighborhood = NEIGHBORHOOD_NAMES[idx]
        location_score = NEIGHBORHOOD_SCORES[idx]

        size_m2 = round(float(rng.uniform(25, 200)), 1)
        # Price correlates with location and size, plus noise
        base_price_per_m2 = 1500 + location_score * 400 + rng.normal(0, 300)
        price_usd = round(float(base_price_per_m2 * size_m2), 2)
        price_usd = max(price_usd, 10000.0)  # floor

        amenity_count = int(rng.integers(0, 11))
        floor = int(rng.integers(0, 25))
        age_years = int(rng.integers(0, 80))

        price_per_m2 = price_usd / size_m2

        listings.append({
            "price_usd": price_usd,
            "size_m2": size_m2,
            "neighborhood": neighborhood,
            "location_score": location_score,
            "amenity_count": amenity_count,
            "floor": floor,
            "age_years": age_years,
            "price_per_m2": round(price_per_m2, 2),
        })

    return listings


def compute_ground_truth_rankings(listings: list[dict]) -> list[int]:
    """Rank listings by price_per_m2 (higher = better investment signal).

    Returns a list of ranks (1 = highest price_per_m2).
    """
    price_per_m2 = [entry["price_per_m2"] for entry in listings]
    # argsort descending, then convert to 1-based ranks
    order = np.argsort(price_per_m2)[::-1]
    ranks = np.empty(len(price_per_m2), dtype=int)
    for rank, idx in enumerate(order):
        ranks[idx] = rank + 1
    return ranks.tolist()


def save_dataset(output_dir: str | None = None):
    """Generate and save train/test splits."""
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))

    listings = generate_listings(50, seed=42)
    rankings = compute_ground_truth_rankings(listings)

    # Attach rankings
    for listing, rank in zip(listings, rankings):
        listing["ground_truth_rank"] = rank

    train = listings[:40]
    test = listings[40:]

    train_path = os.path.join(output_dir, "listings_train.json")
    test_path = os.path.join(output_dir, "listings_test.json")

    with open(train_path, "w") as f:
        json.dump(train, f, indent=2)
    with open(test_path, "w") as f:
        json.dump(test, f, indent=2)

    print(f"Saved {len(train)} train listings to {train_path}")
    print(f"Saved {len(test)} test listings to {test_path}")
    return train, test


if __name__ == "__main__":
    save_dataset()
