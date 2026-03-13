# Available models and pricing - loaded from pricing.csv as DataFrame
# Anthropic: https://www.anthropic.com/pricing#anthropic-api
# OpenAI: https://platform.openai.com/docs/pricing
# DeepSeek: https://api-docs.deepseek.com/quick_start/pricing/
# Gemini: https://ai.google.dev/gemini-api/docs/pricing

import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

# Load pricing data from CSV
_pricing_csv_path = Path(__file__).parent / "pricing.csv"
# Utility constant
M = 1_000_000


def _load_pricing_dataframe() -> pd.DataFrame:
    """Load pricing data from CSV file as a pandas DataFrame."""
    df = pd.read_csv(_pricing_csv_path)

    # Strip whitespace from string columns only
    for col in df.columns:
        if df[col].dtype == "object":  # Only strip string columns
            df[col] = df[col].str.strip()

    # Strip column names
    df.columns = df.columns.str.strip()

    # Convert price columns to numeric (handling N/A as 0)
    df["input_price"] = pd.to_numeric(
        df["input_price"].replace("N/A", "0"), errors="coerce"
    )
    df["output_price"] = pd.to_numeric(
        df["output_price"].replace("N/A", "0"), errors="coerce"
    )

    # Convert tier 2 price columns to numeric (empty/NaN stays as NaN)
    df["input_price_tier2"] = pd.to_numeric(
        df["input_price_tier2"].replace("N/A", ""), errors="coerce"
    )
    df["output_price_tier2"] = pd.to_numeric(
        df["output_price_tier2"].replace("N/A", ""), errors="coerce"
    )

    # Convert tier threshold to numeric (empty/NaN stays as NaN)
    df["tier_threshold"] = pd.to_numeric(df["tier_threshold"], errors="coerce")

    # Convert prices from per-1M-tokens to per-token
    df["input_price"] = df["input_price"] / M
    df["output_price"] = df["output_price"] / M
    df["input_price_tier2"] = df["input_price_tier2"] / M
    df["output_price_tier2"] = df["output_price_tier2"] / M

    # Convert is_reasoning to boolean
    df["is_reasoning"] = df["is_reasoning"] == "True"

    # Convert think_temp_fixed to boolean (handle both string "1" and int 1)
    df["think_temp_fixed"] = df["think_temp_fixed"].astype(str) == "1"

    # Convert requires_reasoning to boolean (handle both string "1" and int 1)
    df["requires_reasoning"] = df["requires_reasoning"].astype(str) == "1"

    # Set index to model_name for fast lookups
    df = df.set_index("model_name")

    return df


# Load pricing dataframe
_PRICING_DF = _load_pricing_dataframe()


def get_model_prices(model_name: str, input_tokens: Optional[int] = None) -> dict:
    """Get both input and output prices for a model.

    For models with tiered pricing, the tier is determined by input token count.
    If input_tokens is provided and exceeds the tier threshold, tier 2 prices are returned.

    Args:
        model_name: The name of the model.
        input_tokens: Optional input token count to determine pricing tier.

    Returns:
        Dict with 'input_price' and 'output_price' keys (per-token prices).
    """
    if model_name not in _PRICING_DF.index:
        raise ValueError(f"Model {model_name} not found in pricing data")
    row = _PRICING_DF.loc[model_name]

    # Check if tiered pricing applies
    tier_threshold = row.get("tier_threshold")
    has_tiered_pricing = pd.notna(tier_threshold) and tier_threshold > 0

    if (
        has_tiered_pricing
        and input_tokens is not None
        and input_tokens > tier_threshold
    ):
        # Use tier 2 pricing
        input_price = row["input_price_tier2"]
        output_price = row["output_price_tier2"]
        # Fall back to tier 1 if tier 2 prices are not defined
        if pd.isna(input_price):
            input_price = row["input_price"]
        if pd.isna(output_price):
            output_price = row["output_price"]
    else:
        # Use tier 1 pricing
        input_price = row["input_price"]
        output_price = row["output_price"]

    return {
        "input_price": input_price,
        "output_price": output_price,
    }


def calculate_cost(
    model_name: str, input_tokens: int, output_tokens: int
) -> Tuple[float, float]:
    """Calculate input and output costs for a model with tiered pricing support.

    For models with tiered pricing (e.g., Gemini), the tier is determined by
    the input token count. If input_tokens exceeds the tier threshold, tier 2
    prices are used for BOTH input and output.

    Args:
        model_name: The name of the model.
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens (including thinking tokens).

    Returns:
        Tuple of (input_cost, output_cost).
    """
    prices = get_model_prices(model_name, input_tokens=input_tokens)
    input_cost = prices["input_price"] * input_tokens
    output_cost = prices["output_price"] * output_tokens
    return input_cost, output_cost


def model_exists(model_name: str) -> bool:
    """Check if a model exists in pricing data."""
    return model_name in _PRICING_DF.index


def is_reasoning_model(model_name: str) -> bool:
    """Check if a model is a reasoning model."""
    if model_name not in _PRICING_DF.index:
        return False
    return _PRICING_DF.loc[model_name, "is_reasoning"]


def get_provider(model_name: str) -> Optional[str]:
    """Get the provider for a given model."""
    if model_name not in _PRICING_DF.index:
        return None
    return _PRICING_DF.loc[model_name, "provider"]


def has_fixed_temperature(model_name: str) -> bool:
    """Check if a model requires temperature fixed to 1.0."""
    if model_name not in _PRICING_DF.index:
        return False
    return _PRICING_DF.loc[model_name, "think_temp_fixed"]


def requires_reasoning(model_name: str) -> bool:
    """Check if a model requires reasoning effort to be set."""
    if model_name not in _PRICING_DF.index:
        return False
    return _PRICING_DF.loc[model_name, "requires_reasoning"]
