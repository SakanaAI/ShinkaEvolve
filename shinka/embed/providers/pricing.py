# Available embedding models and pricing - loaded from pricing.csv as DataFrame
# OpenAI: https://platform.openai.com/docs/pricing
# Gemini: https://ai.google.dev/gemini-api/docs/pricing

import pandas as pd
from pathlib import Path
from typing import Optional

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

    # Convert price column to numeric (handling N/A as 0)
    df["input_price"] = pd.to_numeric(
        df["input_price"].replace("N/A", "0"), errors="coerce"
    )

    # Convert prices from per-1M-tokens to per-token
    df["input_price"] = df["input_price"] / M

    # Set index to model_name for fast lookups
    df = df.set_index("model_name")

    return df


# Load pricing dataframe
_PRICING_DF = _load_pricing_dataframe()


def get_model_price(model_name: str) -> float:
    """Get the input price per token for a model.

    Returns the input price (embeddings only have input costs).
    """
    if model_name not in _PRICING_DF.index:
        raise ValueError(f"Embedding model {model_name} not found in pricing data")
    return _PRICING_DF.loc[model_name, "input_price"]


def model_exists(model_name: str) -> bool:
    """Check if an embedding model exists in pricing data."""
    return model_name in _PRICING_DF.index


def get_provider(model_name: str) -> Optional[str]:
    """Get the provider for a given embedding model."""
    if model_name not in _PRICING_DF.index:
        return None
    return _PRICING_DF.loc[model_name, "provider"]


def get_all_models() -> list:
    """Get list of all available embedding model names."""
    return _PRICING_DF.index.tolist()


def get_models_by_provider(provider: str) -> list:
    """Get list of embedding models for a given provider."""
    return _PRICING_DF[_PRICING_DF["provider"] == provider].index.tolist()
