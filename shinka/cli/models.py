#!/usr/bin/env python3
"""CLI for listing environment-available LLM and embedding models."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from shinka.embed.providers.pricing import (
    get_all_providers as get_all_embedding_providers,
)
from shinka.embed.providers.pricing import (
    get_models_by_provider as get_embedding_models_by_provider,
)
from shinka.env import load_shinka_dotenv
from shinka.llm.providers.pricing import get_all_providers, get_models_by_provider

PROVIDER_ENV_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "azure": ("AZURE_OPENAI_API_KEY", "AZURE_API_ENDPOINT", "AZURE_API_VERSION"),
    "bedrock": ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION_NAME"),
    "deepseek": ("DEEPSEEK_API_KEY",),
    "google": ("GEMINI_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
}


def _build_parser() -> argparse.ArgumentParser:
    description = (
        "Inspect current environment variables and discovered .env files, then "
        "emit JSON for pricing.csv LLM and embedding models that are usable in the "
        "current environment."
    )
    epilog = (
        "Output shape:\n"
        "  {\n"
        '    "embedding": [...],\n'
        '    "llm": [...]\n'
        "  }\n\n"
        "Verbose output shape:\n"
        "  {\n"
        '    "available_providers": [\n'
        '      {"provider": "google", "env_vars": {"GEMINI_API_KEY": true}, '
        '"llm_models": [...], "embedding_models": [...]}\n'
        "    ],\n"
        '    "embedding": [...],\n'
        '    "llm": [...]\n'
        "  }\n\n"
        "Readiness checks are strict and provider-specific:\n"
        "  anthropic: ANTHROPIC_API_KEY\n"
        "  azure: AZURE_OPENAI_API_KEY + AZURE_API_ENDPOINT + AZURE_API_VERSION\n"
        "  bedrock: AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY + AWS_REGION_NAME\n"
        "  deepseek: DEEPSEEK_API_KEY\n"
        "  google: GEMINI_API_KEY\n"
        "  openai: OPENAI_API_KEY\n"
        "  openrouter: OPENROUTER_API_KEY\n\n"
        "Security:\n"
        "  only availability booleans are printed; API key values are never shown\n\n"
        "Default output:\n"
        "  JSON object with separate embedding and llm lists\n"
        "Verbose output:\n"
        "  full JSON payload with provider details and the same top-level lists"
    )
    parser = argparse.ArgumentParser(
        prog="shinka_models",
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the full JSON payload instead of only the available model list.",
    )
    return parser


def _env_var_status(env_var_names: tuple[str, ...]) -> dict[str, bool]:
    return {
        env_var_name: bool(os.getenv(env_var_name, "").strip())
        for env_var_name in sorted(env_var_names)
    }


def _build_provider_entry(provider: str) -> dict[str, Any] | None:
    env_var_names = PROVIDER_ENV_REQUIREMENTS.get(provider)
    if env_var_names is None:
        return None

    env_vars = _env_var_status(env_var_names)
    if not all(env_vars.values()):
        return None

    llm_models = sorted(get_models_by_provider(provider))
    embedding_models = sorted(get_embedding_models_by_provider(provider))
    if not llm_models and not embedding_models:
        return None

    return {
        "provider": provider,
        "env_vars": env_vars,
        "llm_models": llm_models,
        "embedding_models": embedding_models,
    }


def _build_payload() -> dict[str, Any]:
    all_providers = sorted(
        set(get_all_providers()) | set(get_all_embedding_providers())
    )
    available_providers = [
        provider_entry
        for provider in all_providers
        if (provider_entry := _build_provider_entry(provider)) is not None
    ]
    llm_models = sorted(
        model
        for provider_entry in available_providers
        for model in provider_entry["llm_models"]
    )
    embedding_models = sorted(
        model
        for provider_entry in available_providers
        for model in provider_entry["embedding_models"]
    )
    return {
        "available_providers": available_providers,
        "embedding": embedding_models,
        "llm": llm_models,
    }


def main(argv: list[str] | None = None) -> int:
    load_shinka_dotenv()
    parser = _build_parser()
    args = parser.parse_args(argv)
    payload = _build_payload()
    output = payload if args.verbose else {
        "embedding": payload["embedding"],
        "llm": payload["llm"],
    }
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
