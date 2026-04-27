#!/usr/bin/env python3
"""CLI for listing environment-available LLM and embedding models."""

from __future__ import annotations

import argparse
import json
from typing import Any

from shinka.env import load_shinka_dotenv
from shinka.model_availability import (
    build_model_availability_payload,
    build_provider_availability_entry,
    env_var_status,
    provider_env_requirements as get_provider_env_requirements,
)


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
        "  google: GEMINI_API_KEY or GOOGLE_GENAI_USE_VERTEXAI + GOOGLE_CLOUD_PROJECT + GOOGLE_CLOUD_LOCATION\n"
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
    return env_var_status(env_var_names)


def provider_env_requirements(provider: str) -> tuple[str, ...] | None:
    return get_provider_env_requirements(provider)


def _build_provider_entry(provider: str) -> dict[str, Any] | None:
    return build_provider_availability_entry(provider)


def _build_payload() -> dict[str, Any]:
    return build_model_availability_payload()


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
