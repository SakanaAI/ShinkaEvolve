from __future__ import annotations

import json
from pathlib import Path

import pytest

from shinka.tools.pricing.generate_csvs import (
    EMBEDDING_HEADERS,
    LLM_HEADERS,
    MODELS_DEV_USER_AGENT,
    TargetModel,
    generate_embedding_rows,
    generate_llm_rows,
    load_models_dev_payload,
    main,
    render_csv,
)


def _model(
    *,
    input_price: float,
    output_price: float = 0.0,
    reasoning: bool = False,
    temperature: bool = True,
    tiers: list[dict] | None = None,
) -> dict:
    cost: dict = {"input": input_price, "output": output_price}
    if tiers is not None:
        cost["tiers"] = tiers
    return {
        "cost": cost,
        "reasoning": reasoning,
        "temperature": temperature,
    }


def _payload() -> dict:
    return {
        "amazon-bedrock": {
            "models": {
                "us.anthropic.claude-sonnet-4-6": _model(
                    input_price=3.0,
                    output_price=15.0,
                    reasoning=True,
                    temperature=True,
                    tiers=[
                        {
                            "input": 6.0,
                            "output": 22.5,
                            "tier": {"type": "context", "size": 200000},
                        }
                    ],
                )
            }
        },
        "anthropic": {
            "models": {
                "claude-sonnet-4-20250514": _model(
                    input_price=3.0,
                    output_price=15.0,
                    reasoning=True,
                    temperature=True,
                ),
                "claude-sonnet-4-5-20250929": _model(
                    input_price=3.0,
                    output_price=15.0,
                    reasoning=True,
                    temperature=True,
                ),
            }
        },
        "azure": {
            "models": {
                "text-embedding-3-small": _model(input_price=0.02),
            }
        },
        "google": {
            "models": {
                "gemini-2.5-pro": _model(
                    input_price=1.25,
                    output_price=10.0,
                    reasoning=True,
                    tiers=[
                        {
                            "input": 2.5,
                            "output": 15.0,
                            "tier": {"type": "context", "size": 200000},
                        }
                    ],
                ),
            }
        },
        "openai": {
            "models": {
                "o3-mini": _model(
                    input_price=1.1,
                    output_price=4.4,
                    reasoning=True,
                    temperature=False,
                ),
            }
        },
    }


def test_generate_llm_rows_preserves_schema_and_maps_aliases():
    rows = generate_llm_rows(
        _payload(),
        [
            TargetModel("llm", "o3-mini-2025-01-31", "openai"),
            TargetModel("llm", "claude-4-sonnet-20250514", "anthropic"),
            TargetModel("llm", "claude-sonnet-4-5-20250929", "anthropic"),
            TargetModel("llm", "us.anthropic.claude-sonnet-4-6-v1:0", "bedrock"),
            TargetModel("llm", "gemini-2.5-pro", "google"),
        ],
    )

    assert list(rows[0]) == LLM_HEADERS
    assert rows[0] == {
        "model_name": "o3-mini-2025-01-31",
        "provider": "openai",
        "input_price": "1.1",
        "output_price": "4.4",
        "input_price_tier2": "",
        "output_price_tier2": "",
        "tier_threshold": "",
        "is_reasoning": "True",
        "think_temp_fixed": "1",
        "requires_reasoning": "0",
    }
    assert rows[1]["is_reasoning"] == "True"
    assert rows[1]["think_temp_fixed"] == "1"
    assert rows[2]["input_price_tier2"] == "6.0"
    assert rows[2]["tier_threshold"] == "200000"
    assert rows[3]["provider"] == "bedrock"
    assert rows[3]["input_price_tier2"] == "6.0"
    assert rows[3]["tier_threshold"] == "200000"
    assert rows[4]["input_price_tier2"] == "2.5"


def test_generate_embedding_rows_maps_azure_alias_and_manual_google_entry():
    rows = generate_embedding_rows(
        _payload(),
        [
            TargetModel("embedding", "azure-text-embedding-3-small", "azure"),
            TargetModel("embedding", "gemini-embedding-2-preview", "google"),
        ],
    )

    assert list(rows[0]) == EMBEDDING_HEADERS
    assert rows == [
        {
            "model_name": "azure-text-embedding-3-small",
            "provider": "azure",
            "input_price": "0.02",
        },
        {
            "model_name": "gemini-embedding-2-preview",
            "provider": "google",
            "input_price": "0.2",
        },
    ]


def test_load_models_dev_payload_uses_browser_safe_user_agent(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"ok": true}'

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(
        "shinka.tools.pricing.generate_csvs.urllib.request.urlopen", fake_urlopen
    )

    assert load_models_dev_payload("https://models.dev/api.json") == {"ok": True}
    assert captured["request"].headers["User-agent"] == MODELS_DEV_USER_AGENT
    assert captured["timeout"] == 60


def test_generator_fails_when_target_has_no_upstream_or_overlay():
    with pytest.raises(ValueError, match="No models.dev model or manual LLM overlay"):
        generate_llm_rows(
            _payload(),
            [TargetModel("llm", "missing-model", "openai")],
        )


def test_check_mode_reports_drift(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    payload_path = tmp_path / "models.json"
    llm_path = tmp_path / "llm.csv"
    embedding_path = tmp_path / "embedding.csv"
    payload_path.write_text(json.dumps(_payload()), encoding="utf-8")
    llm_path.write_text(
        render_csv(
            LLM_HEADERS,
            [
                {
                    "model_name": "o3-mini-2025-01-31",
                    "provider": "openai",
                    "input_price": "0.0",
                    "output_price": "0.0",
                    "input_price_tier2": "",
                    "output_price_tier2": "",
                    "tier_threshold": "",
                    "is_reasoning": "False",
                    "think_temp_fixed": "0",
                    "requires_reasoning": "0",
                }
            ],
        ),
        encoding="utf-8",
    )
    embedding_path.write_text(
        render_csv(
            EMBEDDING_HEADERS,
            [
                {
                    "model_name": "azure-text-embedding-3-small",
                    "provider": "azure",
                    "input_price": "0.02",
                }
            ],
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--api-json",
            str(payload_path),
            "--llm-output",
            str(llm_path),
            "--embedding-output",
            str(embedding_path),
            "--check",
        ]
    )

    assert exit_code == 1
    assert str(llm_path) in capsys.readouterr().err
