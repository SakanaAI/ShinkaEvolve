from typing import Any, Iterable, List, Union, Optional
import random
from .providers.pricing import (
    is_reasoning_model,
    has_fixed_temperature,
    requires_reasoning,
)
from .providers.model_resolver import resolve_model_backend
from .providers.errors import NonRetryableLLMError
from .constants import (
    CLAUDE_ADAPTIVE_MODELS,
    GEMINI_THINKING_LEVEL_BY_EFFORT,
    GEMINI_THINKING_LEVEL_MODELS,
    OPENAI_GPT6_MODELS,
    OPENAI_GPT6_REQUIRED_REASONING_MODELS,
)
import logging

logger = logging.getLogger(__name__)

# Every effort a config may name. Each provider branch below maps all of them.
# "min" is Shinka's lowest tier; "minimal" is OpenAI's named level, passed
# through where the API has it and treated as "min" elsewhere. "disabled" asks
# for no reasoning where the model allows that; "none" is the same except on
# legacy OpenAI reasoning models, which receive the literal effort "none".
REASONING_EFFORTS = (
    "disabled",
    "none",
    "min",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)
# "" is the sampling default and "auto" was the client default before the
# provider split; both mean that no reasoning is requested.
_REASONING_EFFORT_ALIASES = {"": "disabled", "auto": "disabled"}
_warned_about_auto_effort = False

THINKING_TOKENS = {
    "min": 1024,
    "minimal": 1024,
    "low": 2048,
    "medium": 4096,
    "high": 8192,
    "xhigh": 12288,
    "max": 16384,
}

NO_TEMPERATURE_MODELS = {
    "claude-opus-4-8",
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    *GEMINI_THINKING_LEVEL_MODELS,
    "us.anthropic.claude-opus-4-8",
}


def normalize_reasoning_effort(effort: Optional[str]) -> str:
    """Return the canonical name of a configured reasoning effort, or raise."""
    name = "" if effort is None else str(effort).strip().lower()
    if name == "auto":
        _warn_about_auto_effort()
    name = _REASONING_EFFORT_ALIASES.get(name, name)
    if name not in REASONING_EFFORTS:
        raise NonRetryableLLMError(
            f"Unsupported reasoning effort: {effort!r}. "
            f"Use one of: {', '.join(REASONING_EFFORTS)}."
        )
    return name


def _warn_about_auto_effort() -> None:
    global _warned_about_auto_effort
    if not _warned_about_auto_effort:
        _warned_about_auto_effort = True
        logger.warning(
            'Reasoning effort "auto" is deprecated and treated as "disabled"; '
            "name an effort explicitly."
        )


def normalize_reasoning_efforts(
    efforts: Union[Iterable[Optional[str]], str, None],
) -> List[str]:
    """Validate every configured effort up front, not only the one sampled."""
    if efforts is None or isinstance(efforts, str):
        efforts = [efforts]
    return [normalize_reasoning_effort(effort) for effort in efforts]


def _current_reasoning_effort(effort: str) -> str:
    """Map a normalised effort onto the levels of the current native APIs."""
    if effort in ("disabled", "none"):
        return "none"
    if effort in ("min", "minimal"):
        return "low"
    return effort


def _current_claude_kwargs(model: str, effort: str, max_tokens: int) -> dict[str, Any]:
    effort = _current_reasoning_effort(effort)
    thinking = {"type": "adaptive", "display": "summarized"}
    if effort == "none":
        effort = "low"
        if model == "claude-sonnet-5-5":
            thinking = {"type": "between_tools"}
        elif model == "claude-haiku-5-5":
            thinking = {"type": "disabled"}
    return {
        "max_tokens": min(max_tokens, 128_000),
        "thinking": thinking,
        "output_config": {"effort": effort},
    }


def sample_batch_kwargs(
    num_samples: int,
    model_names: Union[List[str], str] = "gpt-4o-mini-2024-07-18",
    temperatures: Union[List[float], float] = 0.0,
    max_tokens: Union[List[int], int] = 4096,
    reasoning_efforts: Union[List[str], str] = "",
    model_sample_probs: Optional[List[float]] = None,
    unique_filter: bool = False,
):
    """Sample a dictionary of kwargs for a given model."""
    all_kwargs: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = num_samples * 10  # Prevent infinite loops

    while len(all_kwargs) < num_samples and attempts < max_attempts:
        kwargs_dict = sample_model_kwargs(
            model_names=model_names,
            temperatures=temperatures,
            max_tokens=max_tokens,
            reasoning_efforts=reasoning_efforts,
            model_sample_probs=model_sample_probs,
        )

        if unique_filter:
            if kwargs_dict not in all_kwargs:
                all_kwargs.append(kwargs_dict)
        else:
            all_kwargs.append(kwargs_dict)

        attempts += 1

    if len(all_kwargs) < num_samples:
        logger.info(
            f"Could not generate {num_samples} unique kwargs combinations "
            f"after {max_attempts} attempts"
        )
        logger.info(f"Returning {len(all_kwargs)} unique kwargs combinations.")

    return all_kwargs


def sample_model_kwargs(
    model_names: Union[List[str], str] = "gpt-4o-mini-2024-07-18",
    temperatures: Union[List[float], float] = 0.0,
    max_tokens: Union[List[int], int] = 4096,
    reasoning_efforts: Union[List[str], str] = "",
    model_sample_probs: Optional[List[float]] = None,
):
    """Sample a dictionary of kwargs for a given model."""
    # Make all inputs lists
    if isinstance(model_names, str):
        model_names = [model_names]
    if isinstance(temperatures, (int, float)):
        temperatures = [temperatures]
    if isinstance(max_tokens, int):
        max_tokens = [max_tokens]
    reasoning_efforts = normalize_reasoning_efforts(reasoning_efforts)

    kwargs_dict: dict[str, Any] = {}

    # 1. SAMPLE: model name
    if model_sample_probs is not None:
        if len(model_sample_probs) != len(model_names):
            raise ValueError(
                "model_sample_probs must have the same length as model_names"
            )
        if not abs(sum(model_sample_probs) - 1.0) < 1e-9:
            raise ValueError("model_sample_probs must sum to 1")
        kwargs_dict["model_name"] = random.choices(
            model_names, weights=model_sample_probs, k=1
        )[0]
    else:
        kwargs_dict["model_name"] = random.choice(model_names)

    model_name = kwargs_dict["model_name"]
    resolved_model = resolve_model_backend(model_name)
    api_model_name = resolved_model.api_model_name
    provider = resolved_model.provider

    if provider == "headless":
        return kwargs_dict

    # Current native APIs have effort contracts distinct from legacy budgets.
    if provider in ("openai", "azure_openai") and api_model_name in OPENAI_GPT6_MODELS:
        effort = _current_reasoning_effort(random.choice(reasoning_efforts))
        if effort == "none" and api_model_name in OPENAI_GPT6_REQUIRED_REASONING_MODELS:
            effort = "low"
        reasoning = {"effort": effort}
        if effort != "none":
            reasoning["summary"] = "auto"
        return {
            **kwargs_dict,
            "max_output_tokens": random.choice(max_tokens),
            "reasoning": reasoning,
        }

    claude_model = (
        api_model_name.rsplit(".", 1)[-1] if provider == "bedrock" else api_model_name
    )
    if provider in ("anthropic", "bedrock") and claude_model in CLAUDE_ADAPTIVE_MODELS:
        return {
            **kwargs_dict,
            **_current_claude_kwargs(
                claude_model,
                random.choice(reasoning_efforts),
                random.choice(max_tokens),
            ),
        }

    # 2. SAMPLE: reasoning effort
    if is_reasoning_model(model_name):
        r_effort = random.choice(reasoning_efforts)
    else:
        r_effort = "disabled"
    # Only the OpenAI Responses API has a literal "none" effort.
    if r_effort == "none" and provider not in ("openai", "openrouter", "azure_openai"):
        r_effort = "disabled"

    # Some opennrouter models only support running with reasoning effort
    if requires_reasoning(model_name) and r_effort == "disabled":
        r_effort = "low"

    # 3. SAMPLE: temperature with possible reasoning restrictions
    include_temperature = api_model_name not in NO_TEMPERATURE_MODELS and not (
        provider in ("openai", "azure_openai") and is_reasoning_model(model_name)
    )
    if (
        include_temperature
        and has_fixed_temperature(model_name)
        and (
            r_effort != "disabled"
            or provider in ("openai", "openrouter", "azure_openai")
        )
    ):
        kwargs_dict["temperature"] = 1.0
    elif include_temperature:
        kwargs_dict["temperature"] = random.choice(temperatures)

    # 4.a) SET: max_output_tokens for OpenAI reasoning effort
    if provider in ("openai", "openrouter", "azure_openai") and is_reasoning_model(
        model_name
    ):
        kwargs_dict["max_output_tokens"] = random.choice(max_tokens)
        if r_effort == "disabled":
            kwargs_dict["reasoning"] = {"effort": None}
        else:
            # "none", "minimal" and "xhigh" exist only on some of these models.
            kwargs_dict["reasoning"] = {
                "effort": {"min": "low", "max": "high"}.get(r_effort, r_effort)
            }

        # 4.b.1) SET: auto-summarization for OpenAI reasoning effort
        if provider == "openai" and r_effort != "disabled":
            kwargs_dict["reasoning"]["summary"] = "auto"

    # 4.b) SET: max_tokens for Google reasoning effort
    elif provider == "google" and is_reasoning_model(model_name):
        kwargs_dict["max_tokens"] = random.choice(max_tokens)
        if api_model_name in GEMINI_THINKING_LEVEL_MODELS:
            if r_effort != "disabled":
                kwargs_dict["thinking_level"] = (
                    "MINIMAL"
                    if api_model_name == "gemini-3.5-flash-lite"
                    and r_effort in ("min", "minimal")
                    else GEMINI_THINKING_LEVEL_BY_EFFORT[r_effort]
                )
        elif r_effort != "disabled":
            t = THINKING_TOKENS[r_effort]
            thinking_tokens = t if t < kwargs_dict["max_tokens"] else 1024
            kwargs_dict["thinking_budget"] = thinking_tokens
        else:
            if api_model_name in ("gemini-2.5-pro", "gemini-3-pro-preview"):
                kwargs_dict["thinking_budget"] = 128
            else:
                kwargs_dict["thinking_budget"] = 0

    # 4.c) SET: max_tokens for Anthropic or Bedrock reasoning effort
    elif provider in ("anthropic", "bedrock") and is_reasoning_model(model_name):
        kwargs_dict["max_tokens"] = min(random.choice(max_tokens), 64000)
        think_bool = r_effort != "disabled"
        if think_bool:
            # filter thinking tokens to be smaller than max_tokens
            # not auto THINKING_TOKENS
            t = THINKING_TOKENS[r_effort]
            thinking_tokens = t if t < kwargs_dict["max_tokens"] else 1024
            # sample only from thinking tokens that are valid
            kwargs_dict["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_tokens,
            }

    # 4.d) SET: max_tokens and thinking mode for DeepSeek reasoning models
    elif provider == "deepseek" and is_reasoning_model(model_name):
        kwargs_dict["max_tokens"] = random.choice(max_tokens)
        if r_effort == "disabled":
            kwargs_dict["extra_body"] = {"thinking": {"type": "disabled"}}
        else:
            kwargs_dict["reasoning_effort"] = "max" if r_effort == "max" else "high"
            kwargs_dict["extra_body"] = {"thinking": {"type": "enabled"}}

    # 4.e) SET: max_tokens for all other models
    else:
        # Non-reasoning models or other providers
        if provider in ("anthropic", "bedrock", "deepseek", "local_openai"):
            kwargs_dict["max_tokens"] = random.choice(max_tokens)
        else:
            kwargs_dict["max_output_tokens"] = random.choice(max_tokens)

    return kwargs_dict
