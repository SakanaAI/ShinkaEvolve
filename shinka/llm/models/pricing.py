# Available models and pricing
# Anthropic: https://www.anthropic.com/pricing#anthropic-api
# OpenAI: https://platform.openai.com/docs/pricing
# DeepSeek: https://api-docs.deepseek.com/quick_start/pricing/
# Gemini: https://ai.google.dev/gemini-api/docs/pricing

M = 1000000

CLAUDE_MODELS = {
    "claude-3-5-haiku-20241022": {
        "input_price": 0.8 / M,
        "output_price": 4.0 / M,
    },
    "claude-3-5-sonnet-20241022": {
        "input_price": 3.0 / M,
        "output_price": 15.0 / M,
    },
    "claude-3-opus-20240229": {
        "input_price": 15.0 / M,
        "output_price": 75.0 / M,
    },
    "claude-3-7-sonnet-20250219": {
        "input_price": 3.0 / M,
        "output_price": 15.0 / M,
    },
    "us.anthropic.claude-3-7-sonnet-20250219-v1:0": {
        "input_price": 3.0 / M,
        "output_price": 15.0 / M,
    },
    "claude-4-sonnet-20250514": {
        "input_price": 3.0 / M,
        "output_price": 15.0 / M,
    },
    "us.anthropic.claude-sonnet-4-20250514-v1:0": {
        "input_price": 3.0 / M,
        "output_price": 15.0 / M,
    },
    "claude-sonnet-4-5-20250929": {
        "input_price": 3.0 / M,
        "output_price": 15.0 / M,
    },
}

OPENAI_MODELS = {
    "gpt-4o-mini": {
        "input_price": 0.15 / M,
        "output_price": 0.6 / M,
    },
    "gpt-4o-2024-08-06": {
        "input_price": 2.5 / M,
        "output_price": 10.0 / M,
    },
    "gpt-4o-mini-2024-07-18": {
        "input_price": 0.15 / M,
        "output_price": 0.6 / M,
    },
    "o1-2024-12-17": {
        "input_price": 15.0 / M,
        "output_price": 60.0 / M,
    },
    "o3-mini-2025-01-31": {
        "input_price": 1.1 / M,
        "output_price": 4.4 / M,
    },
    "o3-mini": {
        "input_price": 1.1 / M,
        "output_price": 4.4 / M,
    },
    "gpt-4.5-preview-2025-02-27": {
        "input_price": 75.0 / M,
        "output_price": 150.0 / M,
    },
    "gpt-4.1-2025-04-14": {
        "input_price": 2.0 / M,
        "output_price": 8.0 / M,
    },
    "gpt-4.1": {
        "input_price": 2.0 / M,
        "output_price": 8.0 / M,
    },
    "gpt-4.1-mini-2025-04-14": {
        "input_price": 0.4 / M,
        "output_price": 1.6 / M,
    },
    "gpt-4.1-mini": {
        "input_price": 0.4 / M,
        "output_price": 1.6 / M,
    },
    "gpt-4.1-nano-2025-04-14": {
        "input_price": 0.1 / M,
        "output_price": 1.4 / M,
    },
    "gpt-4.1-nano": {
        "input_price": 0.1 / M,
        "output_price": 1.4 / M,
    },
    "o3-2025-04-16": {
        "input_price": 10.0 / M,
        "output_price": 40.0 / M,
    },
    "o4-mini-2025-04-16": {
        "input_price": 1.1 / M,
        "output_price": 4.4 / M,
    },
    "o4-mini": {
        "input_price": 1.1 / M,
        "output_price": 4.4 / M,
    },
    "gpt-5": {
        "input_price": 1.25 / M,
        "output_price": 10.0 / M,
    },
    "gpt-5-mini": {
        "input_price": 0.25 / M,
        "output_price": 2.0 / M,
    },
    "gpt-5-nano": {
        "input_price": 0.05 / M,
        "output_price": 0.4 / M,
    },
    "gpt-5.1": {
        "input_price": 1.25 / M,
        "output_price": 10.0 / M,
    },
}


DEEPSEEK_MODELS = {
    "deepseek-chat": {
        "input_price": 0.27 / M,
        "output_price": 1.1 / M,
    },
    "deepseek-reasoner": {
        "input_price": 0.55 / M,
        "output_price": 2.19 / M,
    },
}

GEMINI_MODELS = {
    "gemini-2.5-pro": {
        "input_price": 1.25 / M,
        "output_price": 10.0 / M,
    },
    "gemini-2.5-flash": {
        "input_price": 0.3 / M,
        "output_price": 2.5 / M,
    },
    "gemini-2.5-flash-lite-preview-06-17": {
        "input_price": 0.1 / M,
        "output_price": 0.4 / M,
    },
    "gemini-3-pro-preview" : {
        "input_price": 2.0 / M,
        "output_price": 12.0 / M,
    },
}

BEDROCK_MODELS = {
    "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0": CLAUDE_MODELS[
        "claude-3-5-sonnet-20241022"
    ],
    "bedrock/anthropic.claude-3-5-haiku-20241022-v1:0": CLAUDE_MODELS[
        "claude-3-5-haiku-20241022"
    ],
    "bedrock/anthropic.claude-3-opus-20240229-v1:0": CLAUDE_MODELS[
        "claude-3-opus-20240229"
    ],
    "bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0": CLAUDE_MODELS[
        "claude-3-7-sonnet-20250219"
    ],
    "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0": CLAUDE_MODELS[
        "claude-4-sonnet-20250514"
    ],
}

REASONING_OAI_MODELS = [
    "o3-mini-2025-01-31",
    "o1-2024-12-17",
    "o3-2025-04-16",
    "o4-mini-2025-04-16",
    "o4-mini",
    "o3-mini",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
]

REASONING_CLAUDE_MODELS = [
    "claude-3-7-sonnet-20250219",
    "claude-4-sonnet-20250514",
    "claude-sonnet-4-5-20250929",
]

REASONING_DEEPSEEK_MODELS = [
    "deepseek-reasoner",
]

REASONING_GEMINI_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite-preview-06-17",
    "gemini-3-pro-preview",
]

REASONING_AZURE_MODELS = [
    "azure-o3-mini",
    "azure-o4-mini",
    "azure-gpt-5",
    "azure-gpt-5-mini",
    "azure-gpt-5-nano",
]

REASONING_BEDROCK_MODELS = [
    "bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0",
]
