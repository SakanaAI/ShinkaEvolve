_LANGUAGE_ALIASES: dict[str, str] = {
    "jl": "julia",
    "py": "python",
    "python3": "python",
    "c++": "cpp",
    "cxx": "cpp",
    "cc": "cpp",
    "cu": "cuda",
}

_LANGUAGE_EXTENSIONS: dict[str, str] = {
    "cuda": "cu",
    "cpp": "cpp",
    "python": "py",
    "rust": "rs",
    "swift": "swift",
    "json": "json",
    "json5": "json",
    "julia": "jl",
}

_EVOLVE_COMMENT_PREFIXES: dict[str, str] = {
    "cuda": "//",
    "cpp": "//",
    "python": "#",
    "rust": "//",
    "swift": "//",
    "json": "//",
    "json5": "//",
    "julia": "#",
}

_LANGUAGE_FENCE_TAGS: dict[str, tuple[str, ...]] = {
    "cuda": ("cuda", "cu"),
    "cpp": ("cpp", "c++", "cc", "cxx"),
    "python": ("python", "py"),
    "rust": ("rust", "rs"),
    "swift": ("swift",),
    "json": ("json",),
    "json5": ("json5", "json"),
    "julia": ("julia", "jl"),
}


def normalize_language(language: str) -> str:
    normalized = language.strip().lower()
    canonical = _LANGUAGE_ALIASES.get(normalized, normalized)
    if canonical not in _LANGUAGE_EXTENSIONS:
        raise ValueError(f"Language {language} not supported")
    return canonical


def get_language_extension(language: str) -> str:
    return _LANGUAGE_EXTENSIONS[normalize_language(language)]


def get_evolve_comment_prefix(language: str) -> str:
    return _EVOLVE_COMMENT_PREFIXES[normalize_language(language)]


def get_code_fence_languages(language: str) -> tuple[str, ...]:
    canonical = normalize_language(language)
    requested_tag = language.strip().lower()
    ordered_tags = [requested_tag]
    for tag in _LANGUAGE_FENCE_TAGS[canonical]:
        if tag not in ordered_tags:
            ordered_tags.append(tag)
    return tuple(ordered_tags)
