def get_language_extension(language: str) -> str:
    if language == "cuda":
        return "cu"
    elif language == "cpp":
        return "cpp"
    elif language == "python":
        return "py"
    elif language == "rust":
        return "rs"
    elif language == "swift":
        return "swift"
    elif language in ["json", "json5"]:
        return "json"
    else:
        raise ValueError(f"Language {language} not supported")
