from .dbase import Program, ProgramDatabase, DatabaseConfig
from .async_dbase import AsyncProgramDatabase


def __getattr__(name):
    if name in {
        "SystemPromptDatabase",
        "SystemPrompt",
        "SystemPromptConfig",
        "create_system_prompt",
    }:
        from . import prompt_dbase

        return getattr(prompt_dbase, name)
    raise AttributeError(name)

__all__ = [
    "ProgramDatabase",
    "Program",
    "DatabaseConfig",
    "AsyncProgramDatabase",
    "SystemPromptDatabase",
    "SystemPrompt",
    "SystemPromptConfig",
    "create_system_prompt",
]
