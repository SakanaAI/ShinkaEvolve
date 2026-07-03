from .dbase import Repo, RepoDatabase, Program, ProgramDatabase, DatabaseConfig
from .async_dbase import AsyncRepoDatabase


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
    "RepoDatabase",
    "ProgramDatabase",
    "Program",
    "Repo",
    "DatabaseConfig",
    "AsyncRepoDatabase",
    "SystemPromptDatabase",
    "SystemPrompt",
    "SystemPromptConfig",
    "create_system_prompt",
]
