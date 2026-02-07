# from .runner import EvolutionRunner, EvolutionConfig
# from .sampler import PromptSampler
# from .summarizer import MetaSummarizer
# from .novelty_judge import NoveltyJudge
# from .wrap_eval import run_shinka_eval

# __all__ = [
#     "EvolutionRunner",
#     "PromptSampler",
#     "MetaSummarizer",
#     "NoveltyJudge",
#     "EvolutionConfig",
#     "run_shinka_eval",
# ]


# Lazy access so importing shinka.core doesn't eagerly import heavy modules.
# Only import what is actually accessed.

__all__ = [
    "EvolutionRunner",
    "EvolutionConfig",
    "PromptSampler",
    "MetaSummarizer",
    "NoveltyJudge",
    "run_shinka_eval",
]

def __getattr__(name):
    if name == "run_shinka_eval":
        from .wrap_eval import run_shinka_eval
        return run_shinka_eval

    if name in ("EvolutionRunner", "EvolutionConfig"):
        from .runner import EvolutionRunner, EvolutionConfig
        return {"EvolutionRunner": EvolutionRunner, "EvolutionConfig": EvolutionConfig}[name]

    if name == "PromptSampler":
        from .sampler import PromptSampler
        return PromptSampler

    if name == "MetaSummarizer":
        from .summarizer import MetaSummarizer
        return MetaSummarizer

    if name == "NoveltyJudge":
        from .novelty_judge import NoveltyJudge
        return NoveltyJudge

    raise AttributeError(f"module 'shinka.core' has no attribute {name!r}")