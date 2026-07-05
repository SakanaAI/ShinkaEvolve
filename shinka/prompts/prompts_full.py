# Multiple Full Rewrite Prompt Variants
# 1. Default
# 2. Different Algorithm
# 3. Context Motivated
# 4. Structural Redesign
# 5. Parametric Design

# Original/Default Full Rewrite
FULL_SYS_FORMAT_DEFAULT = """
Rewrite the relevant repository implementation to improve performance on the specified metrics.
Edit the repository directly in the active working directory.
Do not return a standalone full-code response for Shinka to apply.

* Maintain the evaluator-facing behavior expected by the original repository while improving the internal implementation.
* Make sure the repository still runs after your changes.
""".rstrip()

# Variant 1: Completely Different Algorithm
FULL_SYS_FORMAT_DIFFERENT = """
Design a completely different algorithm approach to solve the same problem.
Ignore the current implementation and think of alternative algorithmic strategies that could achieve better performance.
Edit the repository directly in the active working directory.
Do not return a standalone full-code response for Shinka to apply.

* Your algorithm should solve the same problem but use a fundamentally different approach.
* Ensure the evaluator-facing behavior is maintained.
* Think outside the box - consider different data structures, algorithms, or paradigms.
""".rstrip()


# Variant 2: Motivated by Context but Different
FULL_SYS_FORMAT_MOTIVATED = """
Create a novel algorithm that draws inspiration from the provided context repository individuals but implements a fundamentally different approach.
Study the patterns and techniques from the examples, then design something new.
Edit the repository directly in the active working directory.
Do not return a standalone full-code response for Shinka to apply.

* Learn from the context repository individuals but don't copy their approaches directly.
* Combine ideas in novel ways or apply insights to different algorithmic paradigms.
* Maintain the evaluator-facing behavior expected by the original repository.
""".rstrip()


# Variant 3: Structural Modification
FULL_SYS_FORMAT_STRUCTURAL = """
Redesign the repository implementation with a different structural approach while potentially using similar core concepts.
Focus on changing the overall architecture, data flow, or repository organization.
Edit the repository directly in the active working directory.
Do not return a standalone full-code response for Shinka to apply.

* Focus on changing the implementation structure: modularization, data flow, control flow, or architectural patterns.
* The core problem-solving approach may be similar but organized differently.
* Ensure the evaluator-facing behavior is maintained.
""".rstrip()


# Variant 4: Parameter-Based Algorithm Design
FULL_SYS_FORMAT_PARAMETRIC = """
Analyze the current repository individual to identify its key parameters and algorithmic components, then design a new algorithm with different parameter settings and configurations.
Edit the repository directly in the active working directory.
Do not return a standalone full-code response for Shinka to apply.

* Identify parameters like: learning rates, iteration counts, thresholds, weights, selection criteria, etc.
* Design a new algorithm with different parameter values or configurations.
* Consider adaptive parameters, different optimization strategies, or alternative heuristics.
* Maintain the evaluator-facing behavior expected by the original repository.
""".rstrip()

# List of all variants for sampling
FULL_SYS_FORMATS = [
    FULL_SYS_FORMAT_DEFAULT,
    FULL_SYS_FORMAT_DIFFERENT,
    FULL_SYS_FORMAT_MOTIVATED,
    FULL_SYS_FORMAT_STRUCTURAL,
    FULL_SYS_FORMAT_PARAMETRIC,
]

# Variant names for debugging/logging
FULL_SYS_FORMAT_NAMES = [
    "default",
    "different_algorithm",
    "context_motivated",
    "structural_redesign",
    "parametric_design",
]

FULL_ITER_MSG = """# Current repository individual

Here is the current repository summary:
{code_content}

Here are the performance metrics of the repository individual:

{performance_metrics}{text_feedback_section}

# Task

Rewrite the relevant repository implementation to improve performance on the specified metrics.

IMPORTANT: Make sure your changes maintain the evaluator-facing behavior expected by the original repository while improving the internal implementation.
""".rstrip()
