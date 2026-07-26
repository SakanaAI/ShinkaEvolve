from typing import Optional, Tuple, List
import logging
from pathlib import Path
from shinka.database import Program
from shinka.llm import LLMClient
from shinka.prompts import NOVELTY_SYSTEM_MSG, NOVELTY_USER_MSG

logger = logging.getLogger(__name__)
DEFAULT_SUMMARY_FILENAME = ".shinka/individual.md"


class NoveltyJudge:
    """Handles novelty assessment using persisted individual repo summaries."""

    def __init__(
        self,
        novelty_llm_client: Optional[LLMClient] = None,
        language: str = "python",
        similarity_threshold: float = 1.0,
        max_novelty_attempts: int = 3,
        summary_filename: str = DEFAULT_SUMMARY_FILENAME,
    ):
        self.novelty_llm_client = novelty_llm_client
        self.language = language
        self.similarity_threshold = similarity_threshold
        self.max_novelty_attempts = max_novelty_attempts
        self.summary_filename = summary_filename

    def _format_summary_for_prompt(self, label: str, summary_text: str) -> str:
        text = summary_text.strip() or "No repository summary recorded."
        return f"{label} repository summary:\n\n{text}"

    def _summary_candidates(self, exec_path: Path) -> List[Path]:
        if exec_path.is_file() and exec_path.as_posix().endswith(self.summary_filename):
            return [exec_path]

        search_root = exec_path if exec_path.is_dir() else exec_path.parent
        return [root / self.summary_filename for root in [search_root, *search_root.parents]]

    def load_proposed_novelty_text(self, exec_fname: str) -> str:
        exec_path = Path(exec_fname)

        for candidate in self._summary_candidates(exec_path):
            if candidate.is_file():
                summary_text = candidate.read_text(encoding="utf-8").strip()
                if summary_text:
                    return summary_text

        if exec_path.is_file():
            logger.warning(
                "No repo summary found for %s; falling back to file contents for novelty check.",
                exec_fname,
            )
            return exec_path.read_text(encoding="utf-8")

        raise FileNotFoundError(
            f"Could not locate repo summary {self.summary_filename!r} for {exec_fname!r}"
        )

    def get_existing_novelty_text(self, most_similar_program: Program) -> str:
        repo_summary = (
            most_similar_program.repo_summary
            or (most_similar_program.metadata or {}).get("repo_summary")
            or ""
        ).strip()
        if repo_summary:
            return repo_summary

        logger.warning(
            "Most similar program %s is missing repo_summary; falling back to raw code.",
            most_similar_program.id,
        )
        return most_similar_program.code

    def build_novelty_user_message(
        self, proposed_summary: str, most_similar_program: Program
    ) -> str:
        existing_summary = self.get_existing_novelty_text(most_similar_program)
        return NOVELTY_USER_MSG.format(
            language="markdown",
            existing_code=self._format_summary_for_prompt("Existing", existing_summary),
            proposed_code=self._format_summary_for_prompt("Proposed", proposed_summary),
        )

    def should_check_novelty(
        self,
        code_embedding: List[float],
        generation: int,
        parent_program: Optional[Program],
        database,
    ) -> bool:
        """
        Check if novelty assessment should be performed.

        Args:
            code_embedding: Embedding vector of the proposed code
            generation: Current generation number
            parent_program: Parent program
            database: Database instance for similarity computation

        Returns:
            Boolean indicating if novelty check should be performed
        """
        if not code_embedding or generation == 0 or not parent_program:
            return False

        # Check if parent program has island information and islands are initialized
        if (
            parent_program.island_idx is not None
            and hasattr(database, "island_manager")
            and database.island_manager is not None
            and hasattr(database.island_manager, "are_all_islands_initialized")
            and database.island_manager.are_all_islands_initialized()
        ):
            return True

        return False

    def assess_novelty_with_rejection_sampling(
        self,
        proposed_summary: str,
        code_embedding: List[float],
        parent_program: Program,
        database,
    ) -> Tuple[bool, dict]:
        """
        Perform novelty assessment with rejection sampling.

        Args:
            proposed_summary: Proposed individual's repository summary text
            code_embedding: Embedding vector of the proposed summary
            parent_program: Parent program for island-based similarity
            database: Database instance for similarity computation

        Returns:
            Tuple of (should_accept, novelty_metadata)
        """
        novelty_metadata = {
            "novelty_checks_performed": 0,
            "novelty_total_cost": 0.0,
            "novelty_explanation": "",
            "max_similarity": 0.0,
            "similarity_scores": [],
        }
        summary_text = (proposed_summary or "").strip()

        for attempt in range(self.max_novelty_attempts):
            # Compute similarities with programs in island
            similarity_scores = database.compute_similarity(
                code_embedding, parent_program.island_idx
            )

            if not similarity_scores:
                logger.info(
                    f"NOVELTY CHECK {attempt + 1}/{self.max_novelty_attempts}: "
                    "Accepting program due to no similarity scores."
                )
                novelty_metadata["similarity_scores"] = []
                return True, novelty_metadata

            max_similarity = max(similarity_scores)
            sorted_similarity_scores = sorted(similarity_scores, reverse=True)
            formatted_similarities = [f"{s:.2f}" for s in sorted_similarity_scores]

            logger.info(f"Top-5 similarity scores: {formatted_similarities[:5]}")

            novelty_metadata["max_similarity"] = max_similarity
            novelty_metadata["similarity_scores"] = similarity_scores

            if max_similarity <= self.similarity_threshold:
                logger.info(
                    f"NOVELTY CHECK {attempt + 1}/{self.max_novelty_attempts}: "
                    f"Accepting program due to low similarity "
                    f"({max_similarity:.3f} <= {self.similarity_threshold})"
                )
                return True, novelty_metadata

            # High similarity detected - check with LLM if configured
            should_reject = True
            novelty_cost = 0.0

            if self.novelty_llm_client is not None:
                # Get the most similar program for LLM comparison
                most_similar_program = database.get_most_similar_program(
                    code_embedding, parent_program.island_idx
                )

                if most_similar_program:
                    try:
                        if not summary_text:
                            raise ValueError(
                                "Proposed repository summary is empty for novelty check"
                            )
                        is_novel, explanation, cost = self.check_llm_novelty(
                            summary_text, most_similar_program
                        )
                        should_reject = not is_novel
                        novelty_cost = cost
                        novelty_metadata["novelty_checks_performed"] += 1
                        novelty_metadata["novelty_total_cost"] += cost
                        novelty_metadata["novelty_explanation"] = explanation
                    except Exception as e:
                        logger.warning(f"Error during LLM novelty check: {e}")
                        should_reject = True  # Default to rejection on error

            if should_reject:
                logger.info(
                    f"NOVELTY CHECK {attempt + 1}/{self.max_novelty_attempts}: "
                    f"Rejecting program due to high similarity "
                    f"({max_similarity:.3f} > {self.similarity_threshold})"
                    + (
                        f" and LLM novelty check (cost: {novelty_cost:.4f})"
                        if novelty_cost > 0
                        else ""
                    )
                    + ". Retrying with different parent/inspirations."
                )
                # Continue to next attempt (rejection sampling)
                continue
            else:
                logger.info(
                    f"NOVELTY CHECK {attempt + 1}/{self.max_novelty_attempts}: "
                    f"Accepting program despite high similarity "
                    f"({max_similarity:.3f} > {self.similarity_threshold}) "
                    f"due to LLM novelty check (cost: {novelty_cost:.4f})."
                )
                return True, novelty_metadata

        # All attempts exhausted, reject the program
        logger.info(
            f"NOVELTY CHECK: Exhausted all {self.max_novelty_attempts} attempts, "
            "rejecting program."
        )
        return False, novelty_metadata

    def check_llm_novelty(
        self, proposed_summary: str, most_similar_program: Program
    ) -> Tuple[bool, str, float]:
        """
        Use LLM to judge if the proposed summary is meaningfully different from
        the most similar program.

        Args:
            proposed_summary: The newly generated individual repo summary
            most_similar_program: The most similar existing program

        Returns:
            Tuple of (is_novel, explanation, api_cost)
        """
        if not self.novelty_llm_client:
            logger.debug("Novelty LLM not configured, skipping novelty check")
            return True, "No novelty LLM configured", 0.0

        user_msg = self.build_novelty_user_message(
            proposed_summary, most_similar_program
        )

        try:
            response = self.novelty_llm_client.query(
                msg=user_msg,
                system_msg=NOVELTY_SYSTEM_MSG,
                llm_kwargs=self.novelty_llm_client.get_kwargs(),
            )

            if response is None or response.content is None:
                logger.warning("Novelty LLM returned empty response")
                return True, "LLM response was empty", 0.0

            content = response.content.strip()
            api_cost = response.cost or 0.0

            # Parse the response
            is_novel = content.upper().startswith(
                "NOVEL"
            ) or content.upper().startswith("**NOVEL**")
            explanation = content
            return is_novel, explanation, api_cost

        except Exception as e:
            logger.error(f"Error in novelty LLM check: {e}")
            return True, f"Error in novelty check: {e}", 0.0

    def log_novelty_skip_message(self, reason: str) -> None:
        """Log a message about skipping novelty check."""
        logger.info(f"NOVELTY CHECK: Skipping rejection sampling - {reason}")
