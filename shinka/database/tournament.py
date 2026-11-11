"""Island-local tournament selection from the highest-scoring archive entries."""

import json
import logging
import sys

import numpy as np

from .parents import ParentSamplingStrategy

logger = logging.getLogger(__name__)


class TournamentSamplingStrategy(ParentSamplingStrategy):
    """Draw distinct contestants from the top K and return a best-scoring one."""

    def _candidates(self, archived):
        join = "JOIN archive a ON a.program_id = p.id" if archived else ""
        island_clause = "AND p.island_idx = ?" if self.island_idx is not None else ""
        parameters = [-sys.float_info.max, sys.float_info.max]
        if self.island_idx is not None:
            parameters.append(self.island_idx)
        parameters.append(self.config.tournament_top_k)
        self.cursor.execute(
            f"""SELECT p.id, p.combined_score FROM programs p {join}
                WHERE p.correct = 1 AND p.combined_score BETWEEN ? AND ?
                {island_clause}
                ORDER BY p.combined_score DESC, p.id ASC LIMIT ?""",
            parameters,
        )
        return self.cursor.fetchall()

    def sample_parent(self):
        candidates = self._candidates(archived=True)
        if not candidates:
            candidates = self._candidates(archived=False)
        if not candidates:
            raise ValueError(
                f"No correct, finite-scored tournament candidates in island {self.island_idx}"
            )
        draws = min(self.config.tournament_num_draws, len(candidates))
        indices = np.random.choice(len(candidates), size=draws, replace=False)
        contestants = [candidates[int(index)] for index in indices]
        best_score = max(row["combined_score"] for row in contestants)
        winners = [row for row in contestants if row["combined_score"] == best_score]
        winner = winners[int(np.random.choice(len(winners)))]
        logger.info(
            "Tournament sampling: %s",
            json.dumps(
                {
                    "island": self.island_idx,
                    "top_k": self.config.tournament_top_k,
                    "pool_size": len(candidates),
                    "draws": draws,
                    "contestants": [dict(row) for row in contestants],
                    "winner": winner["id"],
                }
            ),
        )
        return self.get_program(winner["id"])
