import json
import logging
import random
import sqlite3
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Set
import rich.box  # type: ignore
import rich  # type: ignore
from rich.console import Console as RichConsole  # type: ignore
from rich.table import Table as RichTable  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class IslandMigrationParams:
    migration_rate: float
    migration_interval: int
    island_elitism: float


@dataclass
class MigrationSummary:
    generation: int
    total_migrated: int
    per_island: Dict[int, int]
    migrations: Dict[int, Dict[int, List[str]]]
    policies_used: Dict[int, Dict[str, str]]

class IslandStrategy(ABC):
    """Abstract base class for island strategies."""

    def __init__(
        self,
        cursor: sqlite3.Cursor,
        conn: sqlite3.Connection,
        config: Any,
    ):
        self.cursor = cursor
        self.conn = conn
        self.config = config

    @abstractmethod
    def assign_island(self, program: Any) -> None:
        """Assign an island to a program."""
        pass

    def get_initialized_islands(self) -> List[int]:
        """Get list of islands that have correct programs.
        Default implementation for base class."""
        self.cursor.execute(
            """SELECT DISTINCT island_idx FROM programs
                WHERE correct = 1 AND island_idx IS NOT NULL"""
        )
        islands_with_correct = {
            row["island_idx"]
            for row in self.cursor.fetchall()
            if row["island_idx"] is not None
        }
        return list(islands_with_correct)


class DefaultIslandAssignmentStrategy(IslandStrategy):
    """Default strategy for assigning programs to islands."""

    def get_initialized_islands(self) -> List[int]:
        self.cursor.execute(
            """SELECT DISTINCT island_idx FROM programs
                WHERE correct = 1 AND island_idx IS NOT NULL"""
        )
        islands_with_correct = {
            row["island_idx"]
            for row in self.cursor.fetchall()
            if row["island_idx"] is not None
        }
        return list(islands_with_correct)

    def assign_island(self, program: Any) -> None:
        """
        Assigns an island index to a program.
        - Children are placed on the same island as their parents.
        - Initial correct programs are distributed one per island.
        - Other initial programs are placed randomly, preferring empty islands.
        """
        num_islands = getattr(self.config, "num_islands", 0)
        if num_islands <= 0:
            program.island_idx = 0
            return

        # Check for uninitialized islands (islands with no programs at all)
        islands_with_correct = self.get_initialized_islands()
        islands_without_correct = [
            i for i in range(num_islands) if i not in islands_with_correct
        ]
        if islands_without_correct:
            program.island_idx = min(islands_without_correct)
            logger.debug(
                f"Assigned correct program {program.id} to island "
                f"{program.island_idx} (first without correct program)"
            )
            return

        # If the program has a parent, it inherits the parent's island.
        if program.parent_id:
            self.cursor.execute(
                "SELECT island_idx FROM programs WHERE id = ?", (program.parent_id,)
            )
            row = self.cursor.fetchone()
            if row and row["island_idx"] is not None:
                program.island_idx = row["island_idx"]
                logger.debug(
                    f"Assigned program {program.id} to parent's island "
                    f"{program.island_idx}"
                )
                return

        # Final fallback: assign to a random island
        program.island_idx = random.randint(0, num_islands - 1)
        logger.debug(
            f"Assigned program {program.id} to random island "
            f"{program.island_idx} (all assignment strategies exhausted)"
        )


class CopyInitialProgramIslandStrategy(IslandStrategy):
    """Strategy that copies the initial program to each island."""

    def get_initialized_islands(self) -> List[int]:
        self.cursor.execute(
            """SELECT DISTINCT island_idx FROM programs
                WHERE correct = 1 AND island_idx IS NOT NULL"""
        )
        islands_with_correct = {
            row["island_idx"]
            for row in self.cursor.fetchall()
            if row["island_idx"] is not None
        }
        return list(islands_with_correct)

    def assign_island(self, program: Any) -> None:
        """
        Assigns an island index to a program.
        - Children are placed on the same island as their parents.
        - For the first program added, it gets assigned to island 0 and copies
          are created for all other islands.
        - Other programs follow normal assignment rules.
        """
        num_islands = getattr(self.config, "num_islands", 0)
        if num_islands <= 0:
            program.island_idx = 0
            return

        # Check if this is the very first program in the database
        self.cursor.execute("SELECT COUNT(*) FROM programs")
        program_count = (self.cursor.fetchone() or [0])[0]
        if program_count == 0:
            # This is the first program - assign to island 0
            program.island_idx = 0
            logger.debug(
                f"Assigned first program {program.id} to island 0, "
                "will create copies for other islands"
            )
            # Note: The copying will happen after this program is added
            # We'll set a flag in metadata to indicate copying is needed
            if program.metadata is None:
                program.metadata = {}
            program.metadata["_needs_island_copies"] = True
            return

        # If the program has a parent, it inherits the parent's island.
        if program.parent_id:
            self.cursor.execute(
                "SELECT island_idx FROM programs WHERE id = ?", (program.parent_id,)
            )
            row = self.cursor.fetchone()
            if row and row["island_idx"] is not None:
                program.island_idx = row["island_idx"]
                logger.debug(
                    f"Assigned program {program.id} to parent's island "
                    f"{program.island_idx}"
                )
                return

        # Check for uninitialized islands (islands with no correct programs)
        islands_with_correct = self.get_initialized_islands()
        islands_without_correct = [
            i for i in range(num_islands) if i not in islands_with_correct
        ]
        if islands_without_correct:
            program.island_idx = min(islands_without_correct)
            logger.debug(
                f"Assigned correct program {program.id} to island "
                f"{program.island_idx} (first without correct program)"
            )
            return

        # Final fallback: assign to a random island
        program.island_idx = random.randint(0, num_islands - 1)
        logger.debug(
            f"Assigned program {program.id} to random island "
            f"{program.island_idx} (all assignment strategies exhausted)"
        )


class IslandMigrationStrategy(ABC):
    """Abstract base class for island migration strategies."""

    def __init__(
        self,
        cursor: sqlite3.Cursor,
        conn: sqlite3.Connection,
        config: Any,
    ):
        self.cursor = cursor
        self.conn = conn
        self.config = config

    @abstractmethod
    def perform_migration(
        self,
        current_generation: int,
        eligible_islands: Optional[Sequence[int]] = None,
    ) -> MigrationSummary:
        """Perform migration between islands and return a summary."""
        raise NotImplementedError


class ElitistMigrationStrategy(IslandMigrationStrategy):
    """Migration strategy that supports adaptive parameters and policies."""

    def __init__(
        self,
        cursor: sqlite3.Cursor,
        conn: sqlite3.Connection,
        config: Any,
        param_provider: Optional[Callable[[int], Optional[IslandMigrationParams]]] = None,
        policy_provider: Optional[Callable[[int], Optional[Dict[str, str]]]] = None,
    ):
        super().__init__(cursor, conn, config)
        self.param_provider = param_provider
        self.policy_provider = policy_provider

    def perform_migration(
        self,
        current_generation: int,
        eligible_islands: Optional[Sequence[int]] = None,
    ) -> MigrationSummary:
        num_islands = getattr(self.config, "num_islands", 0)
        if num_islands < 2:
            return MigrationSummary(current_generation, 0, {}, {}, {})

        logger.info(
            f"Performing island migration at generation {current_generation}"
        )

        migrations_summary = defaultdict(lambda: defaultdict(list))
        all_migrated_programs = set()
        per_island_counts: Dict[int, int] = defaultdict(int)
        policies_used: Dict[int, Dict[str, str]] = {}

        island_iter = eligible_islands if eligible_islands else range(num_islands)
        island_best_scores = self._fetch_island_best_scores()

        for source_idx in island_iter:
            params = self.param_provider(source_idx) if self.param_provider else None
            migration_rate = (
                params.migration_rate
                if params is not None
                else getattr(self.config, "migration_rate", 0.1)
            )
            elitism_ratio = (
                params.island_elitism
                if params is not None
                else self._normalize_elitism(
                    getattr(self.config, "island_elitism", True)
                )
            )
            if migration_rate <= 0:
                continue

            island_size = self._get_island_size(source_idx)
            if island_size <= 1:
                continue

            num_migrants = max(1, int(island_size * migration_rate))
            policy = self.policy_provider(source_idx) if self.policy_provider else None
            if policy:
                policies_used[source_idx] = policy
                num_migrants = self._apply_size_policy(num_migrants, policy.get("size"))

            dest_islands = [i for i in range(num_islands) if i != source_idx]
            if not dest_islands:
                continue

            migrants = self._select_migrants(
                source_idx,
                island_size,
                num_migrants,
                elitism_ratio,
                (policy or {}).get("payload"),
            )

            unique_migrants = []
            for migrant_id in migrants:
                if migrant_id not in all_migrated_programs:
                    unique_migrants.append(migrant_id)
                    all_migrated_programs.add(migrant_id)
                else:
                    logger.warning(
                        f"Program {migrant_id[:8]}... already selected for "
                        "migration, skipping duplicate"
                    )

            donor_policy = (policy or {}).get("donor") if policy else None
            for migrant_id in unique_migrants:
                dest_idx = self._select_destination(
                    source_idx,
                    dest_islands,
                    donor_policy,
                    island_best_scores,
                )
                if dest_idx is None:
                    continue
                self._migrate_program(
                    migrant_id, source_idx, dest_idx, current_generation
                )
                migrations_summary[source_idx][dest_idx].append(migrant_id)
                per_island_counts[source_idx] += 1

        self.conn.commit()

        if migrations_summary:
            self._print_migration_summary(migrations_summary)

        total_migrated = sum(per_island_counts.values())
        logger.info(f"Migration complete. Migrated {total_migrated} programs.")
        return MigrationSummary(
            generation=current_generation,
            total_migrated=total_migrated,
            per_island=dict(per_island_counts),
            migrations={k: dict(v) for k, v in migrations_summary.items()},
            policies_used=policies_used,
        )

    def _get_island_size(self, island_idx: int) -> int:
        self.cursor.execute(
            "SELECT COUNT(*) FROM programs WHERE island_idx = ?",
            (island_idx,),
        )
        return (self.cursor.fetchone() or [0])[0]

    def _fetch_island_best_scores(self) -> Dict[int, float]:
        self.cursor.execute(
            "SELECT island_idx, MAX(combined_score) as best "
            "FROM programs WHERE island_idx IS NOT NULL AND correct = 1 "
            "GROUP BY island_idx"
        )
        return {
            row["island_idx"]: row["best"] if row["best"] is not None else 0.0
            for row in self.cursor.fetchall()
            if row["island_idx"] is not None
        }

    def _normalize_elitism(self, value: Any) -> float:
        if isinstance(value, bool):
            return 0.1 if value else 0.0
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return 0.0

    def _apply_size_policy(self, num_migrants: int, policy: Optional[str]) -> int:
        if policy is None:
            return num_migrants
        factors = {"small": 0.5, "medium": 1.0, "large": 1.5}
        factor = factors.get(policy, 1.0)
        adjusted = max(1, int(round(num_migrants * factor)))
        return adjusted

    def _select_destination(
        self,
        source_idx: int,
        dest_islands: List[int],
        donor_policy: Optional[str],
        island_best_scores: Dict[int, float],
    ) -> Optional[int]:
        if not dest_islands:
            return None
        num_islands = getattr(self.config, "num_islands", 0)
        if donor_policy == "ring" and num_islands > 0:
            return (source_idx + 1) % num_islands
        if donor_policy == "topk":
            ranked = sorted(
                ((idx, island_best_scores.get(idx, float("-inf"))) for idx in dest_islands),
                key=lambda item: item[1],
                reverse=True,
            )
            if ranked:
                return ranked[0][0]
        # Default random destination
        return random.choice(dest_islands)

    def _select_migrants(
        self,
        source_idx: int,
        island_size: int,
        num_migrants: int,
        island_elitism: float,
        payload_policy: Optional[str],
    ) -> List[str]:
        selection_query = (
            "SELECT id FROM programs WHERE island_idx = ? "
            "AND generation > 0 AND correct = 1"
        )
        params: List[Any] = [source_idx]

        elite_ids = self._get_elite_ids(source_idx, island_size, island_elitism)
        if elite_ids:
            placeholders = ",".join(["?"] * len(elite_ids))
            selection_query += f" AND id NOT IN ({placeholders})"
            params.extend(elite_ids)

        order_clause = self._payload_order_clause(payload_policy)
        selection_query += f" {order_clause} LIMIT ?"

        available_programs = self._count_available_programs(source_idx) - len(elite_ids)
        if available_programs <= 0:
            logger.debug(
                f"No eligible programs available for migration from island {source_idx}"
            )
            return []

        actual_migrants = max(0, min(num_migrants, available_programs))
        if actual_migrants == 0:
            return []
        params.append(actual_migrants)

        self.cursor.execute(selection_query, params)
        migrants = [row["id"] for row in self.cursor.fetchall()]
        if len(migrants) != len(set(migrants)):
            logger.warning(
                f"Duplicate programs selected for migration from island {source_idx}."
            )
            migrants = list(set(migrants))

        logger.debug(
            f"Selected {len(migrants)} migrants from island {source_idx}"
        )
        return migrants

    def _payload_order_clause(self, payload_policy: Optional[str]) -> str:
        if payload_policy == "elite":
            return "ORDER BY combined_score DESC"
        if payload_policy == "novel":
            return "ORDER BY combined_score ASC"
        return "ORDER BY RANDOM()"

    def _count_available_programs(self, island_idx: int) -> int:
        self.cursor.execute(
            "SELECT COUNT(*) FROM programs WHERE island_idx = ? AND generation > 0 AND correct = 1",
            (island_idx,),
        )
        return (self.cursor.fetchone() or [0])[0]

    def _get_elite_ids(
        self,
        island_idx: int,
        island_size: int,
        elite_ratio: float,
    ) -> List[str]:
        if elite_ratio <= 0 or island_size <= 1:
            return []
        elite_count = max(1, int(round(island_size * elite_ratio)))
        self.cursor.execute(
            "SELECT id FROM programs WHERE island_idx = ? AND generation > 0 AND correct = 1 "
            "ORDER BY combined_score DESC LIMIT ?",
            (island_idx, elite_count),
        )
        return [row["id"] for row in self.cursor.fetchall()]

    def _migrate_program(
        self,
        migrant_id: str,
        source_idx: int,
        dest_idx: int,
        current_generation: int,
    ) -> None:
        """Migrate a single program from source to destination island."""
        # Get current migration history
        self.cursor.execute(
            "SELECT migration_history FROM programs WHERE id = ?", (migrant_id,)
        )
        row = self.cursor.fetchone()
        history = (
            json.loads(row["migration_history"])
            if row and row["migration_history"]
            else []
        )

        # Add new migration event
        history.append(
            {
                "generation": current_generation,
                "from": source_idx,
                "to": dest_idx,
                "timestamp": time.time(),
            }
        )
        history_json = json.dumps(history)

        self.cursor.execute(
            """UPDATE programs
               SET island_idx = ?, migration_history = ?
               WHERE id = ?""",
            (dest_idx, history_json, migrant_id),
        )
        logger.debug(
            f"Migrated program {migrant_id[:8]}... from "
            f"island {source_idx} to {dest_idx}"
        )

    def _print_migration_summary(self, migrations_summary: Dict) -> None:
        """Print a summary table of the migration."""
        console = RichConsole()
        table = RichTable(
            title="[bold]Island Migration Summary[/bold]",
            box=rich.box.ROUNDED,
            border_style="blue",
            show_header=True,
            header_style="bold cyan",
            padding=(0, 1),
            width=120,  # Match program summary table width
        )
        table.add_column("Source", justify="center", style="cyan", width=8)
        table.add_column("Dest", justify="center", style="magenta", width=6)
        table.add_column("Program IDs", justify="left", style="green", width=15)
        table.add_column("Gen.", justify="center", style="yellow", width=10)
        table.add_column("Score", justify="right", style="yellow", width=8)
        table.add_column("Children", justify="right", style="blue", width=13)
        table.add_column(
            "Patch Name",
            justify="left",
            style="white",
            width=30,
            overflow="ellipsis",
        )
        table.add_column(
            "Type", justify="left", style="cyan", width=8, overflow="ellipsis"
        )
        table.add_column("Complexity", justify="right", style="red", width=9)

        for source, destinations in sorted(migrations_summary.items()):
            for dest, progs in sorted(destinations.items()):
                # Get detailed metrics for each program
                for prog_id in progs:
                    self.cursor.execute(
                        """SELECT combined_score as score, children_count,
                                  generation, metadata, complexity
                           FROM programs WHERE id = ?""",
                        (prog_id,),
                    )
                    result = self.cursor.fetchone()

                    if result:
                        score = result["score"]
                        children = result["children_count"] or 0
                        generation = result["generation"] or 0
                        complexity = result["complexity"] or 0
                        metadata = json.loads(result["metadata"] or "{}")

                        # Format score
                        score_str = f"{score:.3f}" if score is not None else "N/A"

                        # Get patch info from metadata
                        patch_name = metadata.get("patch_name", "N/A")
                        patch_type = metadata.get("patch_type", "N/A")

                        table.add_row(
                            f"I{source}",
                            f"I{dest}",
                            prog_id[:8] + "...",
                            f"{generation}",
                            score_str,
                            str(children),
                            (patch_name[:28] if patch_name != "N/A" else "N/A"),
                            patch_type,
                            f"{complexity:.1f}" if complexity else "N/A",
                        )
        console.print(table)


class CombinedIslandManager:
    """Combined island manager that handles all island-related operations."""

    def __init__(
        self,
        cursor: sqlite3.Cursor,
        conn: sqlite3.Connection,
        config: Any,
        assignment_strategy: Optional[IslandStrategy] = None,
        migration_strategy: Optional[IslandMigrationStrategy] = None,
    ):
        self.cursor = cursor
        self.conn = conn
        self.config = config

        self.assignment_strategy = assignment_strategy or (
            CopyInitialProgramIslandStrategy(cursor, conn, config)
        )

        self._island_params: Dict[int, IslandMigrationParams] = {}
        self._pending_islands: Set[int] = set()
        self._last_migration_generation: Dict[int, int] = {}
        self._migration_callback: Optional[Callable[[MigrationSummary], None]] = None
        self._policy_provider: Optional[Callable[[int], Optional[Dict[str, str]]]] = None

        self._initialize_island_params()

        if migration_strategy is None:
            self.migration_strategy = ElitistMigrationStrategy(
                cursor,
                conn,
                config,
                param_provider=self._get_island_params,
                policy_provider=self._resolve_policy,
            )
        else:
            self.migration_strategy = migration_strategy

    def assign_island(self, program: Any) -> None:
        """Assign an island to a program using the configured strategy."""
        self.assignment_strategy.assign_island(program)

    def perform_migration(self, current_generation: int) -> MigrationSummary:
        """Perform migration using the configured strategy."""
        eligible = sorted(self._pending_islands) if self._pending_islands else None
        summary = self.migration_strategy.perform_migration(
            current_generation, eligible_islands=eligible
        )
        if summary.total_migrated > 0:
            for island_idx in summary.per_island.keys():
                self._last_migration_generation[island_idx] = current_generation
        self._pending_islands.clear()
        if self._migration_callback:
            self._migration_callback(summary)
        return summary

    def set_island_params(
        self,
        island_idx: int,
        *,
        migration_rate: Optional[float] = None,
        migration_interval: Optional[int] = None,
        island_elitism: Optional[float] = None,
    ) -> IslandMigrationParams:
        params = self._get_island_params(island_idx)
        if migration_rate is not None:
            params.migration_rate = migration_rate
        if migration_interval is not None:
            params.migration_interval = max(2, int(migration_interval))
        if island_elitism is not None:
            params.island_elitism = max(0.0, float(island_elitism))
        return params

    def set_island_params_bulk(
        self, params: Dict[int, IslandMigrationParams]
    ) -> None:
        for idx, state in params.items():
            self._island_params[idx] = IslandMigrationParams(
                migration_rate=state.migration_rate,
                migration_interval=state.migration_interval,
                island_elitism=state.island_elitism,
            )

    def get_island_params_snapshot(self) -> Dict[int, IslandMigrationParams]:
        return {
            idx: IslandMigrationParams(
                migration_rate=state.migration_rate,
                migration_interval=state.migration_interval,
                island_elitism=state.island_elitism,
            )
            for idx, state in self._island_params.items()
        }

    def register_policy_provider(
        self, provider: Callable[[int], Optional[Dict[str, str]]]
    ) -> None:
        self._policy_provider = provider

    def set_migration_callback(
        self, callback: Callable[[MigrationSummary], None]
    ) -> None:
        self._migration_callback = callback

    def _initialize_island_params(self) -> None:
        num_islands = getattr(self.config, "num_islands", 0)
        default_params = self._default_params()
        for idx in range(num_islands):
            self._island_params[idx] = IslandMigrationParams(
                migration_rate=default_params.migration_rate,
                migration_interval=default_params.migration_interval,
                island_elitism=default_params.island_elitism,
            )
            self._last_migration_generation.setdefault(idx, 0)

    def _default_params(self) -> IslandMigrationParams:
        return IslandMigrationParams(
            migration_rate=getattr(self.config, "migration_rate", 0.1),
            migration_interval=max(2, getattr(self.config, "migration_interval", 10)),
            island_elitism=self._normalize_elitism_value(
                getattr(self.config, "island_elitism", True)
            ),
        )

    def _get_island_params(self, island_idx: int) -> IslandMigrationParams:
        if island_idx not in self._island_params:
            self._island_params[island_idx] = self._default_params()
        return self._island_params[island_idx]

    def _resolve_policy(self, island_idx: int) -> Optional[Dict[str, str]]:
        if not self._policy_provider:
            return None
        return self._policy_provider(island_idx)

    def _normalize_elitism_value(self, value: Any) -> float:
        if isinstance(value, bool):
            return 0.1 if value else 0.0
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return 0.0

    def get_island_idx(self, program_id: str) -> Optional[int]:
        """Get the island index for a given program ID."""
        self.cursor.execute(
            "SELECT island_idx FROM programs WHERE id = ?", (program_id,)
        )
        row = self.cursor.fetchone()
        return row["island_idx"] if row else None

    def get_initialized_islands(self) -> List[int]:
        """Get list of islands that have correct programs."""
        return self.assignment_strategy.get_initialized_islands()

    def are_all_islands_initialized(self) -> bool:
        """Check if all islands have at least one correct program."""
        num_islands = getattr(self.config, "num_islands", 0)
        if num_islands <= 0:
            return True
        initialized_islands = self.get_initialized_islands()
        return len(initialized_islands) >= num_islands

    def should_schedule_migration(self, program: Any) -> bool:
        """Check if migration should be scheduled for the program's island."""
        island_idx = getattr(program, "island_idx", None)
        if island_idx is None or program.generation <= 0:
            return False

        params = self._get_island_params(island_idx)
        interval = max(1, int(params.migration_interval))
        last_gen = self._last_migration_generation.get(island_idx, 0)

        if (program.generation - last_gen) >= interval:
            if island_idx not in self._pending_islands:
                self._pending_islands.add(island_idx)
            return True
        return False

    def get_island_populations(self) -> Dict[int, int]:
        """Get the population count for each island."""
        if not hasattr(self.config, "num_islands") or self.config.num_islands <= 0:
            return {}

        self.cursor.execute(
            "SELECT island_idx, COUNT(id) as count FROM programs GROUP BY island_idx"
        )
        return {
            row["island_idx"]: row["count"]
            for row in self.cursor.fetchall()
            if row["island_idx"] is not None
        }

    def get_migration_info(self) -> Optional[str]:
        """Get migration policy information as a formatted string."""
        if not (
            hasattr(self.config, "migration_interval")
            and hasattr(self.config, "migration_rate")
        ):
            return None

        migration_str = (
            f"{self.config.migration_interval}G, "
            f"{self.config.migration_rate * 100:.0f}%"
        )
        if hasattr(self.config, "island_elitism") and self.config.island_elitism:
            migration_str += "(E)"
        return migration_str

    def format_island_display(self) -> str:
        """Format island populations for display."""
        populations = self.get_island_populations()
        if not populations:
            num_islands = getattr(self.config, "num_islands", 0)
            return f"0 programs in {num_islands} islands"

        island_display = []
        for island_idx, count in sorted(populations.items()):
            island_color = f"color({30 + island_idx % 220})"
            island_display.append(
                f"[{island_color}]I{island_idx}: {count}[/{island_color}]"
            )
        return " | ".join(island_display)

    def copy_program_to_islands(self, program: Any) -> List[str]:
        """
        Copy a program to all other islands.
        Returns a list of new program IDs that were created.
        """
        num_islands = getattr(self.config, "num_islands", 0)
        if num_islands <= 1:
            return []

        created_ids = []
        # Create copies for islands 1 through num_islands-1
        # (original program is already on island 0)
        for island_idx in range(1, num_islands):
            # Create a new program ID
            new_id = str(uuid.uuid4())
            # Copy all program data but change the ID and island_idx
            copy_metadata = program.metadata.copy() if program.metadata else {}
            # Remove the flag that indicates copying is needed
            copy_metadata.pop("_needs_island_copies", None)
            # Add metadata to indicate this is a copy
            copy_metadata["_is_island_copy"] = True
            copy_metadata["_original_program_id"] = program.id
            # Serialize JSON data
            public_metrics_json = json.dumps(program.public_metrics or {})
            private_metrics_json = json.dumps(program.private_metrics or {})
            metadata_json = json.dumps(copy_metadata)
            archive_insp_ids_json = json.dumps(program.archive_inspiration_ids or [])
            top_k_insp_ids_json = json.dumps(program.top_k_inspiration_ids or [])
            embedding_json = json.dumps(program.embedding or [])
            embedding_pca_2d_json = json.dumps(program.embedding_pca_2d or [])
            embedding_pca_3d_json = json.dumps(program.embedding_pca_3d or [])
            migration_history_json = json.dumps(program.migration_history or [])
            # Insert the copy into the database
            # Handle text_feedback - convert to string if it's a list
            text_feedback_str = program.text_feedback
            if isinstance(text_feedback_str, list):
                text_feedback_str = "\n".join(text_feedback_str)
            elif text_feedback_str is None:
                text_feedback_str = ""
            self.cursor.execute(
                """
                INSERT INTO programs
                   (id, code, language, parent_id, archive_inspiration_ids,
                    top_k_inspiration_ids, generation, timestamp, code_diff,
                    combined_score, public_metrics, private_metrics,
                    text_feedback, complexity, embedding, embedding_pca_2d,
                    embedding_pca_3d, embedding_cluster_id, correct,
                    children_count, metadata, island_idx, migration_history)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                           ?, ?, ?, ?, ?, ?)
                """,
                (
                    new_id,
                    program.code,
                    program.language,
                    program.parent_id,
                    archive_insp_ids_json,
                    top_k_insp_ids_json,
                    program.generation,
                    program.timestamp,
                    program.code_diff,
                    program.combined_score,
                    public_metrics_json,
                    private_metrics_json,
                    text_feedback_str,
                    program.complexity,
                    embedding_json,
                    embedding_pca_2d_json,
                    embedding_pca_3d_json,
                    program.embedding_cluster_id,
                    program.correct,
                    program.children_count,
                    metadata_json,
                    island_idx,
                    migration_history_json,
                ),
            )
            created_ids.append(new_id)
            logger.info(
                f"Created copy {new_id[:8]}... of program {program.id[:8]}... "
                f"for island {island_idx}"
            )
        self.conn.commit()
        logger.info(
            f"Created {len(created_ids)} copies of program "
            f"{program.id[:8]}... for islands 1-{num_islands - 1}"
        )
        return created_ids

    def needs_island_copies(self, program: Any) -> bool:
        """Check if a program needs to be copied to other islands."""
        return program.metadata is not None and program.metadata.get(
            "_needs_island_copies", False
        )
