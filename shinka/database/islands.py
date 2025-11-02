import json
import logging
import random
import sqlite3
import time
import uuid
from abc import ABC, abstractmethod
from typing import Optional, Any, Dict, List
from collections import defaultdict
import rich.box  # type: ignore
import rich  # type: ignore
from rich.console import Console as RichConsole  # type: ignore
from rich.table import Table as RichTable  # type: ignore

logger = logging.getLogger(__name__)


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
    def perform_migration(self, current_generation: int) -> bool:
        """Perform migration between islands.
        Returns True if migration occurred."""
        pass


class ElitistMigrationStrategy(IslandMigrationStrategy):
    """Migration strategy that protects elite programs from migration."""

    def perform_migration(self, current_generation: int) -> bool:
        """
        Implements island migration by moving a subset of programs between
        islands. Called periodically based on migration_interval.
        """
        num_islands = getattr(self.config, "num_islands", 0)
        migration_rate = getattr(self.config, "migration_rate", 0.1)
        island_elitism = getattr(self.config, "island_elitism", True)

        if num_islands < 2 or migration_rate <= 0:
            return False  # No migration needed

        logger.info(f"Performing island migration at generation {current_generation}")

        migrations_summary = defaultdict(lambda: defaultdict(list))
        # Track all programs selected for migration
        all_migrated_programs = set()

        # For each island, select migrants to move
        for source_idx in range(num_islands):
            # Count programs in this island
            self.cursor.execute(
                "SELECT COUNT(*) FROM programs WHERE island_idx = ?",
                (source_idx,),
            )
            island_size = (self.cursor.fetchone() or [0])[0]

            if island_size <= 1:
                continue  # Skip tiny islands

            # Number of programs to migrate
            num_migrants = max(1, int(island_size * migration_rate))

            # Select destination islands (all except source)
            dest_islands = [i for i in range(num_islands) if i != source_idx]
            if not dest_islands:
                continue

            # Select migrants based on elitism setting
            migrants = self._select_migrants(source_idx, num_migrants, island_elitism)

            # Filter out any programs already selected for migration
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

            # Move each unique migrant to a new island
            for migrant_id in unique_migrants:
                dest_idx = random.choice(dest_islands)
                self._migrate_program(
                    migrant_id, source_idx, dest_idx, current_generation
                )
                migrations_summary[source_idx][dest_idx].append(migrant_id)

        self.conn.commit()

        if migrations_summary:
            self._print_migration_summary(migrations_summary)

        total_migrated = sum(
            len(progs)
            for dest_dict in migrations_summary.values()
            for progs in dest_dict.values()
        )
        logger.info(f"Migration complete. Migrated {total_migrated} programs.")
        return total_migrated > 0

    def _select_migrants(
        self,
        source_idx: int,
        num_migrants: int,
        island_elitism: bool,
    ) -> List[str]:
        """Select which programs to migrate from an island.
        Excludes generation 0 programs (initial programs and their copies)
        and only considers correct programs.
        """
        # Base query excludes generation 0 programs and only includes
        # correct programs
        selection_query = """
            SELECT id FROM programs
            WHERE island_idx = ? AND generation > 0 AND correct = 1
        """

        if island_elitism:
            # Get IDs of best program to protect from migration
            # Also exclude generation 0 programs from elite selection and
            # only consider correct programs
            elite_query = """
                SELECT id FROM programs
                WHERE island_idx = ? AND generation > 0 AND correct = 1
                ORDER BY combined_score DESC
                LIMIT 1
            """

            self.cursor.execute(elite_query, (source_idx,))
            elite_ids = [row["id"] for row in self.cursor.fetchall()]

            if elite_ids:
                # Exclude elites from migration
                placeholders = ",".join(["?"] * len(elite_ids))
                selection_query += f" AND id NOT IN ({placeholders})"
                selection_query += " ORDER BY RANDOM() LIMIT ?"
                params = [source_idx] + elite_ids + [num_migrants]
            else:
                selection_query += " ORDER BY RANDOM() LIMIT ?"
                params = [source_idx, num_migrants]
        else:
            # Simple random selection (excluding generation 0,
            # only correct programs)
            selection_query += " ORDER BY RANDOM() LIMIT ?"
            params = [source_idx, num_migrants]

        # First check how many correct non-generation-0 programs are available
        self.cursor.execute(
            "SELECT COUNT(*) FROM programs WHERE island_idx = ? AND "
            "generation > 0 AND correct = 1",
            (source_idx,),
        )
        available_programs = (self.cursor.fetchone() or [0])[0]

        if available_programs == 0:
            logger.debug(
                f"No correct generation > 0 programs available for migration "
                f"from island {source_idx} (generation 0 programs are "
                f"protected, "
                f"only correct programs migrate)"
            )
            return []

        # Adjust num_migrants if there aren't enough eligible programs
        actual_migrants = min(num_migrants, available_programs)
        if actual_migrants != num_migrants:
            logger.debug(
                f"Reducing migration count from {num_migrants} to "
                f"{actual_migrants} for island {source_idx} "
                f"(only {available_programs} correct eligible programs "
                f"available)"
            )
            # Update the params list to use the adjusted count
            if isinstance(params, list) and len(params) > 0:
                params[-1] = actual_migrants  # Last param is always the LIMIT

        # Select migrants
        self.cursor.execute(selection_query, params)
        migrants = [row["id"] for row in self.cursor.fetchall()]

        # Validate uniqueness (should always be true, but good to check)
        if len(migrants) != len(set(migrants)):
            logger.warning(
                f"Duplicate programs selected for migration from island "
                f"{source_idx}. Expected {len(migrants)} unique, got "
                f"{len(set(migrants))} unique."
            )
            migrants = list(set(migrants))  # Remove duplicates

        logger.debug(
            f"Selected {len(migrants)} unique correct migrants from island "
            f"{source_idx} (excluded generation 0 programs and incorrect "
            f"programs from migration)"
        )

        return migrants

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
        self.migration_strategy = migration_strategy or (
            ElitistMigrationStrategy(cursor, conn, config)
        )

    def assign_island(self, program: Any) -> None:
        """Assign an island to a program using the configured strategy."""
        self.assignment_strategy.assign_island(program)

    def perform_migration(self, current_generation: int) -> bool:
        """Perform migration using the configured strategy."""
        return self.migration_strategy.perform_migration(current_generation)

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
        """Check if migration should be scheduled based on program
        generation."""
        return (
            program.generation > 0
            and hasattr(self.config, "migration_interval")
            and self.config.migration_interval > 0
            and (program.generation % self.config.migration_interval == 0)
        )

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

            # Add the copied program to the archive if it's correct
            # This ensures it can be used as inspiration for that island
            if program.correct:
                self.cursor.execute(
                    "INSERT OR IGNORE INTO archive (program_id) VALUES (?)",
                    (new_id,),
                )
                logger.debug(f"Added copy {new_id[:8]}... to archive (correct program)")

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
