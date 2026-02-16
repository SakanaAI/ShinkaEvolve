"""
Async database operations for concurrent evolution pipeline.
Provides non-blocking database access for high-throughput proposal generation.
"""

import asyncio
import logging
import time
import threading
import traceback
from typing import List, Optional, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from .complexity import analyze_code_metrics
from .dbase import Program, ProgramDatabase

logger = logging.getLogger(__name__)


# Debugging utilities
class AsyncDBDebugger:
    """Simple debugging for async database operations."""

    def __init__(self):
        self.active_operations = {}
        self.operation_count = 0

    def track_start(self, operation: str, **kwargs):
        """Track operation start."""
        op_id = f"{operation}_{self.operation_count}_{threading.get_ident()}"
        self.operation_count += 1
        self.active_operations[op_id] = {
            "operation": operation,
            "start_time": time.time(),
            "thread_id": threading.get_ident(),
            "stack": traceback.format_stack()[-3:],  # Last 3 stack frames
            **kwargs,
        }
        logger.info(f"🔄 DB_OP_START: {op_id} on thread {threading.get_ident()}")
        return op_id

    def track_end(self, op_id: str, success: bool = True):
        """Track operation end."""
        if op_id in self.active_operations:
            op = self.active_operations[op_id]
            duration = time.time() - op["start_time"]
            status = "✅" if success else "❌"
            logger.info(f"{status} DB_OP_END: {op_id} ({duration:.2f}s)")
            del self.active_operations[op_id]

            if duration > 10:  # Warn about slow operations
                logger.warning(
                    f"⚠️  SLOW DB OPERATION: {op['operation']} took {duration:.2f}s"
                )

    def check_long_running(self):
        """Check for long-running operations."""
        current_time = time.time()
        for op_id, op in self.active_operations.items():
            duration = current_time - op["start_time"]
            if duration > 30:  # Operations longer than 30 seconds
                logger.error(
                    f"🚨 DEADLOCK SUSPECT: {op_id} running for {duration:.1f}s"
                )
                logger.error(f"   Operation: {op['operation']}")
                logger.error(f"   Thread: {op['thread_id']}")
                logger.error(
                    f"   Stack: {op['stack'][-1] if op['stack'] else 'unknown'}"
                )


# Global debugger instance
db_debugger = AsyncDBDebugger()


class AsyncProgramDatabase:
    """Async wrapper around ProgramDatabase for concurrent operations."""

    def __init__(
        self,
        sync_db: ProgramDatabase,
        max_workers: int = 1,
        embedding_recompute_interval: int = 10,
        enable_deadlock_debugging: bool = False,
    ):
        """Initialize with existing sync database and thread pool.

        Args:
            sync_db: The synchronous ProgramDatabase instance
            max_workers: Maximum number of threads for database operations
            embedding_recompute_interval: Programs to add before recomputing
            enable_deadlock_debugging: Enable detailed deadlock monitoring and logging
        """
        self.sync_db = sync_db
        # Use multiple workers for better concurrency with proper coordination
        if max_workers < 1:
            max_workers = 1
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = asyncio.Lock()
        # Semaphore to limit concurrent database operations and prevent deadlocks
        # With WAL mode enabled, we can handle more concurrent operations safely
        concurrent_ops = min(
            max_workers, 8
        )  # Up to 8 concurrent DB operations with WAL mode
        self._db_semaphore = asyncio.Semaphore(concurrent_ops)

        logger.info(
            f"🔧 AsyncDB initialized with {max_workers} workers, {concurrent_ops} concurrent DB ops (WAL mode)"
        )

        # Embedding recomputation control
        self.embedding_recompute_interval = embedding_recompute_interval
        self.programs_added_since_embedding_recompute = 0
        self.last_embedding_recompute_time = time.time()
        self._embedding_recompute_task = None

        # Conditional deadlock monitoring
        self.enable_deadlock_debugging = enable_deadlock_debugging
        self._monitoring_task = None
        if enable_deadlock_debugging:
            self._monitoring_task = asyncio.create_task(self._deadlock_monitor())
            logger.info("🔧 Async database deadlock monitoring started")
        else:
            logger.debug("Deadlock monitoring disabled")

    def _debug_track_start(self, operation: str, **kwargs):
        """Helper to conditionally track debug operations."""
        if self.enable_deadlock_debugging:
            return db_debugger.track_start(operation, **kwargs)
        return None

    def _debug_track_end(self, op_id, success: bool = True):
        """Helper to conditionally end debug tracking."""
        if self.enable_deadlock_debugging and op_id is not None:
            db_debugger.track_end(op_id, success=success)

    async def _deadlock_monitor(self):
        """Background task to monitor for deadlocks."""
        while True:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                if self.enable_deadlock_debugging:
                    db_debugger.check_long_running()

                    # Log current state
                    active_count = len(db_debugger.active_operations)
                    if active_count > 0:
                        logger.info(f"📊 Active DB operations: {active_count}")

            except asyncio.CancelledError:
                logger.info("🛑 Deadlock monitoring stopped")
                break
            except Exception as e:
                logger.error(f"Error in deadlock monitoring: {e}")

    async def sample_async(
        self,
        target_generation=None,
        novelty_attempt=None,
        max_novelty_attempts=None,
        resample_attempt=None,
        max_resample_attempts=None,
    ) -> Tuple[Program, List[Program], List[Program]]:
        """Async version of database sampling for parent and inspirations.

        Returns:
            Tuple of (parent_program, archive_inspirations, top_k_inspirations)
        """
        # Debug tracking (conditional)
        op_id = self._debug_track_start(
            "sample_async",
            target_generation=target_generation,
            novelty_attempt=novelty_attempt,
        )

        # Run in executor with thread-safe database operations
        try:
            # Use semaphore to prevent concurrent database operations
            async with self._db_semaphore:
                await asyncio.sleep(0)  # Yield control to event loop

                def sample_thread_safe():
                    thread_op_id = self._debug_track_start("sample_thread_safe")
                    thread_db = None
                    try:
                        # Create a new ProgramDatabase instance for this thread
                        from .dbase import ProgramDatabase

                        thread_db = ProgramDatabase(self.sync_db.config, read_only=True)
                        result = thread_db.sample(
                            target_generation=target_generation,
                            novelty_attempt=novelty_attempt,
                            max_novelty_attempts=max_novelty_attempts,
                            resample_attempt=resample_attempt,
                            max_resample_attempts=max_resample_attempts,
                        )
                        self._debug_track_end(thread_op_id, success=True)
                        return result
                    except Exception as e:
                        self._debug_track_end(thread_op_id, success=False)
                        logger.error(f"Error in sample_thread_safe: {e}")
                        raise
                    finally:
                        if thread_db:
                            try:
                                thread_db.close()
                            except Exception as e:
                                logger.warning(f"Error closing thread database: {e}")

                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(self.executor, sample_thread_safe)
                self._debug_track_end(op_id, success=True)
                return result
        except Exception as e:
            self._debug_track_end(op_id, success=False)
            logger.error(f"Error in async sample: {e}")
            raise

    async def sample_with_fix_mode_async(
        self,
        target_generation=None,
        novelty_attempt=None,
        max_novelty_attempts=None,
        resample_attempt=None,
        max_resample_attempts=None,
    ) -> Tuple[Program, List[Program], List[Program], bool]:
        """Async version of database sampling with fix mode detection.

        Returns:
            Tuple of (parent_program, archive_inspirations, top_k_inspirations, needs_fix)
            where needs_fix is True if no correct programs exist.
        """
        op_id = self._debug_track_start(
            "sample_with_fix_mode_async",
            target_generation=target_generation,
            novelty_attempt=novelty_attempt,
        )

        try:
            async with self._db_semaphore:
                await asyncio.sleep(0)

                def sample_with_fix_thread_safe():
                    thread_op_id = self._debug_track_start(
                        "sample_with_fix_thread_safe"
                    )
                    thread_db = None
                    try:
                        from .dbase import ProgramDatabase

                        thread_db = ProgramDatabase(self.sync_db.config, read_only=True)
                        result = thread_db.sample_with_fix_mode(
                            target_generation=target_generation,
                            novelty_attempt=novelty_attempt,
                            max_novelty_attempts=max_novelty_attempts,
                            resample_attempt=resample_attempt,
                            max_resample_attempts=max_resample_attempts,
                        )
                        self._debug_track_end(thread_op_id, success=True)
                        return result
                    except Exception as e:
                        self._debug_track_end(thread_op_id, success=False)
                        logger.error(f"Error in sample_with_fix_thread_safe: {e}")
                        raise
                    finally:
                        if thread_db:
                            try:
                                thread_db.close()
                            except Exception as e:
                                logger.warning(f"Error closing thread database: {e}")

                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.executor, sample_with_fix_thread_safe
                )
                self._debug_track_end(op_id, success=True)
                return result
        except Exception as e:
            self._debug_track_end(op_id, success=False)
            logger.error(f"Error in async sample_with_fix_mode: {e}")
            raise

    async def update_beam_search_parent_async(self, parent_id: str) -> None:
        """Update beam_search_parent_id on the database.

        This is needed because async sampling uses read-only thread-local databases,
        so beam_search state changes need to be synced back to the main database.
        """
        op_id = self._debug_track_start(
            "update_beam_search_parent_async", parent_id=parent_id
        )
        try:
            async with self._db_semaphore:
                await asyncio.sleep(0)

                def update_thread_safe():
                    # Create a new writable database instance for this thread
                    # (SQLite connections aren't thread-safe)
                    from .dbase import ProgramDatabase

                    thread_db = None
                    try:
                        thread_db = ProgramDatabase(
                            self.sync_db.config,
                            embedding_model=self.sync_db.embedding_model,
                        )
                        thread_db.beam_search_parent_id = parent_id
                        thread_db._update_metadata_in_db(
                            "beam_search_parent_id", parent_id
                        )
                        # Also update the in-memory state on sync_db for consistency
                        self.sync_db.beam_search_parent_id = parent_id
                    finally:
                        if thread_db:
                            try:
                                thread_db.close()
                            except Exception:
                                pass

                loop = asyncio.get_event_loop()
                await loop.run_in_executor(self.executor, update_thread_safe)
                self._debug_track_end(op_id, success=True)
        except Exception as e:
            self._debug_track_end(op_id, success=False)
            logger.warning(f"Could not update beam_search_parent_id: {e}")

    async def add_program_async(
        self,
        program: Program,
        parent_id: Optional[str] = None,
        archive_insp_ids: Optional[List[str]] = None,
        top_k_insp_ids: Optional[List[str]] = None,
        code_diff: Optional[str] = None,
        meta_patch_data: Optional[Dict[str, Any]] = None,
        code_embedding: Optional[List[float]] = None,
        embed_cost: float = 0.0,
    ) -> None:
        """Async version of adding a program to the database.

        Args:
            program: Program to add
            parent_id: ID of parent program
            archive_insp_ids: List of archive inspiration IDs
            top_k_insp_ids: List of top-k inspiration IDs
            code_diff: Code diff from parent
            meta_patch_data: Metadata about patch generation
            code_embedding: Code embedding vector
            embed_cost: Cost of embedding generation
        """
        # Debug tracking
        op_id = self._debug_track_start(
            "add_program_async",
            program_id=program.id,
            generation=program.generation,
            parent_id=parent_id,
        )

        try:
            # Prepare program data outside the lock to reduce lock time
            await asyncio.sleep(0)  # Yield control to event loop

            # Asynchronously calculate complexity if not provided
            if program.complexity == 0.0:
                try:
                    loop = asyncio.get_event_loop()
                    # Get language from program, default to python
                    language = getattr(program, "language", "python")
                    code_metrics = await loop.run_in_executor(
                        self.executor,
                        analyze_code_metrics,
                        program.code,
                        language,
                    )
                    program.complexity = code_metrics.get("complexity_score", 0.0)
                    if program.metadata is None:
                        program.metadata = {}
                    program.metadata["code_analysis_metrics"] = code_metrics
                except Exception as e:
                    logger.warning(
                        f"Could not calculate complexity for program {program.id}: {e}"
                    )
                    # Fallback to length
                    program.complexity = float(len(program.code))

            # Set additional metadata using setattr for dynamic attributes
            if parent_id:
                setattr(program, "parent_id", parent_id)
            if archive_insp_ids:
                setattr(program, "archive_inspiration_ids", archive_insp_ids)
            if top_k_insp_ids:
                setattr(program, "top_k_inspiration_ids", top_k_insp_ids)
            if code_diff:
                setattr(program, "code_diff", code_diff)
            if meta_patch_data:
                setattr(program, "meta_patch_data", meta_patch_data)
            if code_embedding:
                setattr(program, "code_embedding", code_embedding)
            setattr(program, "embed_cost", embed_cost)

            # Use semaphore to prevent concurrent database operations that can deadlock
            async with self._db_semaphore:
                await self._add_program_fast_async(program)

                # Track programs and schedule embedding recomputation (inside semaphore)
                async with self._lock:
                    self.programs_added_since_embedding_recompute += 1
                    should_recompute = (
                        self.programs_added_since_embedding_recompute
                        >= self.embedding_recompute_interval
                    )

            # Schedule embedding recomputation outside lock
            if should_recompute:
                self._schedule_embedding_recomputation()

            self._debug_track_end(op_id, success=True)

        except Exception as e:
            self._debug_track_end(op_id, success=False)
            logger.error(f"Error in async add_program: {e}")
            raise

    async def add_programs_batch_async(
        self,
        programs_data: List[
            Tuple[
                Program,
                Optional[str],
                Optional[List[str]],
                Optional[List[str]],
                Optional[str],
                Optional[Dict[str, Any]],
                Optional[List[float]],
                float,
            ]
        ],
    ) -> None:
        """Add multiple programs in a batch for improved performance.

        Args:
            programs_data: List of tuples containing program data
        """
        if not programs_data:
            return

        # Debug tracking
        op_id = self._debug_track_start(
            "add_programs_batch_async",
            batch_size=len(programs_data),
            program_ids=[p[0].id for p in programs_data[:3]],  # First 3 IDs for context
        )

        try:
            # Prepare all programs outside the lock
            await asyncio.sleep(0)  # Yield control to event loop

            prepared_programs = []
            for program_data in programs_data:
                (
                    program,
                    parent_id,
                    archive_insp_ids,
                    top_k_insp_ids,
                    code_diff,
                    meta_patch_data,
                    code_embedding,
                    embed_cost,
                ) = program_data

                # Set additional metadata using setattr for dynamic attributes
                if parent_id:
                    setattr(program, "parent_id", parent_id)
                if archive_insp_ids:
                    setattr(program, "archive_inspiration_ids", archive_insp_ids)
                if top_k_insp_ids:
                    setattr(program, "top_k_inspiration_ids", top_k_insp_ids)
                if code_diff:
                    setattr(program, "code_diff", code_diff)
                if meta_patch_data:
                    setattr(program, "meta_patch_data", meta_patch_data)
                if code_embedding:
                    setattr(program, "code_embedding", code_embedding)
                setattr(program, "embed_cost", embed_cost)

                prepared_programs.append(program)

            # Use lock only for the actual database writes
            async with self._lock:
                for program in prepared_programs:
                    await self._add_program_fast_async(program)

                # Track programs and schedule embedding recomputation
                self.programs_added_since_embedding_recompute += len(prepared_programs)
                should_recompute = (
                    self.programs_added_since_embedding_recompute
                    >= self.embedding_recompute_interval
                )

            # Schedule embedding recomputation outside lock
            if should_recompute:
                self._schedule_embedding_recomputation()

            logger.info(
                f"Successfully added batch of {len(prepared_programs)} programs"
            )

            self._debug_track_end(op_id, success=True)

        except Exception as e:
            self._debug_track_end(op_id, success=False)
            logger.error(f"Error in batch add_programs: {e}")
            raise

    def _add_program_fast(self, program: Program):
        """Fast program addition that defers expensive operations (deprecated - use async version)."""
        # Temporarily disable expensive operations in sync database
        original_embedding_method = self.sync_db._recompute_embeddings_and_clusters
        self.sync_db._recompute_embeddings_and_clusters = lambda: None

        try:
            # Add program without expensive operations
            self.sync_db.add(program, verbose=True)
        finally:
            # Restore original methods
            self.sync_db._recompute_embeddings_and_clusters = original_embedding_method

    async def _add_program_fast_async(self, program: Program):
        """Async fast program addition that defers expensive operations."""

        def add_program_sync():
            # Create a new database instance for this thread with full functionality
            from .dbase import ProgramDatabase

            thread_db = None
            try:
                thread_db = ProgramDatabase(
                    self.sync_db.config,
                    embedding_model=self.sync_db.embedding_model,
                )

                # Temporarily disable expensive operations
                original_embedding_method = thread_db._recompute_embeddings_and_clusters
                thread_db._recompute_embeddings_and_clusters = lambda: None

                try:
                    # Use the full database add method which includes island assignment
                    thread_db.add(program, verbose=True)
                finally:
                    # Restore original methods
                    thread_db._recompute_embeddings_and_clusters = (
                        original_embedding_method
                    )

            except Exception as e:
                logger.error(f"Error in add_program_sync: {e}")
                raise
            finally:
                if thread_db:
                    try:
                        thread_db.close()
                    except Exception as e:
                        logger.warning(
                            f"Error closing thread database in add_program_sync: {e}"
                        )

        # Run the thread-safe database operation in an executor
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.executor, add_program_sync)

    def _schedule_embedding_recomputation(self):
        """Schedule embedding recomputation as a background task."""
        # Cancel any existing recomputation task
        if self._embedding_recompute_task and not self._embedding_recompute_task.done():
            self._embedding_recompute_task.cancel()

        # Schedule new recomputation task
        self._embedding_recompute_task = asyncio.create_task(
            self._recompute_embeddings_background()
        )
        self.programs_added_since_embedding_recompute = 0
        logger.info(
            f"Scheduled embedding recomputation after "
            f"{self.embedding_recompute_interval} program additions"
        )

    async def _recompute_embeddings_background(self):
        """Run embedding recomputation in the background without blocking."""
        try:
            # Small delay to avoid immediate blocking
            await asyncio.sleep(0.1)

            # Run thread-safe embedding recomputation on executor
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                self.sync_db._recompute_embeddings_and_clusters_thread_safe,
            )

            self.last_embedding_recompute_time = time.time()
            logger.info("Background embedding recomputation completed")

        except asyncio.CancelledError:
            logger.info("Embedding recomputation task was cancelled")
        except Exception as e:
            logger.error(f"Error in background embedding recomputation: {e}")

    async def get_async(self, program_id: str) -> Optional[Program]:
        """Async version of get program by ID."""
        # Debug tracking
        op_id = self._debug_track_start("get_async", program_id=program_id)

        try:
            await asyncio.sleep(0)  # Yield control to event loop

            def get_thread_safe():
                thread_op_id = self._debug_track_start("get_thread_safe")
                try:
                    from .dbase import ProgramDatabase

                    thread_db = ProgramDatabase(self.sync_db.config, read_only=True)
                    try:
                        result = thread_db.get(program_id)
                        self._debug_track_end(thread_op_id, success=True)
                        return result
                    finally:
                        thread_db.close()
                except Exception as e:
                    self._debug_track_end(thread_op_id, success=False)
                    raise

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self.executor, get_thread_safe)
            self._debug_track_end(op_id, success=True)
            return result
        except Exception as e:
            self._debug_track_end(op_id, success=False)
            logger.error(f"Error in async get: {e}")
            raise

    async def get_best_program_async(self) -> Optional[Program]:
        """Async version of get best program."""
        # Debug tracking
        op_id = self._debug_track_start("get_best_program_async")

        try:
            await asyncio.sleep(0)  # Yield control to event loop

            def get_best_thread_safe():
                thread_op_id = self._debug_track_start("get_best_thread_safe")
                try:
                    from .dbase import ProgramDatabase

                    thread_db = ProgramDatabase(self.sync_db.config, read_only=True)
                    try:
                        result = thread_db.get_best_program()
                        self._debug_track_end(thread_op_id, success=True)
                        return result
                    finally:
                        thread_db.close()
                except Exception as e:
                    self._debug_track_end(thread_op_id, success=False)
                    raise

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self.executor, get_best_thread_safe)
            self._debug_track_end(op_id, success=True)
            return result
        except Exception as e:
            self._debug_track_end(op_id, success=False)
            logger.error(f"Error in async get_best_program: {e}")
            raise

    async def batch_sample_async(
        self, num_samples: int
    ) -> List[Tuple[Program, List[Program], List[Program]]]:
        """Sample multiple parent/inspiration sets concurrently.

        Args:
            num_samples: Number of samples to generate

        Returns:
            List of (parent, archive_inspirations, top_k_inspirations) tuples
        """
        # Debug tracking
        op_id = self._debug_track_start("batch_sample_async", num_samples=num_samples)

        try:
            # Create tasks for concurrent sampling
            tasks = [self.sample_async() for _ in range(num_samples)]

            # Execute all samples concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out exceptions and return valid results
            valid_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"Sample {i} failed: {result}")
                else:
                    valid_results.append(result)

            self._debug_track_end(op_id, success=True)
            return valid_results

        except Exception as e:
            self._debug_track_end(op_id, success=False)
            logger.error(f"Error in batch_sample_async: {e}")
            raise

    async def close_async(self):
        """Close the async database wrapper."""
        logger.info("🔧 Closing async database with monitoring...")

        # Cancel monitoring task if it exists
        if (
            hasattr(self, "_monitoring_task")
            and self._monitoring_task is not None
            and not self._monitoring_task.done()
        ):
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        # Cancel any pending embedding recomputation
        if self._embedding_recompute_task and not self._embedding_recompute_task.done():
            self._embedding_recompute_task.cancel()
            try:
                await self._embedding_recompute_task
            except asyncio.CancelledError:
                pass

        # Print final debug report (only if debugging enabled)
        if self.enable_deadlock_debugging:
            active_ops = len(db_debugger.active_operations)
            if active_ops > 0:
                logger.warning(f"🚨 Closing with {active_ops} active operations!")
                for op_id, op in db_debugger.active_operations.items():
                    duration = time.time() - op["start_time"]
                    logger.warning(f"  {op_id}: {op['operation']} ({duration:.1f}s)")

        # Shutdown the thread pool executor
        self.executor.shutdown(wait=True)
        logger.info("Async database closed")

    async def force_recompute_embeddings_async(self):
        """
        Force an immediate recomputation of embeddings and clusters, and wait for
        it to complete. Useful at the end of a run to ensure data is up to date.
        """
        # Debug tracking
        op_id = self._debug_track_start("force_recompute_embeddings_async")

        try:
            logger.info("Forcing final embedding and cluster recomputation...")
            # Schedule the task
            self._schedule_embedding_recomputation()

            # Wait for the scheduled task to complete
            if self._embedding_recompute_task:
                try:
                    await self._embedding_recompute_task
                except asyncio.CancelledError:
                    # This is okay if a new task was scheduled, but we should wait for the new one
                    pass
            logger.info("Final embedding and cluster recomputation complete.")
            self._debug_track_end(op_id, success=True)

        except Exception as e:
            self._debug_track_end(op_id, success=False)
            logger.error(f"Error in force_recompute_embeddings_async: {e}")
            raise

    async def get_programs_by_generation_async(self, generation: int) -> List[Program]:
        """Async get programs by generation."""
        # Debug tracking
        op_id = self._debug_track_start(
            "get_programs_by_generation_async", generation=generation
        )

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self.sync_db.get_programs_by_generation_thread_safe,
                generation,
            )
            self._debug_track_end(op_id, success=True)
            return result
        except Exception as e:
            self._debug_track_end(op_id, success=False)
            logger.error(f"Error in async get_programs_by_generation: {e}")
            return []  # Return empty list on error

    async def get_total_program_count_async(self) -> int:
        """Async get total program count - much faster than checking each generation."""
        op_id = self._debug_track_start("get_total_program_count_async")

        try:
            loop = asyncio.get_event_loop()

            def count_programs_thread_safe():
                """Thread-safe program counting."""
                thread_db = None
                try:
                    from .dbase import ProgramDatabase

                    thread_db = ProgramDatabase(self.sync_db.config, read_only=True)
                    thread_db.cursor.execute("SELECT COUNT(*) FROM programs")
                    count = thread_db.cursor.fetchone()[0]
                    return count
                finally:
                    if thread_db:
                        try:
                            thread_db.close()
                        except Exception as close_e:
                            logger.warning(f"Error closing thread database: {close_e}")

            result = await loop.run_in_executor(
                self.executor, count_programs_thread_safe
            )
            self._debug_track_end(op_id, success=True)
            return result
        except Exception as e:
            self._debug_track_end(op_id, success=False)
            logger.error(f"Error in get_total_program_count_async: {e}")
            return 0  # Return 0 on error

    async def get_top_programs_async(
        self, n: int = 10, correct_only: bool = True
    ) -> List[Program]:
        """Async get top programs."""
        # Debug tracking
        op_id = self._debug_track_start(
            "get_top_programs_async", n=n, correct_only=correct_only
        )

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self.sync_db.get_top_programs_thread_safe,
                n,
                correct_only,
            )
            self._debug_track_end(op_id, success=True)
            return result
        except Exception as e:
            self._debug_track_end(op_id, success=False)
            logger.error(f"Error in async get_top_programs: {e}")
            return []

    async def compute_percentile_async(
        self, score: float, correct_only: bool = True
    ) -> float:
        """Compute the percentile rank of a score among all correct programs.

        This is used for percentile-based fitness calculation for system prompts.
        A percentile of 0.8 means the score beats 80% of all correct programs.

        Args:
            score: The score to compute percentile for
            correct_only: If True, only consider correct programs

        Returns:
            Percentile rank (0-1), where 1 means best
        """
        op_id = self._debug_track_start(
            "compute_percentile_async", score=score, correct_only=correct_only
        )

        try:
            loop = asyncio.get_event_loop()

            def compute_percentile_thread_safe():
                """Thread-safe percentile computation."""
                thread_db = None
                try:
                    from .dbase import ProgramDatabase

                    thread_db = ProgramDatabase(self.sync_db.config, read_only=True)

                    # Get all scores from correct programs
                    if correct_only:
                        thread_db.cursor.execute(
                            "SELECT combined_score FROM programs "
                            "WHERE correct = 1 AND combined_score IS NOT NULL"
                        )
                    else:
                        thread_db.cursor.execute(
                            "SELECT combined_score FROM programs "
                            "WHERE combined_score IS NOT NULL"
                        )

                    rows = thread_db.cursor.fetchall()
                    if not rows:
                        return 0.5  # No programs yet, neutral percentile

                    all_scores = [row[0] for row in rows]

                    # Compute percentile: fraction of programs this score beats
                    beats = sum(1 for s in all_scores if score > s)
                    ties = sum(1 for s in all_scores if score == s)

                    # Use average rank for ties (standard percentile calculation)
                    percentile = (beats + 0.5 * ties) / len(all_scores)
                    return percentile

                finally:
                    if thread_db:
                        try:
                            thread_db.close()
                        except Exception as close_e:
                            logger.warning(f"Error closing thread database: {close_e}")

            result = await loop.run_in_executor(
                self.executor, compute_percentile_thread_safe
            )
            self._debug_track_end(op_id, success=True)
            return result

        except Exception as e:
            self._debug_track_end(op_id, success=False)
            logger.error(f"Error in compute_percentile_async: {e}")
            return 0.5  # Return neutral percentile on error

    # Delegate other methods to sync database
    def __getattr__(self, name):
        """Delegate unknown methods to sync database."""
        return getattr(self.sync_db, name)
