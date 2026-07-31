#!/usr/bin/env python3
"""
Shinka Visualization Module

This module provides visualization capabilities for Shinka evolution results.
It serves a web interface for exploring evolution databases and meta files.
"""

import argparse
import base64
import contextlib
import errno
import http.server
import ipaddress
import json
import markdown
import os
import re
import socket
import socketserver
import sqlite3
import stat
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import webbrowser
import weakref
from pathlib import Path, PureWindowsPath
from typing import Optional, Dict, Any, Tuple, Callable, Literal

from shinka.database import DatabaseConfig, ProgramDatabase
from shinka.database import SystemPromptConfig, SystemPromptDatabase

# We'll use a simple text-to-PDF approach instead of complex dependencies
WEASYPRINT_AVAILABLE = False

DEFAULT_PORT = 8000
CACHE_EXPIRATION_SECONDS = 5  # Cache data for 5 seconds
db_cache: Dict[
    Tuple[str, str],
    Tuple[Tuple[int, int, int, int, int], float, Any],
] = {}
db_cache_lock = threading.Lock()


class PathValidationError(ValueError):
    """Raised when a user-supplied path escapes the served search root."""


class DatabaseViewRaceError(sqlite3.OperationalError):
    """Raised when SQLite sidecars rotate while a stable view is being built."""


class _DatabaseMainSnapshot:
    def __init__(
        self,
        version: Tuple[int, int, int],
        directory: tempfile.TemporaryDirectory,
        path: str,
    ):
        self.version = version
        self.directory = directory
        self.path = path
        self.lock = threading.Lock()
        self.leases = 0
        self.evicted = False


class DatabaseRequestHandler(http.server.SimpleHTTPRequestHandler):
    _database_main_cache_lock = threading.Lock()
    _database_main_cache: Dict[Tuple[int, int], _DatabaseMainSnapshot] = {}
    _database_main_cache_build_locks: Dict[Tuple[int, int], threading.Lock] = {}
    _database_main_cache_limit = 2
    _database_main_cache_max_bytes = 2 * 1024**3

    def __init__(
        self,
        *args,
        search_root=None,
        canonical_search_root=None,
        search_root_descriptor=None,
        allowed_hosts=...,
        **kwargs,
    ):
        self.search_root = search_root or os.getcwd()
        self._canonical_search_root = canonical_search_root or os.path.realpath(
            self.search_root
        )
        self._search_root_descriptor = search_root_descriptor
        # Attributes must be set before super().__init__, which handles the
        # request immediately. `...` means "keep the class default allowlist".
        if allowed_hosts is not ...:
            self.allowed_hosts = allowed_hosts
        super().__init__(*args, **kwargs)

    def end_headers(self):
        """Disable browser caching for local HTML shells to avoid stale embedded JS."""
        # Prevent MIME sniffing on every response (DB-sourced content is served).
        self.send_header("X-Content-Type-Options", "nosniff")
        parsed_url = urllib.parse.urlparse(self.path)
        if parsed_url.path in ("/", "/index.html", "/viz_tree.html", "/compare.html"):
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, format, *args):
        """Override to provide more detailed logging."""
        print(f"\n[SERVER] {format % args}")

    def _make_failed_node_id(self, generation: int) -> str:
        return f"failed:proposal:{generation}"

    def _parse_failed_node_generation(self, node_id: str) -> Optional[int]:
        prefix = "failed:proposal:"
        if not node_id.startswith(prefix):
            return None
        try:
            return int(node_id[len(prefix) :])
        except ValueError:
            return None

    def _read_failure_json(
        self, failure_json_path: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        if not failure_json_path:
            return None
        try:
            failure_path = Path(self._resolve_within_root(failure_json_path))
            if not failure_path.exists():
                return None
            return json.loads(self._read_text_within_root(failure_path))
        except PathValidationError:
            raise
        except Exception:
            return None

    def _language_from_suffix(self, suffix: str) -> str:
        ext = suffix.lstrip(".").lower()
        return {
            "py": "python",
            "js": "javascript",
            "ts": "typescript",
            "cpp": "cpp",
            "cc": "cpp",
            "cxx": "cpp",
            "cu": "cuda",
            "go": "go",
            "sv": "verilog",
            "f90": "fortran",
            "f95": "fortran",
            "f03": "fortran",
            "f08": "fortran",
        }.get(ext, ext or "python")

    def _resolve_failed_node_language(
        self,
        details: Dict[str, Any],
        failure_payload: Optional[Dict[str, Any]],
    ) -> str:
        for source in (failure_payload or {}, details):
            language = source.get("language")
            if language:
                return str(language)

        generated_code_path = ((failure_payload or {}).get("artifacts", {}) or {}).get(
            "generated_code_path"
        )
        if generated_code_path:
            return self._language_from_suffix(Path(generated_code_path).suffix)

        failure_json_path = details.get("failure_json_path")
        if failure_json_path:
            failure_path = Path(self._resolve_within_root(failure_json_path))
            candidates = self._main_candidate_names(failure_path.parent)
            if candidates:
                return self._language_from_suffix(Path(candidates[0]).suffix)

        return "python"

    def _resolve_failed_node_code_path(
        self,
        details: Dict[str, Any],
        failure_payload: Optional[Dict[str, Any]],
    ) -> Optional[Path]:
        generated_code_path = ((failure_payload or {}).get("artifacts", {}) or {}).get(
            "generated_code_path"
        )
        if generated_code_path:
            code_path = self._existing_path_within_root(generated_code_path)
            if code_path is not None:
                return code_path

        failure_json_path = details.get("failure_json_path")
        if not failure_json_path:
            return None

        failure_path = Path(self._resolve_within_root(failure_json_path))
        language = self._resolve_failed_node_language(details, failure_payload)
        preferred_suffix = {
            "python": ".py",
            "javascript": ".js",
            "typescript": ".ts",
            "cpp": ".cpp",
            "cuda": ".cu",
            "go": ".go",
            "verilog": ".sv",
            "fortran": ".f90",
        }.get(language)
        if preferred_suffix:
            preferred_path = failure_path.parent / f"main{preferred_suffix}"
            resolved_path = self._existing_path_within_root(preferred_path)
            if resolved_path is not None:
                return resolved_path

        for candidate_name in self._main_candidate_names(failure_path.parent):
            candidate = failure_path.parent / candidate_name
            resolved_path = self._existing_path_within_root(candidate)
            if resolved_path is not None:
                return resolved_path
        return None

    def _existing_path_within_root(
        self, path: os.PathLike[str] | str
    ) -> Optional[Path]:
        resolved_path = Path(self._resolve_within_root(os.fspath(path)))
        return resolved_path if resolved_path.exists() else None

    def _main_candidate_names(self, directory: Path) -> list[str]:
        try:
            names = self._list_directory_within_root(directory)
        except FileNotFoundError:
            return []
        return sorted(name for name in names if name.startswith("main."))

    def _build_failed_node_dict(
        self,
        *,
        generation: int,
        created_at: float,
        details: Dict[str, Any],
        include_code: bool = False,
    ) -> Dict[str, Any]:
        failure_json_path = details.get("failure_json_path")
        failure_payload = self._read_failure_json(failure_json_path)
        metadata = dict(details)
        if failure_payload:
            for key in [
                "failure_json_path",
                "language",
                "generated_code_available",
                "downstream_eval_submitted",
                "artifacts",
                "attempts",
                "api_costs",
                "embed_cost",
                "novelty_cost",
                "novelty_explanation",
                "max_similarity",
            ]:
                if key in failure_payload:
                    metadata[key] = failure_payload[key]

        language = self._resolve_failed_node_language(details, failure_payload)
        code = None
        if include_code and failure_payload:
            code_path = self._resolve_failed_node_code_path(details, failure_payload)
            if code_path is not None:
                try:
                    code = self._read_text_within_root(code_path)
                except PathValidationError:
                    raise
                except Exception:
                    code = None

        return {
            "id": self._make_failed_node_id(generation),
            "code": code,
            "language": language,
            "parent_id": details.get("parent_id"),
            "archive_inspiration_ids": details.get("archive_inspiration_ids") or [],
            "top_k_inspiration_ids": details.get("top_k_inspiration_ids") or [],
            "island_idx": None,
            "generation": generation,
            "timestamp": created_at,
            "code_diff": None,
            "combined_score": 0.0,
            "public_metrics": {},
            "private_metrics": {},
            "text_feedback": details.get("failure_reason", ""),
            "correct": False,
            "children_count": 0,
            "complexity": 0.0,
            "embedding": [],
            "embedding_pca_2d": [],
            "embedding_pca_3d": [],
            "embedding_cluster_id": None,
            "migration_history": [],
            "metadata": metadata,
            "in_archive": False,
            "system_prompt_id": metadata.get("system_prompt_id"),
        }

    def _load_failed_proposal_nodes(
        self,
        abs_db_path: str,
        *,
        include_code: bool = False,
        generation: Optional[int] = None,
    ) -> list[Dict[str, Any]]:
        with self._connect_database_within_root(
            abs_db_path, timeout=5.0, isolation_level=None
        ) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("PRAGMA busy_timeout = 5000;")
            query = """
                SELECT generation, details, created_at
                FROM attempt_log
                WHERE status = 'failed'
                  AND json_valid(details)
                  AND json_extract(details, '$.node_kind') = 'failed_proposal'
            """
            params: list[Any] = []
            if generation is not None:
                query += " AND generation = ?"
                params.append(generation)
            query += " ORDER BY generation ASC, created_at DESC, id DESC"
            cursor.execute(query, params)

            selected: Dict[int, Dict[str, Any]] = {}
            for row in cursor.fetchall():
                gen = int(row["generation"])
                if gen in selected:
                    continue
                try:
                    details = json.loads(row["details"])
                except json.JSONDecodeError:
                    continue
                selected[gen] = self._build_failed_node_dict(
                    generation=gen,
                    created_at=float(row["created_at"]),
                    details=details,
                    include_code=include_code,
                )
            return [selected[g] for g in sorted(selected)]

    # Host header values allowed when the server is bound to loopback. Overwritten
    # per-instance via the handler factory when an explicit external host is used.
    allowed_hosts = frozenset({"localhost", "127.0.0.1", "::1", "[::1]"})

    def _host_allowed(self) -> bool:
        """Reject Host headers outside the allowlist (DNS-rebinding defense)."""
        if self.allowed_hosts is None:
            return True  # Explicit external bind: operator opted out of the check.
        hostname = _hostname_from_host_header(self.headers.get("Host", ""))
        return hostname in self.allowed_hosts if hostname is not None else False

    def do_GET(self):
        if not self._host_allowed():
            self.send_error(403, "Host not allowed")
            return None
        try:
            return self._dispatch_get()
        except PathValidationError as exc:
            print(f"[SERVER] Rejected path-traversal attempt: {exc}")
            self.send_error(403, "Access denied")
            return None

    def list_directory(self, path):
        """Never serve auto-generated directory listings."""
        self.send_error(403, "Directory listing is disabled")
        return None

    def _dispatch_get(self):
        print(f"\n[SERVER] Received GET request for: {self.path}")
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        query = urllib.parse.parse_qs(parsed_url.query)

        if path == "/list_databases":
            return self.handle_list_databases()

        if path == "/get_programs" and "db_path" in query:
            db_path = query["db_path"][0]
            return self.handle_get_programs(db_path)

        if path == "/get_programs_summary" and "db_path" in query:
            db_path = query["db_path"][0]
            return self.handle_get_programs_summary(db_path)

        if path == "/get_program_count" and "db_path" in query:
            db_path = query["db_path"][0]
            return self.handle_get_program_count(db_path)

        if path == "/get_program_details" and "db_path" in query and "id" in query:
            db_path = query["db_path"][0]
            program_id = query["id"][0]
            return self.handle_get_program_details(db_path, program_id)

        if path == "/get_meta_files" and "db_path" in query:
            db_path = query["db_path"][0]
            return self.handle_get_meta_files(db_path)

        if (
            path == "/get_meta_content"
            and "db_path" in query
            and ("processed_count" in query or "generation" in query)
        ):
            db_path = query["db_path"][0]
            processed_count = query.get("processed_count", query.get("generation"))[0]
            return self.handle_get_meta_content(db_path, processed_count)

        if (
            path == "/download_meta_pdf"
            and "db_path" in query
            and ("processed_count" in query or "generation" in query)
        ):
            db_path = query["db_path"][0]
            processed_count = query.get("processed_count", query.get("generation"))[0]
            return self.handle_download_meta_pdf(db_path, processed_count)

        if (
            path == "/get_plots"
            and "db_path" in query
            and "generation" in query
            and "program_id" in query
        ):
            db_path = query["db_path"][0]
            generation = query["generation"][0]
            program_id = query["program_id"][0]
            return self.handle_get_plots(db_path, generation, program_id)

        if path.startswith("/plot_file/"):
            return self.handle_serve_plot_file()

        if path == "/get_system_prompts" and "db_path" in query:
            db_path = query["db_path"][0]
            return self.handle_get_system_prompts(db_path)

        if path == "/get_database_stats" and "db_path" in query:
            db_path = query["db_path"][0]
            return self.handle_get_database_stats(db_path)

        if path == "/":
            print("[SERVER] Root path requested, serving index.html")
            self.path = "/index.html"

        # Serve static files from the webui directory
        return http.server.SimpleHTTPRequestHandler.do_GET(self)

    def handle_list_databases(self):
        """Scan the search root directory for .db files."""
        print(
            f"[SERVER] Received request for database list, "
            f"searching in: {self.search_root}"
        )
        db_files = []
        date_pattern = re.compile(r"_(\d{8}_\d{6})")

        # Get the task name from the search root directory name
        task_name = os.path.basename(self.search_root)

        print(f"[SERVER] Scanning for .db files in: {self.search_root}")
        for client_path in self._walk_files_within_root():
            filename = os.path.basename(client_path)
            if filename.lower() in ("prompts.db", "prompts.sqlite"):
                continue

            path_parts = client_path.split(os.sep)
            display_name = (
                "/".join(path_parts[:-1])
                if len(path_parts) >= 2
                else client_path
            )
            task = path_parts[0] if len(path_parts) >= 2 else task_name

            match = date_pattern.search(client_path)
            sort_key = match.group(1) if match else "0"
            db_files.append(
                {
                    "path": client_path,
                    "name": display_name,
                    "task": task,
                    "sort_key": sort_key,
                    "actual_path": client_path,
                }
            )
            print(
                f"[SERVER] Found DB: {client_path} "
                f"(task: '{task}', result: '{display_name}')"
            )

        if not db_files:
            print("[SERVER] No database files found in search directory.")

        # Sort databases by the extracted date, newest first
        db_files.sort(key=lambda x: x.get("sort_key", "0"), reverse=True)

        # Remove sort_key before sending to client (but keep actual_path)
        for db in db_files:
            del db["sort_key"]

        print(f"[SERVER] Sending {len(db_files)} databases:")
        for i, db in enumerate(db_files):
            print(f"  [{i}] task='{db['task']}', result='{db['name']}'")

        self.send_json_response(db_files)
        print(f"[SERVER] Served DB list with {len(db_files)} entries, sorted by date.")

    def _resolve_within_root(self, relative_path: str) -> str:
        """Resolve a user-supplied path under search_root, rejecting escapes.

        Returns the canonical absolute path if it is contained within
        search_root; raises PathValidationError otherwise. Absolute inputs and
        ``..`` traversal both fail the containment check, and symlinks are
        resolved (realpath) before checking so an in-root symlink pointing
        outside cannot be used to escape.
        """
        root = self._canonical_root()
        candidate = os.path.realpath(os.path.join(root, relative_path))
        try:
            contained = os.path.commonpath([root, candidate]) == root
        except ValueError:
            # Different drives / mixed absolute-relative: treat as an escape.
            contained = False
        if not contained:
            raise PathValidationError(f"Path escapes search root: {relative_path!r}")
        return candidate

    @staticmethod
    def _supports_descriptor_traversal() -> bool:
        return (
            os.open in os.supports_dir_fd
            and os.listdir in os.supports_fd
            and os.stat in os.supports_dir_fd
            and hasattr(os, "O_DIRECTORY")
            and hasattr(os, "O_NOFOLLOW")
        )

    def _open_descriptor_within_root(
        self, path: os.PathLike[str] | str, flags: int
    ) -> int:
        resolved_path = self._resolve_within_root(os.fspath(path))
        if not self._supports_descriptor_traversal():
            return os.open(resolved_path, flags)

        root = self._canonical_root()
        relative_path = os.path.relpath(resolved_path, root)
        root_descriptor = self._duplicate_search_root_descriptor(root)
        try:
            return self._open_relative_descriptor(
                root_descriptor,
                relative_path,
                flags,
                original_path=os.fspath(path),
            )
        finally:
            os.close(root_descriptor)

    def _open_relative_descriptor(
        self,
        root_descriptor: int,
        relative_path: str,
        flags: int,
        *,
        original_path: str,
    ) -> int:
        relative_parts = Path(relative_path).parts
        if os.path.isabs(relative_path) or any(
            part in (os.curdir, os.pardir) for part in relative_parts
        ):
            raise PathValidationError(f"Path escapes search root: {original_path!r}")

        descriptor = os.dup(root_descriptor)
        try:
            for part in relative_parts[:-1]:
                next_descriptor = os.open(
                    part,
                    self._directory_open_flags(),
                    dir_fd=descriptor,
                )
                os.close(descriptor)
                descriptor = next_descriptor

            if not relative_parts:
                return os.dup(descriptor)
            return os.open(
                relative_parts[-1],
                flags | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
                dir_fd=descriptor,
            )
        except OSError as exc:
            if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                raise PathValidationError(
                    f"Path changed during access: {original_path!r}"
                ) from exc
            raise
        finally:
            os.close(descriptor)

    def _duplicate_search_root_descriptor(self, root: str) -> int:
        pinned_descriptor = getattr(self, "_search_root_descriptor", None)
        if pinned_descriptor is not None:
            return os.dup(pinned_descriptor)
        return self._open_search_root_descriptor(root)

    def _canonical_root(self) -> str:
        root = getattr(self, "_canonical_search_root", None)
        if root is None:
            root = os.path.realpath(self.search_root)
            self._canonical_search_root = root
        return root

    @staticmethod
    def _directory_open_flags() -> int:
        return (
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        )

    @classmethod
    def _open_search_root_descriptor(cls, search_root: os.PathLike[str] | str) -> int:
        if not cls._supports_descriptor_traversal():
            raise PathValidationError(
                "Race-safe path access is unavailable on this platform"
            )

        descriptor = os.open(os.path.sep, cls._directory_open_flags())
        try:
            for part in Path(os.path.realpath(search_root)).parts[1:]:
                next_descriptor = os.open(
                    part,
                    cls._directory_open_flags(),
                    dir_fd=descriptor,
                )
                os.close(descriptor)
                descriptor = next_descriptor
            return descriptor
        except Exception:
            os.close(descriptor)
            raise

    def _read_text_within_root(
        self, path: os.PathLike[str] | str, encoding: str = "utf-8"
    ) -> str:
        descriptor = self._open_descriptor_within_root(path, os.O_RDONLY)
        with os.fdopen(descriptor, "r", encoding=encoding) as file:
            return file.read()

    def _read_bytes_within_root(self, path: os.PathLike[str] | str) -> bytes:
        descriptor = self._open_descriptor_within_root(path, os.O_RDONLY)
        with os.fdopen(descriptor, "rb") as file:
            return file.read()

    def _database_file_identity_within_root(
        self,
        path: os.PathLike[str] | str,
    ) -> Tuple[int, int, int, int, int]:
        descriptor = self._open_descriptor_within_root(path, os.O_RDONLY)
        try:
            file_stat = os.fstat(descriptor)
            if not stat.S_ISREG(file_stat.st_mode):
                raise PathValidationError(f"Database is not a file: {path!r}")
            return (
                file_stat.st_dev,
                file_stat.st_ino,
                file_stat.st_size,
                file_stat.st_mtime_ns,
                file_stat.st_ctime_ns,
            )
        finally:
            os.close(descriptor)

    def _program_response_cache_key(self, resolved_path: str) -> Tuple[str, str]:
        return (
            os.path.normcase(self._canonical_root()),
            os.path.normcase(resolved_path),
        )

    @staticmethod
    def clear_program_response_cache(search_root: str) -> None:
        canonical_root = os.path.normcase(os.path.realpath(search_root))
        with db_cache_lock:
            stale_keys = [key for key in db_cache if key[0] == canonical_root]
            for key in stale_keys:
                db_cache.pop(key, None)

    @classmethod
    def _copy_database_descriptor(
        cls,
        source_descriptor: int,
        destination_descriptor: int,
        destination_name: str,
        *,
        expected_stat: os.stat_result,
    ) -> None:
        source_stat = os.fstat(source_descriptor)
        if not cls._same_database_file(source_stat, expected_stat, contents=True):
            raise DatabaseViewRaceError("Database changed before snapshotting")

        destination = os.open(
            destination_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=destination_descriptor,
        )
        try:
            offset = 0
            remaining = expected_stat.st_size
            while remaining:
                chunk = os.pread(
                    source_descriptor,
                    min(1024 * 1024, remaining),
                    offset,
                )
                if not chunk:
                    raise DatabaseViewRaceError(
                        "Database changed while snapshotting"
                    )
                view = memoryview(chunk)
                while view:
                    written = os.write(destination, view)
                    if written == 0:
                        raise OSError("Unable to write database snapshot")
                    view = view[written:]
                offset += len(chunk)
                remaining -= len(chunk)
        finally:
            os.close(destination)

        source_stat = os.fstat(source_descriptor)
        if not cls._same_database_file(source_stat, expected_stat, contents=True):
            raise DatabaseViewRaceError("Database changed while snapshotting")

    def _copy_optional_database_sidecar(
        self,
        parent_descriptor: int,
        source_name: str,
        destination_descriptor: int,
        destination_name: str,
        *,
        max_bytes: int,
    ) -> Optional[os.stat_result]:
        try:
            source_descriptor = os.open(
                source_name,
                os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
                dir_fd=parent_descriptor,
            )
        except FileNotFoundError:
            return None
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise PathValidationError(
                    f"Database sidecar changed during access: {source_name!r}"
                ) from exc
            raise
        try:
            source_stat = os.fstat(source_descriptor)
            if not stat.S_ISREG(source_stat.st_mode):
                raise PathValidationError(
                    f"Database sidecar is not a file: {source_name!r}"
                )
            if source_stat.st_size > max_bytes:
                raise sqlite3.OperationalError(
                    "database and WAL exceed the 2 GiB WebUI snapshot limit"
                )
            self._copy_database_descriptor(
                source_descriptor,
                destination_descriptor,
                destination_name,
                expected_stat=source_stat,
            )
            return source_stat
        finally:
            os.close(source_descriptor)

    @staticmethod
    def _stat_optional_database_sidecar(
        parent_descriptor: int,
        source_name: str,
    ) -> Optional[os.stat_result]:
        try:
            source_stat = os.stat(
                source_name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return None
        if not stat.S_ISREG(source_stat.st_mode):
            raise PathValidationError(
                f"Database sidecar is not a file: {source_name!r}"
            )
        return source_stat

    @classmethod
    def _rollback_journal_is_hot(
        cls,
        parent_descriptor: int,
        source_name: str,
        expected_stat: os.stat_result,
    ) -> bool:
        try:
            source_descriptor = os.open(
                source_name,
                os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
                dir_fd=parent_descriptor,
            )
        except FileNotFoundError as exc:
            raise DatabaseViewRaceError(
                "Rollback journal changed while snapshotting"
            ) from exc
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise PathValidationError(
                    f"Database sidecar changed during access: {source_name!r}"
                ) from exc
            raise
        try:
            if not cls._same_database_file(
                os.fstat(source_descriptor), expected_stat, contents=True
            ):
                raise DatabaseViewRaceError(
                    "Rollback journal changed while snapshotting"
                )
            header = os.pread(source_descriptor, 8, 0)
            if not cls._same_database_file(
                os.fstat(source_descriptor), expected_stat, contents=True
            ):
                raise DatabaseViewRaceError(
                    "Rollback journal changed while snapshotting"
                )
        finally:
            os.close(source_descriptor)
        return header == b"\xd9\xd5\x05\xf9\x20\xa1\x63\xd7"

    @contextlib.contextmanager
    def _database_staging_directory(
        self,
        resolved_database_path: str,
        excluded_parents: set[str],
    ):
        for staging_parent in self._database_staging_parents(
            resolved_database_path,
        ):
            if staging_parent in excluded_parents:
                continue
            try:
                temp_directory = tempfile.TemporaryDirectory(
                    prefix="shinka-webui-db-",
                    dir=staging_parent,
                )
            except OSError:
                continue
            directory_descriptor = None
            try:
                directory_descriptor = os.open(
                    temp_directory.name,
                    self._directory_open_flags(),
                )
                yield directory_descriptor, temp_directory.name, staging_parent
                return
            finally:
                if directory_descriptor is not None:
                    os.close(directory_descriptor)
                temp_directory.cleanup()

        raise sqlite3.OperationalError("no secure writable staging directory exists")

    @classmethod
    def _database_staging_parents(
        cls,
        resolved_database_path: str,
    ) -> list[str]:
        database_parent = os.path.dirname(resolved_database_path)
        candidates = [
            tempfile.gettempdir(),
            "/var/tmp",
            "/tmp",
            "/dev/shm",
        ]
        candidates.extend(
            os.fspath(parent)
            for parent in reversed(Path(database_parent).parents)
        )

        staging_parents = []
        seen = set()
        for candidate in candidates:
            canonical_candidate = os.path.realpath(candidate)
            if canonical_candidate in seen or canonical_candidate == database_parent:
                continue
            seen.add(canonical_candidate)
            try:
                os.stat(canonical_candidate)
            except OSError:
                continue
            if not cls._is_secure_staging_parent(canonical_candidate):
                continue
            if not os.access(canonical_candidate, os.W_OK | os.X_OK):
                continue
            staging_parents.append(canonical_candidate)
        return staging_parents

    @staticmethod
    def _database_cache_key(
        database_stat: os.stat_result,
    ) -> Tuple[int, int, int]:
        return (
            database_stat.st_size,
            database_stat.st_mtime_ns,
            database_stat.st_ctime_ns,
        )

    def _cached_database_main(
        self,
        resolved_database_path: str,
        database_descriptor: int,
        database_stat: os.stat_result,
    ) -> _DatabaseMainSnapshot:
        if database_stat.st_size > self._database_main_cache_max_bytes:
            raise sqlite3.OperationalError(
                "database exceeds the 2 GiB WebUI snapshot limit"
            )
        source_key = (database_stat.st_dev, database_stat.st_ino)
        cache_key = self._database_cache_key(database_stat)
        with self._database_main_cache_lock:
            build_lock = self._database_main_cache_build_locks.setdefault(
                source_key,
                threading.Lock(),
            )

        with build_lock:
            discarded: list[_DatabaseMainSnapshot] = []
            with self._database_main_cache_lock:
                cached = self._database_main_cache.pop(source_key, None)
                if cached is not None and cached.version == cache_key:
                    cached.leases += 1
                    self._database_main_cache[source_key] = cached
                    return cached
                if cached is not None:
                    cached.evicted = True
                    discarded.append(cached)
                self._evict_database_main_cache_entries(
                    incoming_size=database_stat.st_size,
                    discarded=discarded,
                )
            self._cleanup_database_snapshots(discarded)

            staging_parents = self._database_staging_parents(
                resolved_database_path,
            )
            if not staging_parents:
                raise sqlite3.OperationalError(
                    "no secure writable staging directory exists"
                )
            temp_directory, cached_path = self._copy_database_main_to_staging(
                staging_parents,
                database_descriptor,
                database_stat,
            )

            cached = _DatabaseMainSnapshot(
                cache_key,
                temp_directory,
                cached_path,
            )
            cached.leases = 1
            discarded = []
            with self._database_main_cache_lock:
                self._evict_database_main_cache_entries(
                    incoming_size=database_stat.st_size,
                    discarded=discarded,
                )
                self._database_main_cache[source_key] = cached
            self._cleanup_database_snapshots(discarded)
            return cached

    def _copy_database_main_to_staging(
        self,
        staging_parents: list[str],
        database_descriptor: int,
        database_stat: os.stat_result,
    ) -> Tuple[tempfile.TemporaryDirectory, str]:
        last_error = None
        for staging_parent in staging_parents:
            temp_directory = None
            directory_descriptor = None
            succeeded = False
            try:
                temp_directory = tempfile.TemporaryDirectory(
                    prefix="shinka-webui-main-",
                    dir=staging_parent,
                )
                directory_descriptor = os.open(
                    temp_directory.name,
                    self._directory_open_flags(),
                )
                self._copy_database_descriptor(
                    database_descriptor,
                    directory_descriptor,
                    "database.sqlite",
                    expected_stat=database_stat,
                )
                cached_path = os.path.join(
                    temp_directory.name,
                    "database.sqlite",
                )
                os.chmod(cached_path, 0o400)
                succeeded = True
                return temp_directory, cached_path
            except OSError as exc:
                if not self._is_retryable_staging_error(exc):
                    raise
                last_error = exc
            finally:
                if directory_descriptor is not None:
                    os.close(directory_descriptor)
                if temp_directory is not None and not succeeded:
                    temp_directory.cleanup()

        raise sqlite3.OperationalError(
            "no secure staging directory has enough space for the database snapshot"
        ) from last_error

    @staticmethod
    def _is_retryable_staging_error(exc: OSError) -> bool:
        return exc.errno in {
            errno.EACCES,
            errno.EPERM,
            errno.ENOSPC,
            errno.EROFS,
            getattr(errno, "EDQUOT", -1),
        }

    @classmethod
    def _evict_database_main_cache_entries(
        cls,
        *,
        incoming_size: int,
        discarded: list[_DatabaseMainSnapshot],
    ) -> None:
        cached_bytes = sum(
            entry.version[0] for entry in cls._database_main_cache.values()
        )
        while cls._database_main_cache and (
            len(cls._database_main_cache) >= cls._database_main_cache_limit
            or cached_bytes + incoming_size > cls._database_main_cache_max_bytes
        ):
            oldest_key = next(iter(cls._database_main_cache))
            evicted = cls._database_main_cache.pop(oldest_key)
            evicted.evicted = True
            cached_bytes -= evicted.version[0]
            discarded.append(evicted)

    @staticmethod
    def _cleanup_database_snapshots(
        snapshots: list[_DatabaseMainSnapshot],
    ) -> None:
        for snapshot in snapshots:
            if snapshot.leases == 0:
                snapshot.directory.cleanup()

    @classmethod
    def _release_database_main_snapshot(
        cls,
        snapshot: _DatabaseMainSnapshot,
    ) -> None:
        cleanup = False
        with cls._database_main_cache_lock:
            snapshot.leases -= 1
            cleanup = snapshot.evicted and snapshot.leases == 0
        if cleanup:
            snapshot.directory.cleanup()

    @classmethod
    def clear_database_snapshot_cache(cls) -> None:
        with cls._database_main_cache_lock:
            cached = list(cls._database_main_cache.values())
            cls._database_main_cache.clear()
            cls._database_main_cache_build_locks.clear()
            for snapshot in cached:
                snapshot.evicted = True
        cls._cleanup_database_snapshots(cached)

    @classmethod
    def _stage_cached_database_main(
        cls,
        cached_path: str,
        destination_descriptor: int,
    ) -> None:
        try:
            os.link(
                cached_path,
                "database.sqlite",
                dst_dir_fd=destination_descriptor,
                follow_symlinks=False,
            )
            return
        except OSError as exc:
            if exc.errno not in {
                errno.EACCES,
                errno.EPERM,
                errno.EXDEV,
                errno.EMLINK,
                errno.ENOSYS,
                errno.EROFS,
                getattr(errno, "ENOTSUP", errno.EINVAL),
                getattr(errno, "EOPNOTSUPP", errno.EINVAL),
            }:
                raise

        cached_descriptor = os.open(
            cached_path,
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            cached_stat = os.fstat(cached_descriptor)
            cls._copy_database_descriptor(
                cached_descriptor,
                destination_descriptor,
                "database.sqlite",
                expected_stat=cached_stat,
            )
        finally:
            os.close(cached_descriptor)

    @staticmethod
    def _is_secure_staging_parent(candidate: str) -> bool:
        effective_uid = getattr(os, "geteuid", lambda: -1)()
        trusted_uids = {0, effective_uid}
        current = candidate
        while True:
            try:
                current_stat = os.stat(current)
            except OSError:
                return False
            if not stat.S_ISDIR(current_stat.st_mode):
                return False
            if current_stat.st_uid not in trusted_uids:
                return False
            untrusted_writable = current_stat.st_mode & 0o022
            if untrusted_writable and not current_stat.st_mode & stat.S_ISVTX:
                return False
            parent = os.path.dirname(current)
            if parent == current:
                return True
            current = parent

    @staticmethod
    def _verify_database_sidecars(
        parent_descriptor: int,
        expected_sidecars: Dict[str, Optional[os.stat_result]],
        *,
        verify_contents: bool,
    ) -> None:
        for source_name, expected_stat in expected_sidecars.items():
            try:
                current_stat = os.stat(
                    source_name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                current_stat = None
            if expected_stat is None and current_stat is None:
                continue
            if expected_stat is None or current_stat is None:
                raise DatabaseViewRaceError(
                    f"Database sidecar changed while pinning {source_name!r}"
                )
            sidecar_contents_matter = verify_contents and not source_name.endswith(
                "-shm"
            )
            if not DatabaseRequestHandler._same_database_file(
                current_stat,
                expected_stat,
                contents=sidecar_contents_matter,
            ):
                raise DatabaseViewRaceError(
                    f"Database sidecar changed while pinning {source_name!r}"
                )

    @staticmethod
    def _same_database_file(
        current_stat: os.stat_result,
        expected_stat: os.stat_result,
        *,
        contents: bool,
    ) -> bool:
        if (
            current_stat.st_dev != expected_stat.st_dev
            or current_stat.st_ino != expected_stat.st_ino
            or not stat.S_ISREG(current_stat.st_mode)
        ):
            return False
        if not contents:
            return True
        return (
            current_stat.st_size == expected_stat.st_size
            and current_stat.st_mtime_ns == expected_stat.st_mtime_ns
            and current_stat.st_ctime_ns == expected_stat.st_ctime_ns
        )

    @staticmethod
    def _sqlite_read_only_uri(path: os.PathLike[str] | str) -> str:
        path_string = os.fspath(path)
        extended_prefix = "\\\\?\\"
        if path_string.startswith("\\\\.\\"):
            raise PathValidationError("Windows device paths are not supported")
        if path_string.startswith(extended_prefix):
            extended_path = path_string[len(extended_prefix) :]
            if extended_path.casefold().startswith("unc\\"):
                path_string = "\\\\" + extended_path[4:]
            elif re.match(r"^[A-Za-z]:[\\/]", extended_path):
                path_string = extended_path
            else:
                raise PathValidationError("Windows device paths are not supported")

        if path_string.startswith("\\\\"):
            # Keep the URI authority empty. SQLite rejects remote authorities
            # unless built with SQLITE_ALLOW_URI_AUTHORITY, while the Windows
            # VFS accepts an absolute //server/share path as UNC.
            quoted_path = urllib.parse.quote(
                path_string.replace("\\", "/"),
                safe="/:",
            )
            return "file://" + quoted_path + "?mode=ro"
        if re.match(r"^[A-Za-z]:[\\/]", path_string):
            uri = PureWindowsPath(path_string).as_uri()
        else:
            uri = Path(path_string).as_uri()
        return uri + "?mode=ro"

    @classmethod
    def _sqlite_read_only_target(
        cls,
        path: os.PathLike[str] | str,
    ) -> Tuple[str, bool]:
        return cls._sqlite_read_only_uri(path), True

    def _stage_database_view(
        self,
        *,
        cached_main_path: str,
        parent_descriptor: int,
        source_name: str,
        destination_descriptor: int,
        max_wal_bytes: int,
    ) -> Dict[str, Optional[os.stat_result]]:
        self._stage_cached_database_main(
            cached_main_path,
            destination_descriptor,
        )

        sidecars = {}
        for suffix in ("-wal", "-shm", "-journal"):
            sidecar_name = source_name + suffix
            if suffix == "-wal":
                sidecar_stat = self._copy_optional_database_sidecar(
                    parent_descriptor,
                    sidecar_name,
                    destination_descriptor,
                    "database.sqlite" + suffix,
                    max_bytes=max_wal_bytes,
                )
            else:
                sidecar_stat = self._stat_optional_database_sidecar(
                    parent_descriptor,
                    sidecar_name,
                )
                if (
                    suffix == "-journal"
                    and sidecar_stat is not None
                    and self._rollback_journal_is_hot(
                        parent_descriptor,
                        sidecar_name,
                        sidecar_stat,
                    )
                ):
                    raise DatabaseViewRaceError(
                        "Rollback journal is active while snapshotting"
                    )
            sidecars[sidecar_name] = sidecar_stat
        return sidecars

    def _verify_database_view(
        self,
        *,
        database_descriptor: int,
        database_stat: os.stat_result,
        parent_descriptor: int,
        source_name: str,
        sidecars: Dict[str, Optional[os.stat_result]],
    ) -> None:
        try:
            current_database_stat = os.stat(
                source_name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError as exc:
            raise DatabaseViewRaceError(
                "Database path changed while pinning SQLite sidecars"
            ) from exc
        if not self._same_database_file(
            current_database_stat,
            database_stat,
            contents=False,
        ):
            raise DatabaseViewRaceError(
                "Database path changed while pinning SQLite sidecars"
            )
        if not self._same_database_file(
            os.fstat(database_descriptor), database_stat, contents=True
        ):
            raise DatabaseViewRaceError("Database changed while snapshotting")
        self._verify_database_sidecars(
            parent_descriptor,
            sidecars,
            verify_contents=True,
        )

    @contextlib.contextmanager
    def _connect_database_within_root(
        self,
        path: os.PathLike[str] | str,
        *,
        timeout: float,
        isolation_level: Optional[
            Literal["DEFERRED", "EXCLUSIVE", "IMMEDIATE"]
        ] = None,
    ):
        if not self._supports_descriptor_traversal():
            resolved_path = self._resolve_within_root(os.fspath(path))
            connection_target, use_uri = self._sqlite_read_only_target(resolved_path)
            connection = sqlite3.connect(
                connection_target,
                uri=use_uri,
                timeout=timeout,
                isolation_level=isolation_level,
            )
            try:
                if not use_uri:
                    connection.execute("PRAGMA query_only = ON")
                yield connection
            finally:
                connection.close()
            return

        resolved_path = self._resolve_within_root(os.fspath(path))
        database_descriptor = self._open_descriptor_within_root(
            resolved_path, os.O_RDONLY
        )
        parent_descriptor = None
        try:
            database_stat = os.fstat(database_descriptor)
            if not stat.S_ISREG(database_stat.st_mode):
                raise PathValidationError(f"Database is not a file: {path!r}")
            parent_descriptor = self._open_descriptor_within_root(
                os.path.dirname(resolved_path),
                os.O_RDONLY | os.O_DIRECTORY,
            )
            source_name = os.path.basename(resolved_path)
            race_attempts = 0
            excluded_staging_parents: set[str] = set()
            while race_attempts < 3:
                attempt_database_stat = os.fstat(database_descriptor)
                active_staging_parent = None
                connection_yielded = False
                try:
                    cached_main = self._cached_database_main(
                        resolved_path,
                        database_descriptor,
                        attempt_database_stat,
                    )
                    with contextlib.ExitStack() as snapshot_stack:
                        snapshot_stack.callback(
                            self._release_database_main_snapshot,
                            cached_main,
                        )
                        if not cached_main.lock.acquire(timeout=timeout):
                            raise DatabaseViewRaceError(
                                "database snapshot is busy"
                            )
                        snapshot_stack.callback(cached_main.lock.release)
                        (
                            staging_descriptor,
                            staging_path,
                            active_staging_parent,
                        ) = snapshot_stack.enter_context(
                            self._database_staging_directory(
                                resolved_path,
                                excluded_staging_parents,
                            )
                        )
                        sidecars = self._stage_database_view(
                            cached_main_path=cached_main.path,
                            parent_descriptor=parent_descriptor,
                            source_name=source_name,
                            destination_descriptor=staging_descriptor,
                            max_wal_bytes=(
                                self._database_main_cache_max_bytes
                                - attempt_database_stat.st_size
                            ),
                        )
                        self._verify_database_view(
                            database_descriptor=database_descriptor,
                            database_stat=attempt_database_stat,
                            parent_descriptor=parent_descriptor,
                            source_name=source_name,
                            sidecars=sidecars,
                        )
                        stable_path = os.path.join(staging_path, "database.sqlite")
                        connection = None
                        try:
                            connection = sqlite3.connect(
                                self._sqlite_read_only_uri(stable_path),
                                uri=True,
                                timeout=timeout,
                                isolation_level=isolation_level,
                            )
                            connection.execute("BEGIN")
                            connection.execute("PRAGMA schema_version").fetchone()
                        except sqlite3.DatabaseError:
                            try:
                                self._verify_database_view(
                                    database_descriptor=database_descriptor,
                                    database_stat=attempt_database_stat,
                                    parent_descriptor=parent_descriptor,
                                    source_name=source_name,
                                    sidecars=sidecars,
                                )
                            finally:
                                if connection is not None:
                                    connection.close()
                            raise
                        try:
                            self._verify_database_view(
                                database_descriptor=database_descriptor,
                                database_stat=attempt_database_stat,
                                parent_descriptor=parent_descriptor,
                                source_name=source_name,
                                sidecars=sidecars,
                            )
                        except Exception:
                            connection.close()
                            raise
                        try:
                            connection_yielded = True
                            yield connection
                        finally:
                            connection.close()
                        return
                except OSError as exc:
                    if (
                        connection_yielded
                        or active_staging_parent is None
                        or not self._is_retryable_staging_error(exc)
                    ):
                        raise
                    excluded_staging_parents.add(active_staging_parent)
                    continue
                except DatabaseViewRaceError:
                    race_attempts += 1
                    if race_attempts == 3:
                        raise sqlite3.OperationalError(
                            "database is busy while pinning SQLite sidecars"
                        )
        finally:
            if parent_descriptor is not None:
                os.close(parent_descriptor)
            os.close(database_descriptor)

    @contextlib.contextmanager
    def _open_program_database_within_root(self, path: os.PathLike[str] | str):
        with self._connect_database_within_root(path, timeout=60.0) as connection:
            database = ProgramDatabase(
                DatabaseConfig(db_path=os.fspath(path)),
                read_only=True,
                connection=connection,
            )
            try:
                yield database
            finally:
                database.close()

    @contextlib.contextmanager
    def _open_prompt_database_within_root(self, path: os.PathLike[str] | str):
        with self._connect_database_within_root(path, timeout=30.0) as connection:
            database = SystemPromptDatabase(
                SystemPromptConfig(db_path=os.fspath(path)),
                read_only=True,
                connection=connection,
            )
            try:
                yield database
            finally:
                database.close()

    def _list_directory_within_root(
        self, path: os.PathLike[str] | str
    ) -> list[str]:
        if not self._supports_descriptor_traversal():
            return os.listdir(self._resolve_within_root(os.fspath(path)))
        descriptor = self._open_descriptor_within_root(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            return os.listdir(descriptor)
        finally:
            os.close(descriptor)

    def _walk_files_within_root(self) -> list[str]:
        if not self._supports_descriptor_traversal():
            return [
                os.path.relpath(os.path.join(root, filename), self.search_root)
                for root, _, files in os.walk(self.search_root)
                for filename in files
                if self._is_database_filename(filename)
            ]

        root_descriptor = self._duplicate_search_root_descriptor(
            self._canonical_root()
        )
        try:
            return self._walk_regular_files(root_descriptor)
        finally:
            os.close(root_descriptor)

    def _walk_regular_files(self, root_descriptor: int) -> list[str]:
        paths = []
        pending_directories = [""]
        while pending_directories:
            relative_directory = pending_directories.pop()
            try:
                descriptor = self._open_relative_descriptor(
                    root_descriptor,
                    relative_directory,
                    os.O_RDONLY | os.O_DIRECTORY,
                    original_path=relative_directory,
                )
            except OSError as exc:
                if self._is_skippable_walk_error(exc):
                    continue
                raise
            try:
                child_directories = self._scan_database_directory(
                    descriptor,
                    relative_directory,
                    paths,
                )
                pending_directories.extend(reversed(child_directories))
            except OSError as exc:
                if not self._is_skippable_walk_error(exc):
                    raise
            finally:
                os.close(descriptor)
        return paths

    def _scan_database_directory(
        self,
        descriptor: int,
        prefix: str,
        paths: list[str],
    ) -> list[str]:
        child_directories = []
        for name in os.listdir(descriptor):
            try:
                entry = os.stat(
                    name,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
            except OSError as exc:
                if self._is_skippable_walk_error(exc):
                    continue
                raise
            relative_path = os.path.join(prefix, name) if prefix else name
            if stat.S_ISREG(entry.st_mode):
                if self._is_database_filename(name):
                    paths.append(relative_path)
                continue
            if stat.S_ISLNK(entry.st_mode):
                if not self._is_database_filename(name):
                    continue
                try:
                    file_descriptor = self._open_descriptor_within_root(
                        relative_path, os.O_RDONLY
                    )
                except PathValidationError:
                    continue
                except OSError as exc:
                    if self._is_skippable_walk_error(exc):
                        continue
                    raise
                try:
                    if stat.S_ISREG(os.fstat(file_descriptor).st_mode):
                        paths.append(relative_path)
                finally:
                    os.close(file_descriptor)
                continue
            if not stat.S_ISDIR(entry.st_mode):
                continue
            child_directories.append(relative_path)
        return child_directories

    @staticmethod
    def _is_database_filename(filename: str) -> bool:
        return filename.lower().endswith((".db", ".sqlite"))

    @staticmethod
    def _is_skippable_walk_error(exc: OSError) -> bool:
        return exc.errno in {
            errno.ENOENT,
            errno.EACCES,
            errno.EPERM,
            errno.ELOOP,
            errno.ENOTDIR,
            getattr(errno, "ESTALE", -1),
        }

    def _get_actual_db_path(self, db_path: str) -> str:
        """Validate db_path stays within search_root; return its absolute path.

        Handlers pass the result to ``os.path.join(self.search_root, ...)``,
        which is a no-op on the absolute path returned here.
        """
        return self._resolve_within_root(db_path)

    def handle_get_programs(self, db_path: str):
        """Fetch all programs from a given database file."""
        print(f"[SERVER] Fetching programs from DB: {db_path}")

        # Handle the case where db_path might have the task name prepended
        # Extract the actual path by removing the task name prefix if present
        actual_db_path = self._get_actual_db_path(db_path)

        # Construct absolute path to the database from search root using actual path
        abs_db_path = os.path.join(self.search_root, actual_db_path)
        print(f"[SERVER] Absolute DB path: {abs_db_path} (from {db_path})")
        try:
            database_identity = self._database_file_identity_within_root(abs_db_path)
        except FileNotFoundError:
            self.send_error(404, f"Database file not found: {actual_db_path}")
            return
        cache_key = self._program_response_cache_key(actual_db_path)

        # Check cache first
        with db_cache_lock:
            cached_entry = db_cache.get(cache_key)
        if cached_entry is not None:
            cached_identity, last_fetch_time, cached_data = cached_entry
            if time.time() - last_fetch_time < CACHE_EXPIRATION_SECONDS:
                if cached_identity == database_identity:
                    print(f"[SERVER] Serving from cache for DB: {db_path}")
                    self.send_json_response(cached_data)
                    return

        # Retry logic for the reader with improved WAL mode support
        # More retries with longer delays during active evolution
        max_retries = 8
        delay = 0.2
        for i in range(max_retries):
            try:
                with self._open_program_database_within_root(abs_db_path) as db:
                    # Set WAL mode compatible settings for read-only connections
                    # Longer busy_timeout for concurrent access during evolution
                    if db.cursor:
                        db.cursor.execute("PRAGMA busy_timeout = 30000;")

                    programs = db.get_all_programs()

                    # Convert Program objects to dicts for JSON
                    programs_dict = [p.to_dict() for p in programs]
                programs_dict.extend(
                    self._load_failed_proposal_nodes(
                        abs_db_path, include_code=False
                    )
                )

                # Update cache
                with db_cache_lock:
                    db_cache[cache_key] = (
                        database_identity,
                        time.time(),
                        programs_dict,
                    )

                self.send_json_response(programs_dict)
                success_msg = (
                    f"[SERVER] Successfully served {len(programs)} "
                    f"programs from {db_path} (attempt {i + 1})"
                )
                print(success_msg)
                return  # Success, exit the retry loop

            except PathValidationError:
                raise
            except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
                error_str = str(e).lower()
                is_retryable = (
                    "database is locked" in error_str
                    or "busy" in error_str
                    or "disk i/o error" in error_str  # Occurs during heavy writes
                )
                if is_retryable:
                    print(
                        f"[SERVER] Attempt {i + 1}/{max_retries} - database busy/locked, "
                        f"retrying in {delay:.1f}s... ({e})"
                    )
                    if i < max_retries - 1:
                        time.sleep(delay)
                        delay = min(
                            delay * 1.5, 5.0
                        )  # Longer max delay during active evolution
                        continue
                    else:
                        # Last retry failed
                        err_msg = (
                            f"[SERVER] Database still busy after {max_retries} attempts"
                        )
                        print(err_msg)
                        self.send_error(
                            503,
                            "Database temporarily unavailable - evolution may be running",
                        )
                        return
                else:
                    print(f"[SERVER] Non-recoverable database error: {e}")
                    self.send_error(500, f"Database error: {str(e)}")
                    return

            except PathValidationError:
                raise
            except Exception as e:
                # Catch any other unexpected errors
                print(f"[SERVER] An unexpected error occurred: {e}")
                self.send_error(500, f"An unexpected error occurred: {str(e)}")
                return  # Don't retry on unknown errors
    def handle_get_programs_summary(self, db_path: str):
        """Fetch lightweight program summaries (no code, no embeddings)."""
        print(f"[SERVER] Fetching program summaries from DB: {db_path}")

        actual_db_path = self._get_actual_db_path(db_path)
        abs_db_path = os.path.join(self.search_root, actual_db_path)

        if not os.path.exists(abs_db_path):
            self.send_error(404, f"Database file not found: {actual_db_path}")
            return

        max_retries = 8
        delay = 0.2
        for i in range(max_retries):
            try:
                with self._open_program_database_within_root(abs_db_path) as db:
                    if db.cursor:
                        db.cursor.execute("PRAGMA busy_timeout = 30000;")

                    summaries = db.get_programs_summary()
                summaries.extend(
                    self._load_failed_proposal_nodes(
                        abs_db_path, include_code=False
                    )
                )
                self.send_json_response(summaries)
                print(
                    f"[SERVER] Successfully served {len(summaries)} "
                    f"program summaries from {db_path}"
                )
                return

            except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
                error_str = str(e).lower()
                is_retryable = (
                    "database is locked" in error_str
                    or "busy" in error_str
                    or "disk i/o error" in error_str
                )
                if is_retryable:
                    if i < max_retries - 1:
                        time.sleep(delay)
                        delay = min(delay * 1.5, 5.0)
                        continue
                    else:
                        self.send_error(
                            503,
                            "Database temporarily unavailable - evolution may be running",
                        )
                        return
                else:
                    self.send_error(500, f"Database error: {str(e)}")
                    return

            except PathValidationError:
                raise
            except Exception as e:
                print(f"[SERVER] Error fetching program summaries: {e}")
                self.send_error(500, f"Error: {str(e)}")
                return
    def handle_get_program_count(self, db_path: str):
        """Get program count and max timestamp for efficient change detection."""
        print(f"[SERVER] Fetching program count from DB: {db_path}")

        actual_db_path = self._get_actual_db_path(db_path)
        abs_db_path = os.path.join(self.search_root, actual_db_path)

        if not os.path.exists(abs_db_path):
            self.send_error(404, f"Database file not found: {actual_db_path}")
            return

        max_retries = 8
        delay = 0.2
        for i in range(max_retries):
            try:
                with self._open_program_database_within_root(abs_db_path) as db:
                    if db.cursor:
                        db.cursor.execute("PRAGMA busy_timeout = 30000;")

                    result = db.get_program_count_and_timestamp()
                failed_nodes = self._load_failed_proposal_nodes(
                    abs_db_path, include_code=False
                )
                if failed_nodes:
                    result["count"] += len(failed_nodes)
                    max_failure_timestamp = max(
                        node["timestamp"]
                        for node in failed_nodes
                        if node.get("timestamp") is not None
                    )
                    if (
                        result.get("max_timestamp") is None
                        or max_failure_timestamp > result["max_timestamp"]
                    ):
                        result["max_timestamp"] = max_failure_timestamp
                self.send_json_response(result)
                return

            except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
                error_str = str(e).lower()
                is_retryable = (
                    "database is locked" in error_str
                    or "busy" in error_str
                    or "disk i/o error" in error_str
                )
                if is_retryable:
                    if i < max_retries - 1:
                        time.sleep(delay)
                        delay = min(delay * 1.5, 5.0)
                        continue
                    else:
                        self.send_error(
                            503,
                            "Database temporarily unavailable - evolution may be running",
                        )
                        return
                else:
                    self.send_error(500, f"Database error: {str(e)}")
                    return

            except PathValidationError:
                raise
            except Exception as e:
                print(f"[SERVER] Error fetching program count: {e}")
                self.send_error(500, f"Error: {str(e)}")
                return
    def handle_get_program_details(self, db_path: str, program_id: str):
        """Get full details for a single program (including code and embeddings)."""
        print(f"[SERVER] Fetching program details for ID: {program_id}")

        actual_db_path = self._get_actual_db_path(db_path)
        abs_db_path = os.path.join(self.search_root, actual_db_path)

        if not os.path.exists(abs_db_path):
            self.send_error(404, f"Database file not found: {actual_db_path}")
            return

        failed_generation = self._parse_failed_node_generation(program_id)
        if failed_generation is not None:
            try:
                failed_nodes = self._load_failed_proposal_nodes(
                    abs_db_path,
                    include_code=True,
                    generation=failed_generation,
                )
                if not failed_nodes:
                    self.send_error(404, f"Program not found: {program_id}")
                    return
                self.send_json_response(failed_nodes[0])
                return
            except PathValidationError:
                raise
            except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
                self.send_error(500, f"Database error: {str(e)}")
                return
            except Exception as e:
                print(f"[SERVER] Error fetching failed node details: {e}")
                self.send_error(500, f"Error: {str(e)}")
                return

        max_retries = 8
        delay = 0.2
        for i in range(max_retries):
            try:
                with self._open_program_database_within_root(abs_db_path) as db:
                    if db.cursor:
                        db.cursor.execute("PRAGMA busy_timeout = 30000;")

                    program = db.get(program_id)
                if program is None:
                    self.send_error(404, f"Program not found: {program_id}")
                    return

                self.send_json_response(program.to_dict())
                return

            except PathValidationError:
                raise
            except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
                error_str = str(e).lower()
                is_retryable = (
                    "database is locked" in error_str
                    or "busy" in error_str
                    or "disk i/o error" in error_str
                )
                if is_retryable:
                    if i < max_retries - 1:
                        time.sleep(delay)
                        delay = min(delay * 1.5, 5.0)
                        continue
                    else:
                        self.send_error(
                            503,
                            "Database temporarily unavailable - evolution may be running",
                        )
                        return
                else:
                    self.send_error(500, f"Database error: {str(e)}")
                    return

            except Exception as e:
                print(f"[SERVER] Error fetching program details: {e}")
                self.send_error(500, f"Error: {str(e)}")
                return

    def handle_get_meta_files(self, db_path: str):
        """List available meta files keyed by processed-count suffix."""
        print(f"[SERVER] Listing meta files for DB: {db_path}")

        # Get the actual database path
        actual_db_path = self._get_actual_db_path(db_path)

        # Get the directory containing the database file
        abs_db_path = os.path.join(self.search_root, actual_db_path)
        db_dir = os.path.dirname(abs_db_path)

        # Look in the meta subdirectory
        meta_dir = self._resolve_within_root(os.path.join(db_dir, "meta"))

        if not os.path.exists(meta_dir):
            # Fall back to looking in the db_dir for backward compatibility
            print("[SERVER] Meta subdirectory not found, checking DB directory")
            meta_dir = db_dir

        if not os.path.exists(meta_dir):
            self.send_error(404, f"Meta directory not found: {meta_dir}")
            return

        meta_files = []
        try:
            # Look for meta files named by processed-count suffix
            for file in self._list_directory_within_root(meta_dir):
                if file.startswith("meta_") and file.endswith(".txt"):
                    # Extract processed count from meta_<count>.txt
                    count_str = file[5:-4]  # Remove 'meta_' and '.txt'
                    try:
                        processed_count = int(count_str)
                        meta_files.append(
                            {
                                "processed_count": processed_count,
                                # Backward-compatible alias for older clients.
                                "generation": processed_count,
                                "filename": file,
                                "path": os.path.join(meta_dir, file),
                            }
                        )
                    except ValueError:
                        # Skip files that don't have valid numeric suffixes
                        continue

            # Sort by processed count
            meta_files.sort(key=lambda x: x["processed_count"])

            print(f"[SERVER] Found {len(meta_files)} meta files")
            self.send_json_response(meta_files)

        except PathValidationError:
            raise
        except Exception as e:
            print(f"[SERVER] Error listing meta files: {e}")
            self.send_error(500, f"Error listing meta files: {str(e)}")

    def handle_get_meta_content(self, db_path: str, processed_count: str):
        """Get the content of a specific meta file by processed count."""
        print(
            f"[SERVER] Fetching meta content for DB: {db_path}, "
            f"processed_count: {processed_count}"
        )

        # Get the actual database path
        actual_db_path = self._get_actual_db_path(db_path)

        # Get the directory containing the database file
        abs_db_path = os.path.join(self.search_root, actual_db_path)
        db_dir = os.path.dirname(abs_db_path)

        # Construct the meta file path - try meta subdirectory first
        meta_filename = f"meta_{processed_count}.txt"
        meta_file_path = self._resolve_within_root(
            os.path.join(db_dir, "meta", meta_filename)
        )

        # Fall back to db_dir for backward compatibility
        if not os.path.exists(meta_file_path):
            meta_file_path = self._resolve_within_root(
                os.path.join(db_dir, meta_filename)
            )

        if not os.path.exists(meta_file_path):
            self.send_error(404, f"Meta file not found: {meta_filename}")
            return

        try:
            content = self._read_text_within_root(meta_file_path)

            response_data = {
                "processed_count": int(processed_count),
                # Backward-compatible alias for older clients.
                "generation": int(processed_count),
                "filename": meta_filename,
                "content": content,
            }

            print(
                "[SERVER] Successfully served meta content for "
                f"processed_count {processed_count}"
            )
            self.send_json_response(response_data)

        except PathValidationError:
            raise
        except Exception as e:
            print(f"[SERVER] Error reading meta file: {e}")
            self.send_error(500, f"Error reading meta file: {str(e)}")

    def handle_download_meta_pdf(self, db_path: str, processed_count: str):
        """Convert a specific meta file to PDF and serve it."""
        print(
            "[SERVER] PDF download request for DB: "
            f"{db_path}, processed_count: {processed_count}"
        )

        # Get the actual database path
        actual_db_path = self._get_actual_db_path(db_path)

        # Get the directory containing the database file
        abs_db_path = os.path.join(self.search_root, actual_db_path)
        db_dir = os.path.dirname(abs_db_path)

        # Construct the meta file path - try meta subdirectory first
        meta_filename = f"meta_{processed_count}.txt"
        meta_file_path = self._resolve_within_root(
            os.path.join(db_dir, "meta", meta_filename)
        )

        # Fall back to db_dir for backward compatibility
        if not os.path.exists(meta_file_path):
            meta_file_path = self._resolve_within_root(
                os.path.join(db_dir, meta_filename)
            )

        if not os.path.exists(meta_file_path):
            self.send_error(404, f"Meta file not found: {meta_filename}")
            return

        try:
            content = self._read_text_within_root(meta_file_path)

            pdf_filename = f"meta_{processed_count}.pdf"

            # Try to generate PDF using available methods
            pdf_bytes = self._generate_pdf(content, processed_count)

            if pdf_bytes is None:
                print("[SERVER] All PDF generation methods failed, serving text")
                # Fall back to serving formatted text with PDF headers
                formatted_content = (
                    f"Meta Generation {processed_count}\n{'=' * 50}\n\n{content}"
                )
                pdf_bytes = formatted_content.encode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "application/pdf")
            self.send_header(
                "Content-Disposition", f'attachment; filename="{pdf_filename}"'
            )
            self.send_header("Content-Length", str(len(pdf_bytes)))
            self.end_headers()
            self.wfile.write(pdf_bytes)
            print(f"[SERVER] Successfully served PDF: {pdf_filename}")

        except PathValidationError:
            raise
        except Exception as e:
            print(f"[SERVER] Error converting meta file to PDF: {e}")
            self.send_error(500, f"Error converting to PDF: {str(e)}")

    def handle_get_plots(self, db_path: str, generation: str, program_id: str):
        """List available plot files for a given program."""
        print(
            f"[SERVER] Listing plots for DB: {db_path}, "
            f"gen: {generation}, program: {program_id}"
        )

        # Get the actual database path
        actual_db_path = self._get_actual_db_path(db_path)

        # Get the directory containing the database file
        abs_db_path = os.path.join(self.search_root, actual_db_path)
        db_dir = os.path.dirname(abs_db_path)

        # Construct the plots directory path
        # Structure: db_dir/gen_X/results/plots/
        plots_dir = self._resolve_within_root(
            os.path.join(db_dir, f"gen_{generation}", "results", "plots")
        )

        plot_files = []
        if os.path.exists(plots_dir):
            for filename in self._list_directory_within_root(plots_dir):
                filepath = os.path.join(plots_dir, filename)
                if os.path.isfile(filepath):
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in [".png", ".gif", ".jpg", ".jpeg"]:
                        # Create URL-safe path relative to search_root
                        rel_path = os.path.relpath(filepath, self.search_root)
                        plot_files.append(
                            {
                                "filename": filename,
                                "path": rel_path,
                                "type": "animation" if ext == ".gif" else "image",
                                "ext": ext,
                            }
                        )

            # Sort by filename
            plot_files.sort(key=lambda x: x["filename"])
            print(f"[SERVER] Found {len(plot_files)} plot files in {plots_dir}")
        else:
            print(f"[SERVER] Plots directory not found: {plots_dir}")

        self.send_json_response(plot_files)

    def handle_serve_plot_file(self):
        """Serve a plot file from the search root."""
        # Extract the file path from the URL (after /plot_file/)
        parsed_url = urllib.parse.urlparse(self.path)
        rel_path = urllib.parse.unquote(parsed_url.path[11:])  # Remove /plot_file/

        # Restrict to image files, and containment-check BEFORE touching disk so
        # out-of-root paths can't be probed for existence (403 vs 404 leak) and
        # symlinks pointing outside the root are rejected (realpath + commonpath).
        content_types = {
            ".png": "image/png",
            ".gif": "image/gif",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
        }
        ext = os.path.splitext(rel_path)[1].lower()
        if ext not in content_types:
            self.send_error(403, "Access denied")
            return
        content_type = content_types[ext]

        try:
            abs_path = self._resolve_within_root(rel_path)
        except PathValidationError as exc:
            print(f"[SERVER] Rejected plot-file traversal: {exc}")
            self.send_error(403, "Access denied")
            return

        print(f"[SERVER] Serving plot file: {abs_path}")

        if not os.path.isfile(abs_path):
            self.send_error(404, f"Plot file not found: {rel_path}")
            return

        try:
            file_data = self._read_bytes_within_root(abs_path)

            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(file_data)))
            self.send_header("Cache-Control", "max-age=3600")
            self.end_headers()
            self.wfile.write(file_data)
            print(f"[SERVER] Successfully served plot: {rel_path}")

        except PathValidationError:
            raise
        except Exception as e:
            print(f"[SERVER] Error serving plot file: {e}")
            self.send_error(500, f"Error serving file: {str(e)}")

    def handle_get_system_prompts(self, db_path: str):
        """Fetch all system prompts from the prompts.db in the same directory."""
        print(f"[SERVER] Fetching system prompts for DB: {db_path}")

        # Get the actual database path
        actual_db_path = self._get_actual_db_path(db_path)

        # Construct path to prompts.sqlite (in same directory as programs.sqlite)
        abs_db_path = os.path.join(self.search_root, actual_db_path)
        db_dir = os.path.dirname(abs_db_path)
        prompts_db_path = self._resolve_within_root(
            os.path.join(db_dir, "prompts.sqlite")
        )

        if not os.path.exists(prompts_db_path):
            print(f"[SERVER] Prompts database not found: {prompts_db_path}")
            # Return empty list if no prompts database exists
            self.send_json_response([])
            return

        # Retry logic for the reader with WAL mode support
        # Use more retries and longer delays during active evolution
        max_retries = 8
        delay = 0.2
        for i in range(max_retries):
            try:
                with self._open_prompt_database_within_root(
                    prompts_db_path
                ) as prompt_db:
                    # Set WAL mode compatible settings for read-only connections
                    # Longer busy_timeout for concurrent access during evolution
                    if prompt_db.cursor:
                        prompt_db.cursor.execute("PRAGMA busy_timeout = 30000;")

                    prompts = prompt_db.get_all_prompts()

                # Convert SystemPrompt objects to dicts for JSON
                prompts_dict = [p.to_dict() for p in prompts]

                # Debug: print first prompt's keys and program_generation
                if prompts_dict:
                    print(f"[DEBUG] First prompt keys: {list(prompts_dict[0].keys())}")
                    print(
                        f"[DEBUG] First prompt program_generation: {prompts_dict[0].get('program_generation')}"
                    )
                    print(
                        f"[DEBUG] First SystemPrompt.program_generation: {prompts[0].program_generation if prompts else 'N/A'}"
                    )

                self.send_json_response(prompts_dict)
                success_msg = (
                    f"[SERVER] Successfully served {len(prompts)} "
                    f"system prompts from {prompts_db_path} (attempt {i + 1})"
                )
                print(success_msg)
                return

            except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
                error_str = str(e).lower()
                is_retryable = (
                    "database is locked" in error_str
                    or "busy" in error_str
                    or "disk i/o error" in error_str  # Common with Dropbox/cloud sync
                )
                if is_retryable:
                    print(
                        f"[SERVER] Attempt {i + 1}/{max_retries} - prompts db error, "
                        f"retrying in {delay:.1f}s... ({e})"
                    )
                    if i < max_retries - 1:
                        time.sleep(delay)
                        delay = min(
                            delay * 1.5, 5.0
                        )  # Allow longer delays during active evolution
                        continue
                    else:
                        # Last retry failed - return empty list instead of error
                        # Prompts are optional, don't break the page
                        print(
                            f"[SERVER] Prompts DB unavailable after {max_retries} attempts, returning empty list"
                        )
                        self.send_json_response([])
                        return
                else:
                    print(f"[SERVER] Non-recoverable prompts database error: {e}")
                    # Return empty list instead of 500 - prompts are optional
                    self.send_json_response([])
                    return

            except PathValidationError:
                raise
            except Exception as e:
                print(f"[SERVER] Error fetching system prompts: {e}")
                import traceback

                traceback.print_exc()
                # Return empty list instead of 500 - prompts are optional
                self.send_json_response([])
                return
    def handle_get_database_stats(self, db_path: str):
        """Get quick aggregate stats for a database (count, best score, cost)."""
        actual_db_path = self._get_actual_db_path(db_path)
        abs_db_path = os.path.join(self.search_root, actual_db_path)

        if not os.path.exists(abs_db_path):
            self.send_json_response({"error": "not_found"})
            return

        prompts_db_path = self._resolve_within_root(
            os.path.join(os.path.dirname(abs_db_path), "prompts.sqlite")
        )

        max_retries = 3
        delay = 0.1
        for i in range(max_retries):
            stack = contextlib.ExitStack()
            conn = None
            try:
                conn = stack.enter_context(
                    self._connect_database_within_root(
                        abs_db_path,
                        timeout=5.0,
                        isolation_level=None,
                    )
                )
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("PRAGMA busy_timeout = 5000;")

                # Get aggregate stats in a single query
                # Costs are stored in metadata as: api_costs, embed_cost,
                # novelty_cost, meta_cost
                cursor.execute("""
                    SELECT
                        COUNT(*) as program_count,
                        COUNT(DISTINCT generation) as generation_count,
                        SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as correct_count,
                        MAX(
                            CASE WHEN correct = 1
                            THEN combined_score
                            ELSE NULL END
                        ) as best_score,
                        MAX(generation) as max_generation,
                        MIN(timestamp) as first_update,
                        MAX(timestamp) as last_update,
                        MIN(
                            CASE WHEN json_valid(metadata)
                            THEN json_extract(metadata, '$.pipeline_started_at')
                            ELSE NULL END
                        ) as first_pipeline_start,
                        MAX(
                            CASE WHEN json_valid(metadata)
                            THEN json_extract(metadata, '$.postprocess_finished_at')
                            ELSE NULL END
                        ) as last_postprocess_finish,
                        SUM(
                            COALESCE(
                                CASE WHEN json_valid(metadata)
                                THEN json_extract(metadata, '$.api_costs')
                                ELSE 0 END, 0
                            ) +
                            COALESCE(
                                CASE WHEN json_valid(metadata)
                                THEN json_extract(metadata, '$.embed_cost')
                                ELSE 0 END, 0
                            ) +
                            COALESCE(
                                CASE WHEN json_valid(metadata)
                                THEN json_extract(metadata, '$.novelty_cost')
                                ELSE 0 END, 0
                            ) +
                            COALESCE(
                                CASE WHEN json_valid(metadata)
                                THEN json_extract(metadata, '$.meta_cost')
                                ELSE 0 END, 0
                            )
                        ) as total_cost
                    FROM programs
                """)
                row = cursor.fetchone()

                # Get the generation where best score was achieved
                best_gen = None
                if row["best_score"] is not None:
                    cursor.execute(
                        """
                        SELECT MIN(generation) as best_gen
                        FROM programs
                        WHERE correct = 1
                          AND combined_score = ?
                    """,
                        (row["best_score"],),
                    )
                    best_row = cursor.fetchone()
                    if best_row and best_row["best_gen"] is not None:
                        best_gen = best_row["best_gen"]

                max_gen = row["max_generation"] or 0
                gens_since_improvement = (
                    max_gen - best_gen if best_gen is not None else max_gen
                )
                runtime_start = row["first_pipeline_start"]
                if runtime_start is None:
                    runtime_start = row["first_update"]
                runtime_end = row["last_postprocess_finish"]
                if runtime_end is None:
                    runtime_end = row["last_update"]
                total_runtime_seconds = None
                if runtime_start is not None and runtime_end is not None:
                    total_runtime_seconds = max(0.0, runtime_end - runtime_start)

                stats = {
                    "program_count": row["program_count"] or 0,
                    "generation_count": row["generation_count"] or 0,
                    "correct_count": row["correct_count"] or 0,
                    "best_score": row["best_score"],
                    "best_generation": best_gen,
                    "max_generation": max_gen,
                    "last_update": row["last_update"],
                    "gens_since_improvement": gens_since_improvement,
                    "total_cost": row["total_cost"] or 0,
                    "total_runtime_seconds": total_runtime_seconds,
                    "prompt_count": 0,
                    "prompt_evo_cost": 0,
                    "has_prompt_evo": False,
                }

                # Prompt WAL sidecars are pathname-specific. Release the main
                # snapshot before opening the sibling prompt database, even when
                # both names are hardlinks to the same main inode.
                stack.close()
                stack = contextlib.ExitStack()
                conn = None

                # Check for prompts.db in the same directory
                if os.path.exists(prompts_db_path):
                    try:
                        pconn = stack.enter_context(
                            self._connect_database_within_root(
                                prompts_db_path,
                                timeout=2.0,
                                isolation_level=None,
                            )
                        )
                        pconn.row_factory = sqlite3.Row
                        pcursor = pconn.cursor()
                        pcursor.execute("PRAGMA busy_timeout = 2000;")
                        pcursor.execute("""
                            SELECT COUNT(*) as prompt_count
                            FROM system_prompts
                        """)
                        prow = pcursor.fetchone()
                        stats["prompt_count"] = prow["prompt_count"] or 0
                        stats["has_prompt_evo"] = stats["prompt_count"] > 0

                        # Sum prompt evolution costs from metadata.llm.cost
                        pcursor.execute("""
                            SELECT SUM(
                                CASE WHEN json_valid(metadata)
                                THEN COALESCE(
                                    json_extract(metadata, '$.llm.cost'),
                                    0
                                )
                                ELSE 0 END
                            ) as prompt_cost
                            FROM system_prompts
                        """)
                        pcost_row = pcursor.fetchone()
                        stats["prompt_evo_cost"] = pcost_row["prompt_cost"] or 0
                    except PathValidationError:
                        raise
                    except Exception as pe:
                        print(f"[SERVER] Warning: Error reading prompts.db: {pe}")

                self.send_json_response(stats)
                return

            except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
                error_str = str(e).lower()
                if "locked" in error_str or "busy" in error_str:
                    if i < max_retries - 1:
                        time.sleep(delay)
                        delay *= 2
                        continue
                # Return empty stats on error
                self.send_json_response(
                    {
                        "program_count": 0,
                        "best_score": None,
                        "max_generation": 0,
                        "total_cost": 0,
                        "total_runtime_seconds": None,
                        "error": str(e),
                    }
                )
                return
            except PathValidationError:
                raise
            except Exception as e:
                self.send_json_response(
                    {
                        "program_count": 0,
                        "best_score": None,
                        "max_generation": 0,
                        "total_cost": 0,
                        "total_runtime_seconds": None,
                        "error": str(e),
                    }
                )
                return
            finally:
                stack.close()

    def _generate_pdf(self, content: str, generation: str) -> bytes:
        """Generate PDF from markdown content using available methods."""

        print(f"[SERVER] Attempting to generate PDF for generation {generation}")

        # Method 1: Try simple HTML to PDF using browser print
        try:
            # Preprocess content to fix line break issues
            processed_content = self._fix_line_breaks(content)

            # Convert markdown to HTML with better line break handling
            try:
                html_content = markdown.markdown(
                    processed_content,
                    extensions=["extra", "nl2br"],  # nl2br: newlines to <br>
                )
            except Exception:
                # Fallback if nl2br extension is not available
                html_content = markdown.markdown(
                    processed_content, extensions=["extra"]
                )
                # Manually convert remaining single line breaks to <br>
                html_content = html_content.replace("\n", "<br>\n")

            # Add boxes around program summaries after markdown conversion
            print(
                f"[SERVER] HTML content before boxing (first 500 chars): "
                f"{html_content[:500]}"
            )
            html_content = self._add_program_boxes_html(html_content)
            print(
                f"[SERVER] HTML content after boxing (first 500 chars): "
                f"{html_content[:500]}"
            )

            # Get the logo as base64
            logo_data_uri = self._get_logo_base64()

            # Create a well-formatted HTML document
            html_full = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Meta Generation {generation}</title>
    <style>
        @media print {{
            @page {{ margin: 2cm; size: A4; }}
            body {{ font-size: 12pt; }}
        }}
        body {{ 
            font-family: 'Times New Roman', Times, serif; 
            line-height: 1.6; 
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1 {{ 
            color: #2c3e50; 
            border-bottom: 2px solid #e74c3c;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        h2, h3 {{ 
            color: #2c3e50; 
            margin-top: 1.5em;
            margin-bottom: 0.5em;
        }}
        pre {{ 
            background-color: #f8f9fa; 
            padding: 15px; 
            border-radius: 5px; 
            overflow-x: auto;
            border: 1px solid #e9ecef;
            font-family: 'Courier New', monospace;
            font-size: 11pt;
        }}
        code {{ 
            background-color: #f8f9fa; 
            padding: 2px 4px; 
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 90%;
        }}
        blockquote {{ 
            border-left: 4px solid #e74c3c; 
            margin: 1em 0; 
            padding-left: 1em;
            color: #6c757d;
            font-style: italic;
        }}
        p {{ 
            margin-bottom: 1em; 
            line-height: 1.6;
            text-align: justify;
        }}
        ul, ol {{ margin-bottom: 1em; }}
        li {{ 
            margin-bottom: 0.5em; 
            line-height: 1.5;
        }}
        br {{ 
            line-height: 1.8; 
        }}
        /* Improve spacing for specific content types */
        strong {{ 
            font-weight: bold; 
            color: #2c3e50;
        }}
        em {{ 
            font-style: italic; 
            color: #34495e;
        }}
        /* Header with centered logo styling */
        .header-container {{
            text-align: center;
            margin-bottom: 2em;
            padding-bottom: 1em;
            border-bottom: 2px solid #e74c3c;
        }}
        .header-logo {{
            width: 150px;
            height: 150px;
            margin: 0 auto 15px auto;
            display: block;
        }}
        .header-title {{
            margin: 0;
            color: #2c3e50;
            font-size: 24pt;
            font-weight: bold;
            text-align: center;
        }}
        /* Program summary boxes */
        .program-box {{
            border: 2px solid #e74c3c;
            border-radius: 10px;
            margin: 0.8em 0;
            padding: 0.1em 0.8em;
            background-color: #f8f9fa;
            page-break-inside: avoid;
        }}
        .program-name {{
            font-weight: bold;
            color: #2c3e50;
            font-size: 16pt;
            margin-bottom: 1em;
            border-bottom: 1px solid #bdc3c7;
            padding-bottom: 0.5em;
        }}
        .program-field {{
            margin-top: 1em;
            margin-bottom: 0.5em;
        }}
        .program-field strong {{
            color: #34495e;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="header-container">
        {f'<img src="{logo_data_uri}" alt="Shinka Logo" class="header-logo">' if logo_data_uri else ""}
        <h1 class="header-title">ShinkaEvolve Meta-Scratchpad: \
{generation}</h1>
    </div>
    {html_content}
</body>
</html>"""

            # Try wkhtmltopdf if available
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".html", delete=False
                ) as html_file:
                    html_file.write(html_full)
                    html_file_path = html_file.name

                with tempfile.NamedTemporaryFile(
                    suffix=".pdf", delete=False
                ) as pdf_file:
                    pdf_file_path = pdf_file.name

                # Try wkhtmltopdf directly
                result = subprocess.run(
                    [
                        "wkhtmltopdf",
                        "--page-size",
                        "A4",
                        "--margin-top",
                        "20mm",
                        "--margin-bottom",
                        "20mm",
                        "--margin-left",
                        "20mm",
                        "--margin-right",
                        "20mm",
                        html_file_path,
                        pdf_file_path,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                if result.returncode == 0:
                    with open(pdf_file_path, "rb") as f:
                        pdf_bytes = f.read()
                    print("[SERVER] PDF generated successfully using wkhtmltopdf")
                    return pdf_bytes
                else:
                    print(f"[SERVER] wkhtmltopdf failed: {result.stderr}")

            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                print(f"[SERVER] wkhtmltopdf not available: {e}")
            finally:
                # Clean up temp files
                try:
                    os.unlink(html_file_path)
                    os.unlink(pdf_file_path)
                except (NameError, OSError):
                    pass

            # Try pandoc as fallback
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".html", delete=False
                ) as html_file:
                    html_file.write(html_full)
                    html_file_path = html_file.name

                with tempfile.NamedTemporaryFile(
                    suffix=".pdf", delete=False
                ) as pdf_file:
                    pdf_file_path = pdf_file.name

                result = subprocess.run(
                    ["pandoc", html_file_path, "-o", pdf_file_path],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                if result.returncode == 0:
                    with open(pdf_file_path, "rb") as f:
                        pdf_bytes = f.read()
                    print("[SERVER] PDF generated successfully using pandoc")
                    return pdf_bytes
                else:
                    print(f"[SERVER] pandoc failed: {result.stderr}")

            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                print(f"[SERVER] pandoc not available: {e}")
            finally:
                # Clean up temp files
                try:
                    os.unlink(html_file_path)
                    os.unlink(pdf_file_path)
                except (NameError, OSError):
                    pass

        except Exception as e:
            print(f"[SERVER] HTML generation failed: {e}")

        print("[SERVER] All PDF generation methods failed")
        return None

    def _fix_line_breaks(self, content: str) -> str:
        """Fix line breaks in markdown content for better PDF rendering."""

        # Simple approach: ensure proper paragraph breaks
        # Replace single newlines that should be paragraph breaks with
        # double newlines

        # First, normalize line endings
        content = content.replace("\r\n", "\n").replace("\r", "\n")

        # Split into lines
        lines = content.split("\n")
        result_lines = []

        i = 0
        while i < len(lines):
            current_line = lines[i].strip()

            # Always add the current line
            result_lines.append(current_line)

            # Look ahead to see if we need to add extra spacing
            if i < len(lines) - 1:
                next_line = lines[i + 1].strip()

                # Add extra line break for paragraph separation if:
                # 1. Current line has substantial content
                # 2. Next line starts a new thought (capital letter)
                # 3. Neither line is a markdown special element
                if (
                    current_line
                    and next_line
                    and len(current_line) > 30  # Substantial content
                    and current_line.endswith((".", "!", "?", ";"))  # Sentence ending
                    and next_line[0].isupper()  # Next starts with capital
                    and not next_line.startswith(
                        ("#", "-", "*", "+")
                    )  # Not markdown list/header
                    and not re.match(r"^\*\*\w+:\*\*", next_line)
                ):  # Not bold field
                    result_lines.append("")  # Add blank line

            i += 1

        return "\n".join(result_lines)

    def _add_program_boxes_html(self, html_content: str) -> str:
        """Add HTML boxes around program summaries in converted HTML."""

        # Match entire <p> tags that contain program summaries
        # Pattern matches <p> tags that start with <strong>Program Name:
        program_pattern = r"(<p><strong>Program Name:[^<]*</strong>[\s\S]*?</p>)"

        def wrap_program_html(match):
            program_html = match.group(1).strip()
            return f'<div class="program-box">{program_html}</div>'

        # Replace all program summaries with boxed versions
        result = re.sub(
            program_pattern,
            wrap_program_html,
            html_content,
            flags=re.MULTILINE | re.DOTALL,
        )

        return result

    def _get_logo_base64(self) -> str:
        """Get the Shinka logo as base64 data URI."""
        try:
            # Look for favicon.png in the main shinka package directory
            logo_path = os.path.join(os.path.dirname(__file__), "favicon.png")
            if os.path.exists(logo_path):
                with open(logo_path, "rb") as f:
                    logo_data = f.read()
                encoded = base64.b64encode(logo_data).decode("utf-8")
                return f"data:image/png;base64,{encoded}"
        except Exception as e:
            print(f"[SERVER] Could not load logo: {e}")
        return ""

    def send_json_response(self, data):
        """Helper to send a JSON response."""
        # Clean NaN/Inf values before serializing (Python's json outputs invalid JSON for these)
        clean_data = self._clean_nan_values(data)
        payload = json.dumps(clean_data, default=self._json_encoder).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        # No "Access-Control-Allow-Origin: *": the UI is served same-origin, and
        # a wildcard would let any site the operator visits read the full DB.
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _clean_nan_values(self, obj):
        """Recursively replace NaN and Inf float values with None."""
        import math

        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        elif isinstance(obj, dict):
            return {k: self._clean_nan_values(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_nan_values(item) for item in obj]
        else:
            return obj

    def _json_encoder(self, obj):
        """Custom JSON encoder to handle non-serializable types."""
        import math

        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def create_handler_factory(search_root, allowed_hosts=...):
    """Create a handler factory that passes the search root to handler."""

    canonical_search_root = os.path.realpath(search_root)
    search_root_descriptor = None
    if DatabaseRequestHandler._supports_descriptor_traversal():
        search_root_descriptor = DatabaseRequestHandler._open_search_root_descriptor(
            canonical_search_root
        )

    def handler_factory(*args, **kwargs):
        return DatabaseRequestHandler(
            *args,
            search_root=search_root,
            canonical_search_root=canonical_search_root,
            search_root_descriptor=search_root_descriptor,
            allowed_hosts=allowed_hosts,
            **kwargs,
        )

    if search_root_descriptor is not None:
        weakref.finalize(handler_factory, os.close, search_root_descriptor)
    return handler_factory


def _is_loopback_host(host: str) -> bool:
    if host.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _allowed_hosts_for_bind(
    requested_host: str, bound_host: str
) -> Optional[frozenset[str]]:
    if not _is_loopback_host(bound_host):
        return None

    allowed_hosts = set(DatabaseRequestHandler.allowed_hosts)
    for host in (requested_host, bound_host):
        normalized_host = host.casefold()
        for allowed_host in {
            normalized_host,
            normalized_host.replace("%", "%25"),
        }:
            allowed_hosts.add(allowed_host)
            if ":" in allowed_host:
                allowed_hosts.add(f"[{allowed_host}]")
    return frozenset(allowed_hosts)


def _hostname_from_host_header(host_header: str) -> Optional[str]:
    authority = host_header.strip()
    if not authority:
        return None

    if authority.startswith("["):
        bracket = authority.find("]")
        if bracket < 0:
            return None
        hostname = authority[: bracket + 1]
        remainder = authority[bracket + 1 :]
        if remainder and not (
            remainder.startswith(":") and remainder[1:].isdigit()
        ):
            return None
        return hostname.casefold()

    if authority.count(":") > 1:
        return None
    hostname, separator, port = authority.partition(":")
    if not hostname or (separator and not port.isdigit()):
        return None
    return hostname.casefold()


class _ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


class _ReusableTCPServer6(_ReusableTCPServer):
    address_family = socket.AF_INET6


def _server_class_for_family(family: int) -> type[socketserver.TCPServer]:
    return _ReusableTCPServer6 if family == socket.AF_INET6 else _ReusableTCPServer


def _bind_server(
    host: str,
    port: int,
    request_handler: Callable[..., DatabaseRequestHandler],
) -> socketserver.TCPServer:
    addresses = socket.getaddrinfo(
        host or None,
        port,
        socket.AF_UNSPEC if host else socket.AF_INET,
        socket.SOCK_STREAM,
        flags=socket.AI_PASSIVE if not host else 0,
    )
    bind_errors = []
    attempted_addresses = set()
    for family, _socket_type, _protocol, _canonical_name, address in addresses:
        if family not in (socket.AF_INET, socket.AF_INET6):
            continue
        candidate = (family, address)
        if candidate in attempted_addresses:
            continue
        attempted_addresses.add(candidate)
        server_class = _server_class_for_family(family)
        try:
            return server_class(address, request_handler)
        except OSError as exc:
            bind_errors.append(exc)

    if bind_errors:
        raise bind_errors[-1]
    raise OSError(f"No usable address found for host {host!r}")


def _bound_host_from_server_address(server_address: tuple[Any, ...]) -> str:
    bound_host = str(server_address[0])
    if "%" in bound_host or len(server_address) < 4:
        return bound_host

    scope_id = int(server_address[3])
    if not scope_id:
        return bound_host
    try:
        scope = socket.if_indextoname(scope_id)
    except (AttributeError, OSError):
        scope = str(scope_id)
    return f"{bound_host}%{scope}"


def _browser_url(
    bound_host: str,
    port: int,
    db_path: Optional[str] = None,
) -> str:
    try:
        bound_is_wildcard = ipaddress.ip_address(bound_host).is_unspecified
    except ValueError:
        bound_is_wildcard = not bound_host

    if bound_is_wildcard:
        browser_host = "::1" if ":" in bound_host else "127.0.0.1"
    else:
        browser_host = bound_host
    if ":" in browser_host:
        browser_host = f"[{browser_host.replace('%', '%25')}]"

    path = "/"
    if db_path:
        path = "/viz_tree.html?" + urllib.parse.urlencode({"db_path": db_path})
    return f"http://{browser_host}:{port}{path}"


def start_server(
    port: int,
    search_root: str,
    db_path: Optional[str] = None,
    host: str = "127.0.0.1",
    on_ready: Optional[Callable[[socketserver.TCPServer], None]] = None,
):
    """Start the HTTP server.

    Binds to 127.0.0.1 by default so the evolution database is not exposed to
    the local network. Pass an explicit ``host`` (e.g. "0.0.0.0") to expose it,
    which also relaxes the DNS-rebinding Host-header check.
    """
    # Change to the webui directory inside the shinka package to serve static files
    webui_dir = os.path.dirname(__file__)
    webui_dir = os.path.abspath(webui_dir)

    if not os.path.exists(webui_dir):
        raise FileNotFoundError(f"Webui directory not found: {webui_dir}")

    os.chdir(webui_dir)
    print(f"[DEBUG] Server root directory: {webui_dir}")
    print(f"[DEBUG] Search root directory: {search_root}")

    # On a loopback bind, enforce the Host-header allowlist (anti DNS-rebinding).
    # On an explicit external bind the operator has opted in, so disable it.
    httpd = _bind_server(host, port, DatabaseRequestHandler)
    with httpd:
        bound_host = _bound_host_from_server_address(httpd.server_address)
        allowed_hosts = _allowed_hosts_for_bind(host, bound_host)
        httpd.RequestHandlerClass = create_handler_factory(
            search_root, allowed_hosts=allowed_hosts
        )
        bound_port = int(httpd.server_address[1])
        display_url = _browser_url(bound_host, bound_port)
        msg = f"\n[*] Serving {display_url}  (Ctrl+C to stop)"
        if not _is_loopback_host(bound_host):
            msg += (
                "\n[!] Bound to a non-loopback address: the evolution database "
                "is reachable from the network with no authentication."
            )
        if not DatabaseRequestHandler._supports_descriptor_traversal():
            msg += (
                "\n[!] This platform lacks descriptor-relative file access; "
                "search-root race hardening is unavailable."
            )
        print(msg)
        if on_ready is not None:
            def notify_ready() -> None:
                try:
                    on_ready(httpd)
                except Exception as exc:
                    print(f"[SERVER] Readiness callback failed: {exc}")

            threading.Thread(target=notify_ready, daemon=True).start()
        try:
            httpd.serve_forever()
        finally:
            DatabaseRequestHandler.clear_program_response_cache(search_root)
            DatabaseRequestHandler.clear_database_snapshot_cache()


def main():
    """Main entry point for shinka_visualize command."""
    description = "Serve the Shinka visualization UI for evolution results."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "root_directory",
        nargs="?",
        default=os.getcwd(),
        help=(
            "Root directory to search for database files "
            "(default: current working directory)"
        ),
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Port to listen on (default: 8000).",
    )
    parser.add_argument(
        "--open",
        dest="open_browser",
        action="store_true",
        help="Open browser on the local machine (if DISPLAY is set)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Path to a specific database file to serve.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help=(
            "Address to bind (default: 127.0.0.1, local only). Use 0.0.0.0 to "
            "expose on the network — this serves the database with no auth."
        ),
    )
    args = parser.parse_args()

    # Resolve the root directory to an absolute path
    search_root = os.path.abspath(args.root_directory)

    if not os.path.exists(search_root):
        print(f"Error: Root directory does not exist: {search_root}")
        sys.exit(1)

    print(f"[INFO] Searching for databases in: {search_root}")

    def announce_ready(httpd: socketserver.TCPServer) -> None:
        bound_host = _bound_host_from_server_address(httpd.server_address)
        bound_port = int(httpd.server_address[1])
        viz_url = _browser_url(bound_host, bound_port, args.db)

        if args.open_browser:
            try:
                webbrowser.open_new_tab(viz_url)
                print(f"→ Opening {viz_url} in browser")
            except Exception as e:
                print(f"→ Could not open browser automatically: {e}")
                print(f"→ Visit {viz_url}")
        else:
            print(f"→ Visit {viz_url}")
            print("(remember to forward the port if this is a remote host)")

    try:
        start_server(
            args.port,
            search_root,
            args.db,
            args.host,
            on_ready=announce_ready,
        )
    except KeyboardInterrupt:
        print("\n[*] Shutting down.")


if __name__ == "__main__":
    main()
