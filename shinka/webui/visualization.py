#!/usr/bin/env python3
"""
Shinka Visualization Module

This module provides visualization capabilities for Shinka evolution results.
It serves a web interface for exploring evolution databases and meta files.
"""

import argparse
import base64
import http.server
import json
import markdown
import os
import re
import socketserver
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import webbrowser
from typing import Optional, Dict, Any, Tuple

from shinka.database import DatabaseConfig, ProgramDatabase
from shinka.database import SystemPromptConfig, SystemPromptDatabase

# We'll use a simple text-to-PDF approach instead of complex dependencies
WEASYPRINT_AVAILABLE = False

DEFAULT_PORT = 8000
CACHE_EXPIRATION_SECONDS = 5  # Cache data for 5 seconds
db_cache: Dict[str, Tuple[float, Any]] = {}


class DatabaseRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, search_root=None, **kwargs):
        self.search_root = search_root or os.getcwd()
        super().__init__(*args, **kwargs)

    def log_message(self, format, *args):
        """Override to provide more detailed logging."""
        print(f"\n[SERVER] {format % args}")

    def do_GET(self):
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

        if path == "/get_meta_content" and "db_path" in query and "generation" in query:
            db_path = query["db_path"][0]
            generation = query["generation"][0]
            return self.handle_get_meta_content(db_path, generation)

        if (
            path == "/download_meta_pdf"
            and "db_path" in query
            and "generation" in query
        ):
            db_path = query["db_path"][0]
            generation = query["generation"][0]
            return self.handle_download_meta_pdf(db_path, generation)

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

        if os.path.exists(self.search_root):
            print(f"[SERVER] Scanning for .db files in: {self.search_root}")
            for root, _, files in os.walk(self.search_root):
                for f in files:
                    # Only list program databases, not prompt databases
                    if f.lower() in ("prompts.db", "prompts.sqlite"):
                        continue
                    if f.lower().endswith((".db", ".sqlite")):
                        full_path = os.path.join(root, f)
                        client_path = os.path.relpath(full_path, self.search_root)

                        # Parse path components
                        path_parts = client_path.split(os.sep)

                        # Extract result name (full path to db directory)
                        # e.g., results_coral/aes_block_encrypt/triton
                        display_name = (
                            "/".join(path_parts[:-1])
                            if len(path_parts) >= 2
                            else client_path
                        )

                        # Extract task name (first path component only)
                        # e.g., results_coral
                        if len(path_parts) >= 2:
                            task = path_parts[0]
                        else:
                            task = task_name

                        # Extract date for sorting
                        sort_key = "0"  # Default for paths without a date
                        match = date_pattern.search(client_path)
                        if match:
                            sort_key = match.group(1)

                        db_info = {
                            "path": client_path,
                            "name": display_name,
                            "task": task,
                            "sort_key": sort_key,
                            "actual_path": client_path,
                        }
                        db_files.append(db_info)
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

    def _get_actual_db_path(self, db_path: str) -> str:
        """Convert db_path to the actual file path (now a no-op since path is not modified)."""
        return db_path

    def handle_get_programs(self, db_path: str):
        """Fetch all programs from a given database file."""
        print(f"[SERVER] Fetching programs from DB: {db_path}")

        # Handle the case where db_path might have the task name prepended
        # Extract the actual path by removing the task name prefix if present
        actual_db_path = self._get_actual_db_path(db_path)

        # Check cache first
        if db_path in db_cache:
            last_fetch_time, cached_data = db_cache[db_path]
            if time.time() - last_fetch_time < CACHE_EXPIRATION_SECONDS:
                print(f"[SERVER] Serving from cache for DB: {db_path}")
                self.send_json_response(cached_data)
                return

        # Construct absolute path to the database from search root using actual path
        abs_db_path = os.path.join(self.search_root, actual_db_path)
        print(f"[SERVER] Absolute DB path: {abs_db_path} (from {db_path})")

        if not os.path.exists(abs_db_path):
            self.send_error(404, f"Database file not found: {actual_db_path}")
            return

        # Retry logic for the reader with improved WAL mode support
        # More retries with longer delays during active evolution
        max_retries = 8
        delay = 0.2
        for i in range(max_retries):
            db = None
            try:
                config = DatabaseConfig(db_path=abs_db_path)
                db = ProgramDatabase(config, read_only=True)

                # Set WAL mode compatible settings for read-only connections
                # Longer busy_timeout for concurrent access during evolution
                if db.cursor:
                    db.cursor.execute(
                        "PRAGMA busy_timeout = 30000;"
                    )  # 30 second timeout
                    db.cursor.execute("PRAGMA journal_mode = WAL;")  # Ensure WAL mode

                programs = db.get_all_programs()

                # Convert Program objects to dicts for JSON
                programs_dict = [p.to_dict() for p in programs]

                # Update cache
                db_cache[db_path] = (time.time(), programs_dict)

                self.send_json_response(programs_dict)
                success_msg = (
                    f"[SERVER] Successfully served {len(programs)} "
                    f"programs from {db_path} (attempt {i + 1})"
                )
                print(success_msg)
                return  # Success, exit the retry loop

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

            except Exception as e:
                # Catch any other unexpected errors
                print(f"[SERVER] An unexpected error occurred: {e}")
                self.send_error(500, f"An unexpected error occurred: {str(e)}")
                return  # Don't retry on unknown errors
            finally:
                # Ensure database connection is properly closed
                if db and hasattr(db, "close"):
                    try:
                        db.close()
                    except Exception as e:
                        print(f"[SERVER] Warning: Error closing database: {e}")

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
            db = None
            try:
                config = DatabaseConfig(db_path=abs_db_path)
                db = ProgramDatabase(config, read_only=True)

                if db.cursor:
                    db.cursor.execute("PRAGMA busy_timeout = 30000;")
                    db.cursor.execute("PRAGMA journal_mode = WAL;")

                summaries = db.get_programs_summary()
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

            except Exception as e:
                print(f"[SERVER] Error fetching program summaries: {e}")
                self.send_error(500, f"Error: {str(e)}")
                return
            finally:
                if db and hasattr(db, "close"):
                    try:
                        db.close()
                    except Exception:
                        pass

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
            db = None
            try:
                config = DatabaseConfig(db_path=abs_db_path)
                db = ProgramDatabase(config, read_only=True)

                if db.cursor:
                    db.cursor.execute("PRAGMA busy_timeout = 30000;")

                result = db.get_program_count_and_timestamp()
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

            except Exception as e:
                print(f"[SERVER] Error fetching program count: {e}")
                self.send_error(500, f"Error: {str(e)}")
                return
            finally:
                if db and hasattr(db, "close"):
                    try:
                        db.close()
                    except Exception:
                        pass

    def handle_get_program_details(self, db_path: str, program_id: str):
        """Get full details for a single program (including code and embeddings)."""
        print(f"[SERVER] Fetching program details for ID: {program_id}")

        actual_db_path = self._get_actual_db_path(db_path)
        abs_db_path = os.path.join(self.search_root, actual_db_path)

        if not os.path.exists(abs_db_path):
            self.send_error(404, f"Database file not found: {actual_db_path}")
            return

        max_retries = 8
        delay = 0.2
        for i in range(max_retries):
            db = None
            try:
                config = DatabaseConfig(db_path=abs_db_path)
                db = ProgramDatabase(config, read_only=True)

                if db.cursor:
                    db.cursor.execute("PRAGMA busy_timeout = 30000;")

                program = db.get(program_id)
                if program is None:
                    self.send_error(404, f"Program not found: {program_id}")
                    return

                self.send_json_response(program.to_dict())
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

            except Exception as e:
                print(f"[SERVER] Error fetching program details: {e}")
                self.send_error(500, f"Error: {str(e)}")
                return
            finally:
                if db and hasattr(db, "close"):
                    try:
                        db.close()
                    except Exception:
                        pass

    def handle_get_meta_files(self, db_path: str):
        """List available meta_{gen}.txt files for a given database."""
        print(f"[SERVER] Listing meta files for DB: {db_path}")

        # Get the actual database path
        actual_db_path = self._get_actual_db_path(db_path)

        # Get the directory containing the database file
        abs_db_path = os.path.join(self.search_root, actual_db_path)
        db_dir = os.path.dirname(abs_db_path)

        # Look in the meta subdirectory
        meta_dir = os.path.join(db_dir, "meta")

        if not os.path.exists(meta_dir):
            # Fall back to looking in the db_dir for backward compatibility
            print(f"[SERVER] Meta subdirectory not found, checking DB directory")
            meta_dir = db_dir

        if not os.path.exists(meta_dir):
            self.send_error(404, f"Meta directory not found: {meta_dir}")
            return

        meta_files = []
        try:
            # Look for meta_{gen}.txt files in the meta directory
            for file in os.listdir(meta_dir):
                if file.startswith("meta_") and file.endswith(".txt"):
                    # Extract generation number
                    gen_str = file[5:-4]  # Remove 'meta_' and '.txt'
                    try:
                        generation = int(gen_str)
                        meta_files.append(
                            {
                                "generation": generation,
                                "filename": file,
                                "path": os.path.join(meta_dir, file),
                            }
                        )
                    except ValueError:
                        # Skip files that don't have valid generation numbers
                        continue

            # Sort by generation number
            meta_files.sort(key=lambda x: x["generation"])

            print(f"[SERVER] Found {len(meta_files)} meta files")
            self.send_json_response(meta_files)

        except Exception as e:
            print(f"[SERVER] Error listing meta files: {e}")
            self.send_error(500, f"Error listing meta files: {str(e)}")

    def handle_get_meta_content(self, db_path: str, generation: str):
        """Get the content of a specific meta_{gen}.txt file."""
        print(
            f"[SERVER] Fetching meta content for DB: {db_path}, "
            f"generation: {generation}"
        )

        # Get the actual database path
        actual_db_path = self._get_actual_db_path(db_path)

        # Get the directory containing the database file
        abs_db_path = os.path.join(self.search_root, actual_db_path)
        db_dir = os.path.dirname(abs_db_path)

        # Construct the meta file path - try meta subdirectory first
        meta_filename = f"meta_{generation}.txt"
        meta_file_path = os.path.join(db_dir, "meta", meta_filename)

        # Fall back to db_dir for backward compatibility
        if not os.path.exists(meta_file_path):
            meta_file_path = os.path.join(db_dir, meta_filename)

        if not os.path.exists(meta_file_path):
            self.send_error(404, f"Meta file not found: {meta_filename}")
            return

        try:
            with open(meta_file_path, "r", encoding="utf-8") as f:
                content = f.read()

            response_data = {
                "generation": int(generation),
                "filename": meta_filename,
                "content": content,
            }

            print(
                f"[SERVER] Successfully served meta content for generation {generation}"
            )
            self.send_json_response(response_data)

        except Exception as e:
            print(f"[SERVER] Error reading meta file: {e}")
            self.send_error(500, f"Error reading meta file: {str(e)}")

    def handle_download_meta_pdf(self, db_path: str, generation: str):
        """Convert a specific meta_{gen}.txt file to PDF and serve it."""
        print(
            f"[SERVER] PDF download request for DB: {db_path}, generation: {generation}"
        )

        # Get the actual database path
        actual_db_path = self._get_actual_db_path(db_path)

        # Get the directory containing the database file
        abs_db_path = os.path.join(self.search_root, actual_db_path)
        db_dir = os.path.dirname(abs_db_path)

        # Construct the meta file path - try meta subdirectory first
        meta_filename = f"meta_{generation}.txt"
        meta_file_path = os.path.join(db_dir, "meta", meta_filename)

        # Fall back to db_dir for backward compatibility
        if not os.path.exists(meta_file_path):
            meta_file_path = os.path.join(db_dir, meta_filename)

        if not os.path.exists(meta_file_path):
            self.send_error(404, f"Meta file not found: {meta_filename}")
            return

        try:
            with open(meta_file_path, "r", encoding="utf-8") as f:
                content = f.read()

            pdf_filename = f"meta_{generation}.pdf"

            # Try to generate PDF using available methods
            pdf_bytes = self._generate_pdf(content, generation)

            if pdf_bytes is None:
                print("[SERVER] All PDF generation methods failed, serving text")
                # Fall back to serving formatted text with PDF headers
                formatted_content = (
                    f"Meta Generation {generation}\n{'=' * 50}\n\n{content}"
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
        plots_dir = os.path.join(db_dir, f"gen_{generation}", "results", "plots")

        plot_files = []
        if os.path.exists(plots_dir):
            for filename in os.listdir(plots_dir):
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

        abs_path = os.path.join(self.search_root, rel_path)
        print(f"[SERVER] Serving plot file: {abs_path}")

        if not os.path.exists(abs_path):
            self.send_error(404, f"Plot file not found: {rel_path}")
            return

        # Security check: ensure the path is within search_root
        abs_path = os.path.abspath(abs_path)
        abs_search_root = os.path.abspath(self.search_root)
        if not abs_path.startswith(abs_search_root):
            self.send_error(403, "Access denied")
            return

        # Determine content type
        ext = os.path.splitext(abs_path)[1].lower()
        content_types = {
            ".png": "image/png",
            ".gif": "image/gif",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
        }
        content_type = content_types.get(ext, "application/octet-stream")

        try:
            with open(abs_path, "rb") as f:
                file_data = f.read()

            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(file_data)))
            self.send_header("Cache-Control", "max-age=3600")
            self.end_headers()
            self.wfile.write(file_data)
            print(f"[SERVER] Successfully served plot: {rel_path}")

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
        prompts_db_path = os.path.join(db_dir, "prompts.sqlite")

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
            prompt_db = None
            try:
                config = SystemPromptConfig(db_path=prompts_db_path)
                prompt_db = SystemPromptDatabase(config, read_only=True)

                # Set WAL mode compatible settings for read-only connections
                # Longer busy_timeout for concurrent access during evolution
                if prompt_db.cursor:
                    prompt_db.cursor.execute("PRAGMA busy_timeout = 30000;")
                    prompt_db.cursor.execute("PRAGMA journal_mode = WAL;")

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

            except Exception as e:
                print(f"[SERVER] Error fetching system prompts: {e}")
                import traceback

                traceback.print_exc()
                # Return empty list instead of 500 - prompts are optional
                self.send_json_response([])
                return
            finally:
                if prompt_db and hasattr(prompt_db, "close"):
                    try:
                        prompt_db.close()
                    except Exception as e:
                        print(f"[SERVER] Warning: Error closing prompts database: {e}")

    def handle_get_database_stats(self, db_path: str):
        """Get quick aggregate stats for a database (count, best score, cost)."""
        actual_db_path = self._get_actual_db_path(db_path)
        abs_db_path = os.path.join(self.search_root, actual_db_path)

        if not os.path.exists(abs_db_path):
            self.send_json_response({"error": "not_found"})
            return

        max_retries = 3
        delay = 0.1
        for i in range(max_retries):
            conn = None
            try:
                conn = sqlite3.connect(abs_db_path, timeout=5.0, isolation_level=None)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("PRAGMA busy_timeout = 5000;")

                # Get aggregate stats in a single query
                # Costs are stored in metadata as: api_costs, embed_cost,
                # novelty_cost, meta_cost
                cursor.execute("""
                    SELECT
                        COUNT(*) as program_count,
                        SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as correct_count,
                        MAX(combined_score) as best_score,
                        MAX(generation) as max_generation,
                        MAX(timestamp) as last_update,
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
                best_gen = row["max_generation"] or 0
                if row["best_score"] is not None:
                    cursor.execute(
                        """
                        SELECT MIN(generation) as best_gen
                        FROM programs
                        WHERE combined_score = ?
                    """,
                        (row["best_score"],),
                    )
                    best_row = cursor.fetchone()
                    if best_row and best_row["best_gen"] is not None:
                        best_gen = best_row["best_gen"]

                max_gen = row["max_generation"] or 0
                gens_since_improvement = max_gen - best_gen

                stats = {
                    "program_count": row["program_count"] or 0,
                    "correct_count": row["correct_count"] or 0,
                    "best_score": row["best_score"],
                    "max_generation": max_gen,
                    "last_update": row["last_update"],
                    "gens_since_improvement": gens_since_improvement,
                    "total_cost": row["total_cost"] or 0,
                    "prompt_count": 0,
                    "prompt_evo_cost": 0,
                    "has_prompt_evo": False,
                }

                # Check for prompts.db in the same directory
                db_dir = os.path.dirname(abs_db_path)
                prompts_db_path = os.path.join(db_dir, "prompts.sqlite")
                if os.path.exists(prompts_db_path):
                    try:
                        pconn = sqlite3.connect(
                            prompts_db_path, timeout=2.0, isolation_level=None
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
                        pconn.close()
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
                        "error": str(e),
                    }
                )
                return
            except Exception as e:
                self.send_json_response(
                    {
                        "program_count": 0,
                        "best_score": None,
                        "max_generation": 0,
                        "total_cost": 0,
                        "error": str(e),
                    }
                )
                return
            finally:
                if conn:
                    try:
                        conn.close()
                    except Exception:
                        pass

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
        self.send_header("Access-Control-Allow-Origin", "*")
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


def create_handler_factory(search_root):
    """Create a handler factory that passes the search root to handler."""

    def handler_factory(*args, **kwargs):
        return DatabaseRequestHandler(*args, search_root=search_root, **kwargs)

    return handler_factory


def start_server(port: int, search_root: str, db_path: Optional[str] = None):
    """Start the HTTP server."""
    # Change to the webui directory inside the shinka package to serve static files
    webui_dir = os.path.dirname(__file__)
    webui_dir = os.path.abspath(webui_dir)

    if not os.path.exists(webui_dir):
        raise FileNotFoundError(f"Webui directory not found: {webui_dir}")

    os.chdir(webui_dir)
    print(f"[DEBUG] Server root directory: {webui_dir}")
    print(f"[DEBUG] Search root directory: {search_root}")

    # Create handler factory with search root
    handler_factory = create_handler_factory(search_root)

    # Reuse the socket so you can restart quickly
    class ReusableTCPServer(socketserver.TCPServer):
        allow_reuse_address = True

    with ReusableTCPServer(("", port), handler_factory) as httpd:
        msg = f"\n[*] Serving http://0.0.0.0:{port}  (Ctrl+C to stop)"
        print(msg)
        httpd.serve_forever()


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
    args = parser.parse_args()

    # Resolve the root directory to an absolute path
    search_root = os.path.abspath(args.root_directory)

    if not os.path.exists(search_root):
        print(f"Error: Root directory does not exist: {search_root}")
        sys.exit(1)

    print(f"[INFO] Searching for databases in: {search_root}")

    # Kick off the HTTP server in a daemon thread.
    server_thread = threading.Thread(
        target=start_server,
        args=(args.port, search_root, args.db),
        daemon=True,
    )
    server_thread.start()
    time.sleep(0.8)  # tiny delay so the banner prints before we continue

    # Construct URL, passing db path if provided
    if args.db:
        # If a specific DB is provided, go directly to viz_tree.html
        base_url = f"http://localhost:{args.port}/viz_tree.html"
        url_params = urllib.parse.urlencode({"db_path": args.db})
        viz_url = f"{base_url}?{url_params}"
    else:
        # Otherwise, open the landing page with all results
        viz_url = f"http://localhost:{args.port}/"

    # Try to open a browser if requested
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
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[*] Shutting down.")


if __name__ == "__main__":
    main()
