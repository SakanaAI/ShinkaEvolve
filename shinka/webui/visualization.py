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
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from shinka.database import DatabaseConfig, ProgramDatabase

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

        if path == "/":
            print("[SERVER] Root path requested, serving viz_tree.html")
            self.path = "/viz_tree.html"

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
                    if f.lower().endswith((".db", ".sqlite")):
                        full_path = os.path.join(root, f)
                        client_path = os.path.relpath(full_path, self.search_root)
                        display_name = f"{Path(f).stem} - {Path(client_path).parent}"

                        # Extract date for sorting
                        sort_key = "0"  # Default for paths without a date
                        match = date_pattern.search(client_path)
                        if match:
                            sort_key = match.group(1)

                        # Modify the path structure to include task name for proper organization
                        # If the path doesn't already have 3+ parts, prepend the task name
                        path_parts = client_path.split("/")
                        if len(path_parts) < 3:
                            # Add task name as the first part of the path
                            modified_client_path = f"{task_name}/{client_path}"
                        else:
                            modified_client_path = client_path

                        db_info = {
                            "path": modified_client_path,
                            "name": display_name,
                            "sort_key": sort_key,  # Add key for sorting
                            "actual_path": client_path,  # Keep the actual relative path for file operations
                        }
                        db_files.append(db_info)
                        print(
                            f"[SERVER] Found DB: {client_path} -> {modified_client_path} (sort: {sort_key})"
                        )

        if not db_files:
            print("[SERVER] No database files found in search directory.")

        # Sort databases by the extracted date, newest first
        db_files.sort(key=lambda x: x.get("sort_key", "0"), reverse=True)

        # Remove sort_key before sending to client (but keep actual_path)
        for db in db_files:
            del db["sort_key"]

        self.send_json_response(db_files)
        print(f"[SERVER] Served DB list with {len(db_files)} entries, sorted by date.")

    def _get_actual_db_path(self, db_path: str) -> str:
        """Convert a potentially modified db_path back to the actual file path."""
        task_name = os.path.basename(self.search_root)

        # If the path starts with the task name, remove it
        if db_path.startswith(f"{task_name}/"):
            return db_path[len(task_name) + 1 :]

        return db_path

    def handle_get_programs(self, db_path: str):
        """Fetch all programs from a given database file."""
        print(f"[SERVER] Fetching programs from DB: {db_path}")

        # Handle the case where db_path might have the task name prepended
        # Extract the actual path by removing the task name prefix if present
        actual_db_path = self._get_actual_db_path(db_path)

        # Check cache first - but reduce cache time for active databases
        # Clear cache if it's been more than 2 seconds (for active experiments)
        if db_path in db_cache:
            last_fetch_time, cached_data = db_cache[db_path]
            cache_age = time.time() - last_fetch_time
            if cache_age < 2.0:  # Reduced from 5 to 2 seconds for more frequent updates
                print(f"[SERVER] Serving from cache for DB: {db_path} (age: {cache_age:.1f}s, {len(cached_data)} programs)")
                self.send_json_response(cached_data)
                return
            else:
                # Cache expired, remove it
                del db_cache[db_path]
                print(f"[SERVER] Cache expired for DB: {db_path}, fetching fresh data")

        # Construct absolute path to the database from search root using actual path
        abs_db_path = os.path.join(self.search_root, actual_db_path)
        print(f"[SERVER] Absolute DB path: {abs_db_path} (from {db_path})")

        if not os.path.exists(abs_db_path):
            self.send_error(404, f"Database file not found: {actual_db_path}")
            return

        # Retry logic for the reader with improved WAL mode support
        max_retries = 5  # Increased retries for better resilience
        delay = 0.1  # Shorter initial delay
        for i in range(max_retries):
            db = None
            try:
                config = DatabaseConfig(db_path=abs_db_path)
                # Use read_only=True to avoid locking issues
                # SQLite URI read-only connections should see WAL data if properly configured
                db = ProgramDatabase(config, read_only=True)
                
                if db.cursor:
                    db.cursor.execute("PRAGMA busy_timeout = 10000;")
                    # Ensure WAL mode is enabled (should already be set by database)
                    db.cursor.execute("PRAGMA journal_mode = WAL;")

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
                if "database is locked" in error_str or "busy" in error_str:
                    print(
                        f"[SERVER] Attempt {i + 1}/{max_retries} - database busy, "
                        f"retrying in {delay:.1f}s... ({e})"
                    )
                    if i < max_retries - 1:
                        time.sleep(delay)
                        delay = min(delay * 1.5, 2.0)  # Exponential backoff, max 2s
                        continue
                else:
                    print(f"[SERVER] Non-recoverable database error: {e}")
                    self.send_error(500, f"Database error: {str(e)}")
                    return

                # Last retry failed
                if i == max_retries - 1:
                    err_msg = (
                        f"[SERVER] Database still busy after {max_retries} attempts"
                    )
                    print(err_msg)
                    self.send_error(
                        503,
                        "Database temporarily unavailable - evolution may be running",
                    )

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

    def handle_get_meta_files(self, db_path: str):
        """List available meta_{gen}.txt files for a given database."""
        print(f"[SERVER] Listing meta files for DB: {db_path}")

        # Get the actual database path
        actual_db_path = self._get_actual_db_path(db_path)

        # Get the directory containing the database file
        abs_db_path = os.path.join(self.search_root, actual_db_path)
        db_dir = os.path.dirname(abs_db_path)

        if not os.path.exists(db_dir):
            self.send_error(404, f"Database directory not found: {db_dir}")
            return

        meta_files = []
        try:
            # Look for meta_{gen}.txt files in the same directory as the DB
            for file in os.listdir(db_dir):
                if file.startswith("meta_") and file.endswith(".txt"):
                    # Extract generation number
                    gen_str = file[5:-4]  # Remove 'meta_' and '.txt'
                    try:
                        generation = int(gen_str)
                        meta_files.append(
                            {
                                "generation": generation,
                                "filename": file,
                                "path": os.path.join(db_dir, file),
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

        # Construct the meta file path
        meta_filename = f"meta_{generation}.txt"
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

        # Construct the meta file path
        meta_filename = f"meta_{generation}.txt"
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
        # Use custom JSON encoder to handle NaN values
        payload = json.dumps(data, default=self._json_encoder).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _json_encoder(self, obj):
        """Custom JSON encoder to handle NaN and Inf values."""
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
    base_url = f"http://localhost:{args.port}/viz_tree.html"
    if args.db:
        # URL encode the db path to handle special characters
        url_params = urllib.parse.urlencode({"db_path": args.db})
        viz_url = f"{base_url}?{url_params}"
    else:
        viz_url = base_url

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
