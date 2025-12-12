"""
Async patch application functions for non-blocking file operations.
Provides async versions of patch application and validation.
"""

import asyncio
import logging
from typing import Tuple, Optional
from pathlib import Path
from .apply_diff import apply_diff_patch
from .apply_full import apply_full_patch

try:
    import aiofiles
except ImportError:
    aiofiles = None

logger = logging.getLogger(__name__)


async def apply_patch_async(
    original_str: str,
    patch_str: str,
    patch_dir: str,
    language: str = "python",
    patch_type: str = "diff",
    verbose: bool = False,
) -> Tuple[
    Optional[str], int, Optional[str], Optional[str], Optional[str], Optional[Path]
]:
    """Async version of patch application.

    Args:
        original_str: Original code content
        patch_str: Patch content from LLM
        patch_dir: Directory to write patch files
        language: Programming language
        patch_type: Type of patch (diff, full, cross)
        verbose: Enable verbose logging

    Returns:
        Tuple of (modified_code, num_applied, output_path, error_msg, patch_txt, patch_path)
    """
    loop = asyncio.get_event_loop()

    try:
        # Create patch directory synchronously to avoid race conditions
        try:
            Path(patch_dir).mkdir(parents=True, exist_ok=True)
        except FileExistsError:
            # Another task already created it, which is fine
            pass

        # Choose the appropriate patch function
        if patch_type in ["full", "cross"]:
            patch_func = apply_full_patch
        elif patch_type == "diff":
            patch_func = apply_diff_patch
        else:
            raise ValueError(f"Unknown patch type: {patch_type}")

        # Run patch application in thread pool to avoid blocking
        result = await loop.run_in_executor(
            None,
            lambda: patch_func(
                patch_str=patch_str,
                original_str=original_str,
                patch_dir=patch_dir,
                language=language,
                verbose=verbose,
            ),
        )

        return result

    except Exception as e:
        logger.error(f"Error in async patch application: {e}")
        return None, 0, None, str(e), None, None


async def validate_code_async(
    code_path: str, language: str = "python", timeout: int = 30
) -> Tuple[bool, Optional[str]]:
    """Async code validation using subprocess.

    Args:
        code_path: Path to code file to validate
        language: Programming language
        timeout: Timeout for validation in seconds

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        if language == "python":
            # Use python -m py_compile for syntax checking
            proc = await asyncio.create_subprocess_exec(
                "python",
                "-m",
                "py_compile",
                code_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return False, f"Validation timeout after {timeout}s"

            if proc.returncode == 0:
                return True, None
            else:
                error_msg = stderr.decode() if stderr else "Unknown compilation error"
                return False, error_msg

        elif language == "rust":
            # Use rustc for Rust syntax checking
            proc = await asyncio.create_subprocess_exec(
                "rustc",
                "--crate-type=lib",
                "-Zparse-only",
                code_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return False, f"Validation timeout after {timeout}s"

            if proc.returncode == 0:
                return True, None
            else:
                error_msg = stderr.decode() if stderr else "Unknown compilation error"
                return False, error_msg
        elif language == "cpp":
            # Use g++ for C++ compilation check
            proc = await asyncio.create_subprocess_exec(
                "g++",
                "-fsyntax-only",
                code_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return False, f"Validation timeout after {timeout}s"

            if proc.returncode == 0:
                return True, None
            else:
                error_msg = stderr.decode() if stderr else "Unknown compilation error"
                return False, error_msg
        elif language == "swift":
            # Use swiftc for Swift syntax checking
            proc = await asyncio.create_subprocess_exec(
                "swiftc",
                "-typecheck",
                "-parse-as-library",
                code_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return False, f"Validation timeout after {timeout}s"

            if proc.returncode == 0:
                return True, None
            else:
                error_msg = stderr.decode() if stderr else "Unknown compilation error"
                return False, error_msg
        else:
            # For other languages, just check if file exists and is readable
            try:
                async with aiofiles.open(code_path, "r") as f:
                    content = await f.read()
                    if len(content.strip()) > 0:
                        return True, None
                    else:
                        return False, "Empty code file"
            except Exception as e:
                return False, f"File read error: {str(e)}"

    except Exception as e:
        logger.error(f"Error in async code validation: {e}")
        return False, f"Validation error: {str(e)}"


async def write_file_async(file_path: str, content: str) -> bool:
    """Async file writing.

    Args:
        file_path: Path to write file
        content: Content to write

    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure parent directory exists
        parent_dir = Path(file_path).parent
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: parent_dir.mkdir(parents=True, exist_ok=True)
        )

        if aiofiles:
            # Use aiofiles if available
            async with aiofiles.open(file_path, "w") as f:
                await f.write(content)
        else:
            # Fall back to sync I/O in thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, lambda: Path(file_path).write_text(content)
            )

        return True

    except Exception as e:
        logger.error(f"Error writing file {file_path}: {e}")
        return False


async def read_file_async(file_path: str) -> Optional[str]:
    """Async file reading.

    Args:
        file_path: Path to read file

    Returns:
        File content or None if error
    """
    try:
        if aiofiles:
            # Use aiofiles if available
            async with aiofiles.open(file_path, "r") as f:
                content = await f.read()
        else:
            # Fall back to sync I/O in thread pool
            loop = asyncio.get_event_loop()
            content = await loop.run_in_executor(
                None, lambda: Path(file_path).read_text()
            )
        return content

    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return None


async def copy_file_async(src_path: str, dst_path: str) -> bool:
    """Async file copying.

    Args:
        src_path: Source file path
        dst_path: Destination file path

    Returns:
        True if successful, False otherwise
    """
    try:
        # Read source file
        content = await read_file_async(src_path)
        if content is None:
            return False

        # Write to destination
        return await write_file_async(dst_path, content)

    except Exception as e:
        logger.error(f"Error copying file {src_path} to {dst_path}: {e}")
        return False


async def get_code_embedding_async(
    exec_fname: str, embedding_client, max_chars: int = 10000
) -> Tuple[Optional[list], float]:
    """Async code embedding generation.

    Args:
        exec_fname: Path to code file
        embedding_client: Embedding client instance
        max_chars: Maximum characters to embed

    Returns:
        Tuple of (embedding_vector, cost)
    """
    try:
        # Read code file asynchronously
        code_content = await read_file_async(exec_fname)
        if not code_content:
            return None, 0.0

        # Truncate if too long
        if len(code_content) > max_chars:
            code_content = code_content[:max_chars]

        # Generate embedding in thread pool
        loop = asyncio.get_event_loop()

        if hasattr(embedding_client, "embed_async"):
            # Use async embedding if available
            embedding, cost = await embedding_client.embed_async(code_content)
        else:
            # Fall back to sync embedding in thread pool
            embedding, cost = await loop.run_in_executor(
                None, embedding_client.get_embedding, code_content
            )

        return embedding, cost

    except Exception as e:
        logger.error(f"Error generating code embedding for {exec_fname}: {e}")
        return None, 0.0
