"""Security utilities for safe file operations and command execution."""

import os
import re
import shlex
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Raised when a security violation is detected."""
    pass


def validate_safe_path(base_dir: str, user_path: str) -> Path:
    """
    Ensure user_path is within base_dir to prevent path traversal attacks.

    Args:
        base_dir: The base directory that should contain the user path
        user_path: The user-provided path to validate

    Returns:
        Path: The resolved, validated path

    Raises:
        SecurityError: If the path escapes the base directory
    """
    base = Path(base_dir).resolve()

    # Handle absolute user paths by making them relative to base
    user_path_obj = Path(user_path)
    if user_path_obj.is_absolute():
        # For absolute paths, check if they're already within base
        target = user_path_obj.resolve()
    else:
        target = (base / user_path).resolve()

    # Check if target is within base directory
    try:
        target.relative_to(base)
    except ValueError:
        raise SecurityError(
            f"Path traversal attempt detected: '{user_path}' resolves outside '{base_dir}'"
        )

    return target


def validate_docker_image_name(image: str) -> str:
    """
    Validate Docker image name against official Docker naming conventions.

    Args:
        image: The Docker image name to validate

    Returns:
        str: The validated image name

    Raises:
        SecurityError: If the image name is invalid or potentially malicious
    """
    # Docker naming convention: [registry/][namespace/]repository[:tag][@digest]
    # Allow alphanumeric, dots, underscores, hyphens, colons, slashes, and @
    pattern = r'^[a-z0-9._/-]+(:[\w.-]+)?(@sha256:[a-f0-9]{64})?$'

    if not re.match(pattern, image, re.IGNORECASE):
        raise SecurityError(
            f"Invalid Docker image name: '{image}'. "
            "Image names must follow Docker naming conventions."
        )

    # Additional check for shell metacharacters
    dangerous_chars = [';', '|', '&', '$', '`', '\\', '\n', '\r', '>', '<']
    for char in dangerous_chars:
        if char in image:
            raise SecurityError(
                f"Docker image name contains dangerous character '{char}': {image}"
            )

    return image


def sanitize_command_args(cmd: List[str]) -> str:
    """
    Safely join command arguments using shlex.join() to prevent command injection.

    Args:
        cmd: List of command arguments

    Returns:
        str: Safely joined command string
    """
    if not cmd:
        return ""

    # Use shlex.join for Python 3.8+, which properly quotes shell arguments
    return shlex.join(cmd)


def get_api_key(key_name: str, required: bool = True) -> Optional[str]:
    """
    Safely retrieve API key from environment variables.

    Args:
        key_name: Name of the environment variable
        required: Whether the key is required (raises error if not found)

    Returns:
        Optional[str]: The API key value, or None if not required and not found

    Raises:
        ValueError: If the key is required but not found
    """
    api_key = os.getenv(key_name)

    if not api_key and required:
        raise ValueError(
            f"{key_name} environment variable not set. "
            f"Please configure your API credentials before proceeding."
        )

    if api_key:
        logger.debug(f"Successfully loaded API key: {key_name}")

    return api_key


def validate_file_path(file_path: str, must_exist: bool = False) -> Path:
    """
    Validate that a file path is safe and optionally exists.

    Args:
        file_path: The file path to validate
        must_exist: Whether the file must exist

    Returns:
        Path: The validated path

    Raises:
        SecurityError: If the path is invalid
        FileNotFoundError: If must_exist is True and file doesn't exist
    """
    path = Path(file_path).resolve()

    # Check for null bytes (common in path traversal attacks)
    if '\x00' in str(file_path):
        raise SecurityError(f"Null byte detected in file path: {file_path}")

    if must_exist and not path.exists():
        raise FileNotFoundError(f"Required file not found: {file_path}")

    return path
