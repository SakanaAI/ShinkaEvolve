#!/usr/bin/env python3
"""
Pre-commit hook to ensure all dependencies in pyproject.toml are pinned.

This script checks that:
- All dependencies use == for version pinning (not >=, ~=, or no version)
- Build system requirements are pinned
- Dev dependencies are pinned
"""

import re
import sys
from pathlib import Path


def check_pinned_dependencies(pyproject_path: Path) -> list[str]:
    """
    Check that all dependencies in pyproject.toml are pinned with ==.

    Returns:
        List of error messages for unpinned dependencies
    """
    errors = []
    content = pyproject_path.read_text()

    # Pattern to match dependency lines more precisely
    # Matches lines that start with optional whitespace and a quote, then package name
    dep_pattern = re.compile(r'^\s*"([a-zA-Z0-9_-]+)(==|>=|<=|~=|!=|>|<)?([^"]*)"')

    in_dependencies = False
    in_dev_dependencies = False
    in_build_system = False
    line_num = 0

    for line in content.split('\n'):
        line_num += 1
        stripped = line.strip()

        # Track which section we're in
        if 'dependencies = [' in stripped:
            in_dependencies = True
            in_dev_dependencies = False
            in_build_system = False
            continue
        elif 'dev-dependencies = [' in stripped:
            in_dependencies = False
            in_dev_dependencies = True
            in_build_system = False
            continue
        elif 'requires = [' in stripped:
            # Check if we're in build-system section
            lines_before = content[:content.find(line)]
            if '[build-system]' in lines_before and lines_before.rindex('[build-system]') > lines_before.rfind(']') if ']' in lines_before else True:
                in_dependencies = False
                in_dev_dependencies = False
                in_build_system = True
            continue
        elif stripped == ']' and (in_dependencies or in_dev_dependencies or in_build_system):
            in_dependencies = False
            in_dev_dependencies = False
            in_build_system = False
            continue

        # Check dependencies in relevant sections
        if in_dependencies or in_dev_dependencies or in_build_system:
            # Skip if it's a comment or empty
            if not stripped or stripped.startswith('#'):
                continue

            match = dep_pattern.match(stripped)
            if match:
                package = match.group(1)
                operator = match.group(2) if match.group(2) else None
                version = match.group(3)

                # Check if pinned with ==
                if operator != '==':
                    section = 'dependencies' if in_dependencies else \
                             'dev-dependencies' if in_dev_dependencies else \
                             'build-system requires'
                    if operator:
                        errors.append(
                            f"Line {line_num}: Package '{package}' uses '{operator}' "
                            f"instead of '==' in {section}"
                        )
                    else:
                        errors.append(
                            f"Line {line_num}: Package '{package}' has no version pin "
                            f"in {section}"
                        )

    # Special check for requires-python
    python_req_pattern = re.compile(r'requires-python\s*=\s*"([^"]+)"')
    python_match = python_req_pattern.search(content)
    if python_match:
        python_spec = python_match.group(1)
        # Allow ==3.12.* pattern for Python version
        if not python_spec.startswith('=='):
            errors.append(
                f"requires-python uses '{python_spec}' instead of '==' "
                f"(e.g., '==3.12.*')"
            )

    return errors


def main():
    """Main entry point for the pre-commit hook."""
    pyproject_path = Path('pyproject.toml')

    if not pyproject_path.exists():
        print("ERROR: pyproject.toml not found")
        return 1

    errors = check_pinned_dependencies(pyproject_path)

    if errors:
        print("❌ Dependency pinning check FAILED\n")
        print("All dependencies must be pinned with '==' (not '>=', '~=', or unpinned)\n")
        for error in errors:
            print(f"  {error}")
        print("\nPlease pin all dependencies to specific versions.")
        return 1

    print("✓ All dependencies are properly pinned")
    return 0


if __name__ == '__main__':
    sys.exit(main())
