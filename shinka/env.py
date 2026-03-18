from typing import Optional
from pathlib import Path

from dotenv import load_dotenv


def load_shinka_dotenv(
    package_root: Optional[Path] = None, cwd: Optional[Path] = None
) -> tuple[Path, ...]:
    """Load package and launch-directory dotenv files, with launch-dir precedence."""
    resolved_package_root = (
        package_root if package_root is not None else Path(__file__).resolve().parents[1]
    )
    resolved_cwd = cwd if cwd is not None else Path.cwd()

    env_paths: list[Path] = []
    package_env = resolved_package_root / ".env"
    launch_env = resolved_cwd / ".env"

    if package_env.exists():
        env_paths.append(package_env)
    if launch_env.exists() and launch_env not in env_paths:
        env_paths.append(launch_env)

    for env_path in env_paths:
        load_dotenv(dotenv_path=env_path, override=True)

    return tuple(env_paths)
