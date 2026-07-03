from __future__ import annotations

import asyncio
import os
import shlex
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class ValidationTier:
    name: str
    command: str 
    timeout: Optional[float] = None
    required: bool = True
    env: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ValidationTier":
        timeout = data.get("timeout")
        if isinstance(timeout, str):
            timeout = float(timeout)
        return cls(
            name=str(data.get("name") or data.get("command") or "validation"),
            command=str(data["command"]),
            timeout=float(timeout) if timeout is not None else None,
            required=bool(data.get("required", True)),
            env={str(k): str(v) for k, v in (data.get("env") or {}).items()},
        )


@dataclass
class ValidationTierResult:
    name: str
    command: str
    required: bool
    passed: bool
    returncode: Optional[int]
    duration_seconds: float
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "command": self.command,
            "required": self.required,
            "passed": self.passed,
            "returncode": self.returncode,
            "duration_seconds": self.duration_seconds,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "error": self.error,
        }


async def run_validation_tier(
    tier: ValidationTier,
    *,
    cwd: Path,
) -> ValidationTierResult:
    started = time.time()
    process = await asyncio.create_subprocess_shell(
        tier.command,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=None if not tier.env else {**os.environ, **tier.env},
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=tier.timeout,
        )
        returncode = process.returncode
        error = None
    except asyncio.TimeoutError:
        process.kill()
        stdout, stderr = await process.communicate()
        returncode = process.returncode
        error = f"validation tier timed out after {tier.timeout}s"

    duration = time.time() - started
    return ValidationTierResult(
        name=tier.name,
        command=tier.command,
        required=tier.required,
        passed=returncode == 0 and error is None,
        returncode=returncode,
        duration_seconds=duration,
        stdout=stdout.decode("utf-8", errors="replace"),
        stderr=stderr.decode("utf-8", errors="replace"),
        error=error,
    )


async def run_validation_tiers(
    tiers: list[ValidationTier],
    *,
    cwd: Path,
    stop_on_required_failure: bool = True,
) -> list[ValidationTierResult]:
    results: list[ValidationTierResult] = []
    for tier in tiers:
        result = await run_validation_tier(tier, cwd=cwd)
        results.append(result)
        if stop_on_required_failure and tier.required and not result.passed:
            break
    return results


def render_acceptance_criteria(tiers: list[ValidationTier]) -> str:
    if not tiers:
        return (
            "No explicit cheap validation tiers were configured. Run the cheapest "
            "static, import, compile, or smoke checks you can reasonably run, and "
            "record any gaps in the summary."
        )
    lines = ["Run these validation tiers before returning:"]
    for tier in tiers:
        required = "required" if tier.required else "optional"
        timeout = f", timeout={tier.timeout}s" if tier.timeout else ""
        lines.append(
            f"- {tier.name} ({required}{timeout}): `{shlex.quote(tier.command)}`"
        )
    return "\n".join(lines)
