# Hardened Docker evaluation

`SecureDockerJobConfig` runs Shinka's existing single-file candidate and
`evaluate.py` interface in one hardened local Docker container. It is opt-in:
set `EvolutionConfig.job_type="secure_docker"` and supply an immutable image
digest.

This mode is useful when generated code is untrusted and the main concern is
protecting the host that runs Shinka. It supports every existing candidate
language because it does not interpret the candidate itself: the configured
image supplies the language runtime/compiler, while `evaluate.py` is still
invoked with its existing arguments. The candidate must remain one regular
file; it is never treated as a repository or directory tree.

## Use it

Build or obtain an evaluator image that includes:

- Python and `shinka-evolve`, plus the evaluator's Python dependencies;
- the toolchain required by the candidate task (for example Go, Julia,
  gfortran, Rust, C/C++, CUDA, Swift, Wolfram, or native Verilog tools); and
- no Docker `VOLUME` declarations (the launcher rejects them because they can
  bypass the read-only-root filesystem and consume host-backed storage); and
- no credentials or private data baked into the image.

Pull the exact image digest before starting a run. The secure launcher never
pulls an image implicitly.

```python
from shinka.core import EvolutionConfig, ShinkaEvolveRunner
from shinka.database import DatabaseConfig
from shinka.launch import SecureDockerJobConfig

runner = ShinkaEvolveRunner(
    evo_config=EvolutionConfig(
        job_type="secure_docker",
        language="go",
        init_program_path="examples/go_collatz_steps/initial.go",
        results_dir="results/go_secure",
    ),
    job_config=SecureDockerJobConfig(
        eval_program_path="examples/go_collatz_steps/evaluate.py",
        evaluator_root="examples/go_collatz_steps",
        image="registry.example/shinka/evaluator@sha256:<64-hex-digest>",
        time="00:05:00",
    ),
    db_config=DatabaseConfig(),
)
runner.run()
```

The same mode is available through `shinka_run`:

```bash
shinka_run \
  --task-dir examples/go_collatz_steps \
  --results_dir results/go_secure \
  --num_generations 20 \
  --set evo.job_type=secure_docker \
  --set job.image='registry.example/shinka/evaluator@sha256:<64-hex-digest>'
```

`evaluator_root` defaults to the parent directory of `evaluate.py`; set it when
the evaluator imports sibling modules or reads task assets from a broader task
directory. Keep that tree to ordinary task files: the launcher rejects nested
mounts, sockets, FIFOs, and device files before mounting it.

## Enforced policy

The launcher uses a shell-free `docker create` / inspect / `docker start`
sequence and fails before execution if the configured image is not present or
is not a digest reference. Every evaluation container is:

- networkless and unprivileged;
- non-root, with all Linux capabilities dropped and
  `no-new-privileges` enabled, with Docker's default seccomp profile required;
- read-only except for bounded tmpfs storage;
- bounded by explicit CPU, memory, PID, open-file, wall-time, and log-output
  limits (with swap disabled and Docker's persistent log driver disabled); and
- given only read-only mounts: the evaluator tree, a trusted runtime wrapper,
  and the one candidate file; it is automatically removed when it exits.

`/workspace/results` is a separate, size-limited container tmpfs
(`result_tmpfs_bytes`, 64 MiB by default), not a host bind mount. The trusted
wrapper invokes the evaluator with its ordinary arguments, captures bounded
stdout/stderr, and exports only regular `metrics.json` and `correct.json` files
to the host as bounded JSON objects. This prevents a candidate from filling the host disk or using a
result-path symlink to make the host read or write another file. Other
evaluator artifacts remain in the container intentionally.

The wrapper enforces the configured `time` limit inside the container as well
as the scheduler enforcing it on the host, so an evaluator cannot keep running
indefinitely if the host-side monitor disappears.

On Linux, the Docker engine must be rootless by default. Set
`allow_rootful_dedicated_vm=True` only when the daemon is inside a dedicated
container VM. Docker Desktop supplies a VM boundary on macOS and Windows; on
Windows, configure an explicit non-root numeric `sandbox_user` because a host
POSIX UID:GID is unavailable.

The `/tmp` tmpfs intentionally remains executable: several supported evaluator
patterns compile a single-file candidate into a temporary directory before
running it. The result tmpfs is `noexec`; it is for evaluator result data, not
compiled programs.

## Important boundary limitation

This mode preserves the old evaluator contract by putting `evaluate.py` and the
candidate in the **same** container. That is deliberately a host-containment
boundary, not a confidentiality or score-integrity boundary between the two.

For example, a Python candidate dynamically imported by `evaluate.py` runs in
the evaluator's interpreter and can access anything that interpreter can read.
Likewise, a candidate can forge ordinary score data written by a legacy
evaluator. The wrapper rejects symlink, non-regular, and oversized exports, but
it cannot make an arbitrary legacy evaluator trust malicious ordinary output.

Do not put secrets or undisclosed private test data in `evaluator_root` when
using this compatibility mode. Do not use it with evaluators that need a Docker
socket or unrestricted network access; install their tools directly in the
pinned image instead. A stronger evaluator/candidate boundary requires a
separate candidate-runner protocol and therefore a new evaluator API, which is
intentionally outside this compatibility feature.

For example, a Verilog evaluator can run `iverilog`, `vvp`, Yosys, and OpenSTA
directly from the pinned image. The current `examples/rtllm/evaluate.py` has a
fallback that launches nested Docker containers; that fallback is intentionally
incompatible with this mode because no Docker socket is exposed.

Docker is defense in depth, not a proof of complete safety. A daemon, kernel,
or runtime escape vulnerability can still break containment; rootless Docker on
Linux reduces the blast radius but does not remove it. Treat the evaluator image
and Docker daemon as trusted infrastructure, keep them patched, and use a
dedicated VM for higher-risk workloads.
