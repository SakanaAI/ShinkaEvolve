#!/bin/bash
set -eu

if [ "$#" -lt 1 ]; then
    echo "usage: headless_docker.sh <agent> [headless arguments...]" >&2
    exit 2
fi

case "$1" in
    --check|--version|--help|--show-config)
        exec env \
            PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH" \
            npx -y @roberttlange/headless "$@"
        ;;
esac

agent="$1"
shift

# Headless mounts its requested work directory and provider auth seeds in Docker
# mode. The Antigravity adapter also needs its small config file at the seed
# mount path, so that one file is passed read-only below.
# Headless does not allow session resumption together with --docker. Each
# Shinka proposal has its own worktree, so dropping the generated session
# selector is safe and keeps the Docker wrapper compatible with the provider.
host_home="${HOME}"
stage_home="$(mktemp -d "${TMPDIR:-/tmp}/shinka-headless-home.XXXXXX")"
cleanup_stage() {
    rm -rf "$stage_home"
}
trap cleanup_stage EXIT

# The native Antigravity directory contains hundreds of megabytes of mutable
# conversation databases. Copy only the stable OAuth token into a temporary
# provider home so concurrent Docker proposals do not race while cloning it.
mkdir -p "$stage_home/.gemini/antigravity-cli"
if [ -f "$host_home/.gemini/antigravity-cli/antigravity-oauth-token" ]; then
    cp "$host_home/.gemini/antigravity-cli/antigravity-oauth-token" \
        "$stage_home/.gemini/antigravity-cli/antigravity-oauth-token"
fi
if [ -f "$host_home/.cursor/cli-config.json" ]; then
    mkdir -p "$stage_home/.cursor"
    cp "$host_home/.cursor/cli-config.json" "$stage_home/.cursor/cli-config.json"
fi
# Codex keeps its subscription/API authentication separately from the other
# providers. Copy only the small auth file, not the multi-gigabyte sessions and
# caches under ~/.codex. The host config may target a newer Codex schema.
mkdir -p "$stage_home/.codex"
if [ -f "$host_home/.codex/auth.json" ]; then
    cp "$host_home/.codex/auth.json" "$stage_home/.codex/auth.json"
fi
# Keep the reasoning and service tier explicit in Docker. Codex accepts fast
# and flex; flex avoids the fast service tier for smoke tests and experiments.
codex_service_tier="${SHINKA_HEADLESS_DOCKER_CODEX_SERVICE_TIER:-fast}"
case "$codex_service_tier" in
    fast|flex) ;;
    *)
        echo "SHINKA_HEADLESS_DOCKER_CODEX_SERVICE_TIER must be fast or flex" >&2
        exit 2
        ;;
esac
cat > "$stage_home/.codex/config.toml" <<EOF
service_tier = "$codex_service_tier"
model_reasoning_effort = "medium"
EOF
export HOME="$stage_home"

docker_image="${SHINKA_HEADLESS_DOCKER_IMAGE:-ghcr.io/roberttlange/headless:latest}"
docker_platform="${SHINKA_HEADLESS_DOCKER_PLATFORM:-}"
docker_config_mount="${host_home}/.gemini/config:/tmp/headless-host-home/.gemini/config:ro"
docker_args=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --session)
            shift
            if [ "$#" -gt 0 ]; then
                shift
            fi
            ;;
        --session=*)
            shift
            ;;
        *)
            docker_args+=("$1")
            shift
            ;;
    esac
done

headless_docker_args=(--docker-image "$docker_image")
if [ -n "$docker_platform" ]; then
    headless_docker_args+=(--docker-arg "--platform=$docker_platform")
fi
headless_docker_args+=(--docker-arg --volume --docker-arg "$docker_config_mount")

exec env \
    PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH" \
    npx -y @roberttlange/headless "$agent" --docker \
    "${headless_docker_args[@]}" \
    "${docker_args[@]}"
