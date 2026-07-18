FROM node:22-bookworm-slim@sha256:d415caac2f1f77b98caaf9415c5f807e14bc8d7bdea62561ea2fef4fbd08a73c

ARG TARGETARCH
ARG HEADLESS_VERSION=0.4.0
ARG CLAUDE_VERSION=2.1.119
ARG CODEX_VERSION=0.125.0
ARG GEMINI_VERSION=0.39.1
ARG OPENCODE_VERSION=1.14.25
ARG PI_VERSION=0.70.2
ARG CURSOR_AGENT_VERSION=2026.04.17-787b533
ARG CURSOR_AGENT_SHA256_AMD64=942b2b583239497715f2390336eb8e2df3673703bd167bd59f7be33a27e15c5a
ARG CURSOR_AGENT_SHA256_ARM64=985191ffc606da1317e1399f99306481f58643f345ed2252033b011504c780ce
ARG AGY_RELEASE=1.1.4-6277569641840640
ARG AGY_SHA512_AMD64=a088a1f231d8565b6673cecd8656fc3504e49c89e9c6b8c4116937b5fe7069c8dcfba78bbb2bc5c0ff8e87ba64fe21b63db7001e3a5794504927dad9e89da973
ARG AGY_SHA512_ARM64=8d3c464303b235b6f2c2d441eca07b0c1cc35efa68f7ae16b167a5a2d49373903efdf686b3e41063424f0cf0c5b5d5eb056f7944dade7abf1b8eb225cb8c438c

LABEL io.shinka.headless.version="${HEADLESS_VERSION}" \
      io.shinka.headless.agents="antigravity,claude,codex,cursor,gemini,opencode,pi" \
      io.shinka.antigravity.release="${AGY_RELEASE}" \
      io.shinka.claude.version="${CLAUDE_VERSION}" \
      io.shinka.codex.version="${CODEX_VERSION}" \
      io.shinka.cursor.version="${CURSOR_AGENT_VERSION}" \
      io.shinka.gemini.version="${GEMINI_VERSION}" \
      io.shinka.opencode.version="${OPENCODE_VERSION}" \
      io.shinka.pi.version="${PI_VERSION}" \
      io.shinka.sandbox-user="65532:65532"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        git \
        ripgrep \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g \
        "@roberttlange/headless@${HEADLESS_VERSION}" \
        "@anthropic-ai/claude-code@${CLAUDE_VERSION}" \
        "@google/gemini-cli@${GEMINI_VERSION}" \
        "@mariozechner/pi-coding-agent@${PI_VERSION}" \
        "@openai/codex@${CODEX_VERSION}" \
        "opencode-ai@${OPENCODE_VERSION}"

RUN set -eu; \
    case "${TARGETARCH:-$(dpkg --print-architecture)}" in \
        amd64) cursor_arch="x64"; cursor_sha256="${CURSOR_AGENT_SHA256_AMD64}" ;; \
        arm64) cursor_arch="arm64"; cursor_sha256="${CURSOR_AGENT_SHA256_ARM64}" ;; \
        *) echo "unsupported Cursor architecture: ${TARGETARCH:-unknown}" >&2; exit 1 ;; \
    esac; \
    cursor_tarball="/tmp/cursor-agent.tar.gz"; \
    curl "https://downloads.cursor.com/lab/${CURSOR_AGENT_VERSION}/linux/${cursor_arch}/agent-cli-package.tar.gz" -fsSL -o "${cursor_tarball}"; \
    echo "${cursor_sha256}  ${cursor_tarball}" | sha256sum -c -; \
    mkdir -p /opt/cursor-agent; \
    tar --strip-components=1 -xzf "${cursor_tarball}" -C /opt/cursor-agent; \
    ln -s /opt/cursor-agent/cursor-agent /usr/local/bin/cursor-agent; \
    ln -s /usr/local/bin/cursor-agent /usr/local/bin/agent; \
    rm -f "${cursor_tarball}"

RUN set -eu; \
    case "${TARGETARCH:-$(dpkg --print-architecture)}" in \
        amd64) agy_arch="linux-x64"; agy_file="cli_linux_x64.tar.gz"; agy_sha512="${AGY_SHA512_AMD64}" ;; \
        arm64) agy_arch="linux-arm"; agy_file="cli_linux_arm64.tar.gz"; agy_sha512="${AGY_SHA512_ARM64}" ;; \
        *) echo "unsupported Antigravity architecture: ${TARGETARCH:-unknown}" >&2; exit 1 ;; \
    esac; \
    agy_tarball="/tmp/agy.tar.gz"; \
    curl "https://storage.googleapis.com/antigravity-public/antigravity-cli/${AGY_RELEASE}/${agy_arch}/${agy_file}" -fsSL -o "${agy_tarball}"; \
    echo "${agy_sha512}  ${agy_tarball}" | sha512sum -c -; \
    tar -xzf "${agy_tarball}" -C /tmp antigravity; \
    install -m 0755 /tmp/antigravity /usr/local/bin/agy; \
    rm -f "${agy_tarball}" /tmp/antigravity

RUN groupadd --gid 65532 shinka \
    && useradd --uid 65532 --gid 65532 --home-dir /headless-home \
        --shell /bin/sh shinka \
    && command -v headless \
    && command -v agy \
    && command -v claude \
    && command -v codex \
    && command -v agent \
    && command -v gemini \
    && command -v opencode \
    && command -v pi

ENV AGY_CLI_DISABLE_AUTO_UPDATE=true \
    DISABLE_AUTOUPDATER=1 \
    GEMINI_CLI_NO_UPDATE_NOTIFIER=true \
    NODE_USE_ENV_PROXY=1 \
    NO_UPDATE_NOTIFIER=1 \
    OPENCODE_DISABLE_AUTOUPDATE=true \
    PATH="/headless-home/.local/bin:/headless-home/.cursor/bin:/headless-home/.cursor/cli/bin:${PATH}"

WORKDIR /workspace
USER 65532:65532
