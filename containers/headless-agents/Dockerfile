FROM node:22-bookworm-slim@sha256:f32b81066cde10a75dbac96646099533316d94bac4150c55da1636e1f0ffdc46

ARG TARGETARCH
ARG HEADLESS_VERSION=0.5.0
ARG CLAUDE_VERSION=2.1.221
ARG CODEX_VERSION=0.146.0
ARG GEMINI_VERSION=0.53.1
ARG OPENCODE_VERSION=1.18.12
ARG PI_PACKAGE=@earendil-works/pi-coding-agent
ARG PI_VERSION=0.83.0
ARG CURSOR_AGENT_VERSION=2026.07.23-e383d2b
ARG CURSOR_AGENT_SHA256_AMD64=702ad595213bee5df0268be9f80a19f29fcceaa2a42fc55e39f2b5199051f0c4
ARG CURSOR_AGENT_SHA256_ARM64=f40b99647cb24e0da885e97620a2048034f1fe8961910d573d827d77c4d26dcb
ARG AGY_RELEASE=1.1.10
ARG AGY_SHA512_AMD64=e64d4e58ede0f8440f2b3dc021f9d6d36b05f5c2f74d5a9215c1f11b20d536c8c2e020f4ce5257aa67e940e94c94d5a16d3aa6461cda18ee7f3e74d3a20ca1ac
ARG AGY_SHA512_ARM64=2d64c4e09eb22c824bc298ca61e48cb0e62688353db4122996b2e4bba8c9a1570d2cbf6e6452ed7ef30df0940bf8f058789e466aed5264c1ae2e4613eca5b573

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
        "${PI_PACKAGE}@${PI_VERSION}" \
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
        amd64) agy_file="agy_cli_linux_x64.tar.gz"; agy_sha512="${AGY_SHA512_AMD64}" ;; \
        arm64) agy_file="agy_cli_linux_arm64.tar.gz"; agy_sha512="${AGY_SHA512_ARM64}" ;; \
        *) echo "unsupported Antigravity architecture: ${TARGETARCH:-unknown}" >&2; exit 1 ;; \
    esac; \
    agy_tarball="/tmp/agy.tar.gz"; \
    curl "https://github.com/google-antigravity/antigravity-cli/releases/download/${AGY_RELEASE}/${agy_file}" -fsSL -o "${agy_tarball}"; \
    echo "${agy_sha512}  ${agy_tarball}" | sha512sum -c -; \
    tar -xzf "${agy_tarball}" -C /tmp antigravity; \
    install -m 0755 /tmp/antigravity /usr/local/bin/agy; \
    rm -f "${agy_tarball}" /tmp/antigravity

RUN groupadd --gid 65532 shinka \
    && useradd --uid 65532 --gid 65532 --home-dir /headless-home \
        --shell /bin/sh shinka \
    && mkdir -p /headless-home \
    && chown 65532:65532 /headless-home \
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
    HOME=/headless-home \
    NODE_USE_ENV_PROXY=1 \
    NO_UPDATE_NOTIFIER=1 \
    OPENCODE_DISABLE_AUTOUPDATE=true \
    PATH="/headless-home/.local/bin:/headless-home/.cursor/bin:/headless-home/.cursor/cli/bin:${PATH}"

WORKDIR /workspace
USER 65532:65532
