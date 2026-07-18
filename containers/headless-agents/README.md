# Secure universal Headless agent image

This image provides Headless plus every native CLI supported by Shinka's secure
adapter: Antigravity, Claude Code, Codex, Cursor Agent, Gemini CLI, OpenCode, and
Pi. Package versions and the Node base are pinned; Cursor and Antigravity
downloads are architecture-specific and checksum-verified for AMD64/ARM64.
There is deliberately no separate Antigravity image: it has the same image,
runtime identity, and preflight contract as every other route.

Build in a trusted preparation step, push to the operator registry, and configure
the resulting immutable `repository@sha256:<manifest-digest>` reference. Runtime
uses the dedicated non-root `65532:65532` identity. Shinka invokes the selected
route as:

```text
headless <agent> [--model <model>] [--reasoning-effort <effort>] --allow yolo --json
```

The manual `Publish Headless agents image` GitHub workflow builds AMD64 and
ARM64, publishes `ghcr.io/<owner>/shinka-headless-agents`, and reports the
immutable manifest digest. A local equivalent is:

```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --tag registry.example/shinka/headless-agents:0.4.0 \
  --push \
  containers/headless-agents
```

Preflight checks `io.shinka.headless.agents` plus Headless/native-agent version
and sandbox-user OCI labels, and fails before a proposal if any configured route
or identity is absent. The image build itself verifies every native executable.

The image intentionally does not bundle credentials. Configure one dedicated
minimal profile per selected agent:

| Agent | Allowlisted profile paths |
| --- | --- |
| `antigravity` | `.gemini/antigravity-cli/antigravity-oauth-token` plus bounded Antigravity settings/state |
| `claude` | `.claude.json`, `.claude/settings.json`, `.claude/.credentials.json`, `.claude/auth.json` |
| `codex` | `.codex/auth.json`, `.codex/config.toml` |
| `cursor` | `.cursor/cli-config.json` |
| `gemini` | selected `.gemini` account/settings/state files |
| `opencode` | `.config/opencode` |
| `pi` | `.pi/agent/auth.json`, `.pi/agent/settings.json` |

API-key authentication is also supported only through each agent's explicit
`agent_credential_env_names` allowlist:

| Agent | Permitted credential environment names |
| --- | --- |
| `antigravity` | none; use its container OAuth file |
| `claude` | `ANTHROPIC_API_KEY`, `CLAUDE_CODE_OAUTH_TOKEN` |
| `codex` | `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `CODEX_API_KEY` |
| `cursor` | `CURSOR_API_KEY` |
| `gemini` | `GOOGLE_API_KEY`, `GEMINI_API_KEY` |
| `opencode` | selected OpenAI, Anthropic, OpenRouter, or Google/Gemini API key |
| `pi` | selected OpenAI, Anthropic, OpenRouter, Google/Gemini, or `PI_CODING_AGENT_*` credential/model variables |

An API-auth route still needs a dedicated profile directory, which may be empty.
Never use a normal home directory.

Provider egress is independent of the image. Configure the proxy with only the
DNS suffixes needed by the selected routes. If one run selects several agents,
the proxy allowlist is the union of those reviewed provider endpoints. Add
`models.dev` when Headless must convert non-native token counts to dollar cost.
The image enables Node's standard environment-proxy support so that lookup uses
Shinka's controlled proxy rather than requiring direct egress.

ACP registry/custom-command execution is not included: dynamically resolving an
arbitrary ACP implementation would violate the pinned-binary execution contract.
