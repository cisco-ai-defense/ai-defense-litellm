# cisco-ai-defense-litellm

**Cisco AI Defense integration for LiteLLM**
Inspect, block, or mutate LLM and Model Context Protocol (MCP) traffic flowing through [LiteLLM](https://github.com/BerriAI/litellm) with Cisco AI Defense, in both proxy and embedded-SDK deployments.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quickstart](#quickstart)
  - [LiteLLM Proxy mode](#litellm-proxy-mode)
  - [LiteLLM Python SDK mode](#litellm-python-sdk-mode)
- [Configuration](#configuration)
  - [Environment variables](#environment-variables)
  - [Block modes](#block-modes)
  - [Targeting specific models or MCP servers](#targeting-specific-models-or-mcp-servers)
- [Usage Examples](#usage-examples)
  - [Programmatic install in a Python application](#programmatic-install-in-a-python-application)
  - [Uninstalling the hook](#uninstalling-the-hook)
- [What gets inspected](#what-gets-inspected)
- [Compatibility](#compatibility)
- [Error Handling](#error-handling)
- [License](#license)
- [Support](#support)

---

## Overview

The `cisco-ai-defense-litellm` package provides a single LiteLLM hook that wires every model and MCP call routed through LiteLLM into Cisco AI Defense's `/inspect/chat` and `/inspect/mcp` endpoints. The same hook implementation runs in two deployment shapes:

| Deployment | How it loads | Where it runs |
| --- | --- | --- |
| LiteLLM Proxy / Gateway | Registered as a `callbacks` entry in `litellm.yaml` | Inside the LiteLLM proxy container |
| LiteLLM Python SDK | One-line `install()` from your application code | In-process with your application |

In both modes the hook validates the request before the upstream model call, validates the response before it returns, validates MCP `tools/call` requests and responses, and can mutate either payload when AI Defense returns a sanitized version.

---

## Features

- **Drop-in LiteLLM hook**: Single callback covers chat, completion, embedding, and MCP traffic; no per-route wiring.
- **Proxy and SDK parity**: The same `AIDefenseLiteLLMHook` runs as a LiteLLM proxy callback or as an in-process SDK hook.
- **Request and response inspection**: Sends chat payloads to `/inspect/chat` and MCP JSON-RPC payloads to `/inspect/mcp`.
- **Configurable enforcement**: Block, sanitize, or pass-through per AI Defense verdict; choose between `message`-style blocks and exception-raising blocks.
- **Selective targeting**: Restrict enforcement to specific model aliases or MCP server aliases via env var or constructor argument.
- **Fail-open / fail-closed**: Configure how the hook should behave when AI Defense is unreachable or times out.
- **Structured logging and metadata**: Records every decision under `metadata["ai_defense"]` on the LiteLLM request so it surfaces in LiteLLM logs and downstream observability.
- **Async-native**: Built on LiteLLM's async hook surface; no extra thread pools.

---

## Installation

Pick the extra that matches how you run LiteLLM.

```bash
# Embedding the LiteLLM Python SDK in your own app:
pip install "cisco-ai-defense-litellm[sdk]"

# Running the LiteLLM proxy:
pip install "cisco-ai-defense-litellm[proxy]"
```

> The base `pip install cisco-ai-defense-litellm` installs the hook without pulling in LiteLLM. Use it when LiteLLM is already managed by your environment (for example, the official `litellm` container image).

**Requirements:** Python 3.10 – 3.13.

---

## Quickstart

### LiteLLM Proxy mode

Add a single `callbacks` entry to your `litellm.yaml`:

```yaml
litellm_settings:
  callbacks: cisco_ai_defense_litellm.proxy_handler_instance
  set_verbose: true

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
```

Then export the AI Defense endpoint, API key, and the comma-separated list of LiteLLM aliases you want inspected before starting the proxy:

```bash
export AI_DEFENSE_CHAT_INSPECT_URL="https://<tenant>.ai-defense.cisco.com/inspect/chat"
export AI_DEFENSE_CHAT_API_KEY="<your-api-key>"

# Allow-list of model aliases (from `model_list`) and MCP server aliases
# (from `mcp_servers`) that should be inspected. Anything not listed is
# passed through unchanged. See "Targeting specific models or MCP servers".
export AI_DEFENSE_API_MODE="gpt-4o-mini,my_mcp_server"

litellm --config litellm.yaml --port 4000
```

Every request whose `model` (or MCP server alias) appears in `AI_DEFENSE_API_MODE` is now inspected before reaching the upstream provider.

### LiteLLM Python SDK mode

If your application uses LiteLLM as a library (`import litellm; litellm.completion(...)`), install the hook once at startup:

```python
import litellm
from cisco_ai_defense_litellm import install

install()  # registers the AI Defense hook on the litellm module

response = litellm.completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello, world"}],
)
```

If AI Defense blocks the request, `litellm.completion` raises a `BadRequestError` (or `RuntimeError` when the LiteLLM exception type isn't available) whose message is prefixed with `Blocked by AI Defense:`.

---

## Configuration

All configuration is read from environment variables, so the same package works identically in containers, serverless, and local development. Constructor arguments on `AIDefenseLiteLLMHook` override the env defaults when you need finer control.

### Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `AI_DEFENSE_CHAT_INSPECT_URL` | _unset_ | AI Defense `/inspect/chat` endpoint URL. Required to inspect chat/completion calls. |
| `AI_DEFENSE_MCP_INSPECT_URL` | _unset_ | AI Defense `/inspect/mcp` endpoint URL. Required to inspect MCP `tools/call` traffic. |
| `AI_DEFENSE_CHAT_API_KEY` | _unset_ | API key sent to the chat inspect endpoint. |
| `AI_DEFENSE_MCP_API_KEY` | _unset_ | API key sent to the MCP inspect endpoint. |
| `AI_DEFENSE_API_KEY_HEADER` | `X-Cisco-AI-Defense-API-Key` | Header name used to pass the API key. |
| `AI_DEFENSE_TIMEOUT_SECONDS` | `2.0` | Per-request HTTP timeout when calling AI Defense. |
| `AI_DEFENSE_FAIL_OPEN` | `false` | If `true`, allow the upstream call when AI Defense is unreachable or times out. |
| `AI_DEFENSE_BLOCK_MODE` | `message` | `message` returns a structured refusal response; `exception` raises so callers fail fast. |
| `AI_DEFENSE_SCAN_OUTPUTS` | `true` | When `false`, only requests are inspected; responses pass through. |
| `AI_DEFENSE_API_MODE` | _unset_ | Comma-separated allow-list of model aliases (from `model_list`) and MCP server aliases (from `mcp_servers`) that should be inspected. **When unset or empty, no inspection runs** — every named target is skipped with `target_not_allowed=<name>`. Aliases that look like an AI Defense Gateway target (containing `ai_defense_gateway` or starting with `gateway_`) are auto-skipped regardless of this setting, since the gateway itself performs inspection. |
| `AI_DEFENSE_LOG_LEVEL` | `INFO` | Log level for the `cisco_ai_defense_litellm` logger. |
| `AI_DEFENSE_STDOUT_LOGS` | `false` | When `true`, also emit hook logs to stdout with an `AI_DEFENSE_HOOK` prefix and disable propagation, so verdicts are reliably visible in container logs (e.g. `docker compose logs … \| grep AI_DEFENSE_HOOK`). |
| `AI_DEFENSE_LOG_TEXT` | `false` | When `true`, includes the inspected text content (prompt/response) in log lines. **Keep `false` in production** — enabling it can leak prompt and response data into stdout. |

### Block modes

- `AI_DEFENSE_BLOCK_MODE=message` (default): the hook returns a sanitized refusal message via LiteLLM's standard guardrail surface. The caller sees an HTTP 200 with a "blocked" response — best for end-user chat UIs.
- `AI_DEFENSE_BLOCK_MODE=exception`: the hook raises, so LiteLLM surfaces an error to the caller. Best for service-to-service traffic where you want callers to retry or fail explicitly.

### Targeting specific models or MCP servers

`AI_DEFENSE_API_MODE` is an explicit allow-list. **Only requests whose `model` alias or MCP server alias is in the list are sent to AI Defense for inspection.** Anything not listed — including the case where `AI_DEFENSE_API_MODE` is unset or empty — is passed through unchanged.

```bash
export AI_DEFENSE_API_MODE="gpt-4o-mini,bedrock-claude-sonnet-4-profile,my_mcp_server"
```

This is the opposite of an opt-out / deny-list: leaving the variable unset disables LiteLLM-side inspection entirely.

**Auto-skipped aliases.** Independently of `AI_DEFENSE_API_MODE`, the hook auto-skips any alias that looks like an AI Defense Gateway target — specifically aliases containing `ai_defense_gateway` or starting with `gateway_` (case-insensitive, hyphens normalized to underscores). These deployments use AI Defense Gateway as the upstream and inspection happens inside the gateway, so running it again in the LiteLLM hook would double-inspect.

---

## Usage Examples

### Programmatic install in a Python application

```python
from cisco_ai_defense_litellm import install, AIDefenseLiteLLMHook

hook: AIDefenseLiteLLMHook = install(
    block_mode="exception",
    scan_outputs=True,
    api_modes=("gpt-4o-mini",),
)

print("AI Defense hook installed:", type(hook).__name__)
```

The returned `AIDefenseLiteLLMHook` is the same instance LiteLLM will invoke, so it's safe to keep a reference for inspection or to call `uninstall()` later.

### Uninstalling the hook

```python
from cisco_ai_defense_litellm import is_installed, uninstall

if is_installed():
    uninstall()
```

`install()` is idempotent — calling it more than once returns the existing hook without registering duplicate callbacks.

---

## What gets inspected

| LiteLLM surface | AI Defense endpoint | Direction |
| --- | --- | --- |
| `chat.completions`, `completions`, `embeddings`, `messages` | `/inspect/chat` | request and response |
| MCP `tools/call` requests | `/inspect/mcp` | request and response |
| Streaming responses | `/inspect/chat` | response chunks aggregated before inspection |

Every decision is recorded on the LiteLLM request dict under:

```python
data["metadata"]["ai_defense"] = {
    "input":  {"action": "allow" | "block" | "sanitize", ...},
    "output": {"action": "allow" | "block" | "sanitize", ...},
}
```

This makes verdicts visible to LiteLLM logging, callbacks, and any downstream observability you have wired into LiteLLM.

---

## Compatibility

- **LiteLLM**: `>=1.83, <2` (the `[sdk]` and `[proxy]` extras pin this).
- **Python**: `3.10`, `3.11`, `3.12`, `3.13`.
- **AI Defense API**: Tenant-issued `/inspect/chat` and `/inspect/mcp` endpoints.

---

## Error Handling

- **AI Defense returns block**: hook either returns a refusal message or raises, per `AI_DEFENSE_BLOCK_MODE`.
- **AI Defense returns sanitize**: hook mutates the request or response payload in place and lets the call continue.
- **AI Defense unreachable or timeout**:
  - `AI_DEFENSE_FAIL_OPEN=false` (default): the hook raises, blocking the call.
  - `AI_DEFENSE_FAIL_OPEN=true`: the hook logs a warning and lets the call through.
- **Misconfiguration** (missing endpoint URL or API key): the hook logs at startup and behaves as if the corresponding inspection surface is disabled — it never crashes LiteLLM.

---

## License

Apache-2.0. See [LICENSE](https://github.com/cisco-ai-defense/ai-defense-litellm/blob/main/LICENSE).

---

## Support

- **Issues and feature requests**: [GitHub Issues](https://github.com/cisco-ai-defense/ai-defense-litellm/issues)
- **Source code**: [cisco-ai-defense/ai-defense-litellm](https://github.com/cisco-ai-defense/ai-defense-litellm)
- **Cisco AI Defense product docs**: [https://www.cisco.com/site/us/en/products/security/ai-defense/index.html](https://www.cisco.com/site/us/en/products/security/ai-defense/index.html)
