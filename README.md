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
    - [Pattern 1 — Custom image with pip-installed plugin](#pattern-1--custom-image-with-pip-installed-plugin)
    - [Pattern 2 — Stock LiteLLM image with pip install on host + mount](#pattern-2--stock-litellm-image-with-pip-install-on-host--mount)
    - [Picking between Pattern 1 and Pattern 2](#picking-between-pattern-1-and-pattern-2)
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
  callbacks: cisco_ai_defense.proxy_handler_instance
  set_verbose: true

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
```

The LiteLLM proxy resolves `litellm_settings.callbacks` as a file path relative to the directory containing `litellm.yaml`, **not** as a Python package import (see `litellm/proxy/types_utils/utils.py:get_instance_fn` upstream). You therefore create a one-line shim file `cisco_ai_defense.py` next to your YAML — **this file is not in the wheel**, you create it yourself after `pip install`:

```python
# cisco_ai_defense.py — customer-created, full file contents
from cisco_ai_defense_litellm import proxy_handler_instance
```

This lets you keep `cisco-ai-defense-litellm` installed as a regular pip package while still satisfying the proxy's file-based callback loader. Both patterns below show how to materialize the shim — `RUN echo …` inline in a Dockerfile (Pattern 1), or `cat > … <<'EOF'` once on the host (Pattern 2).

#### Pattern 1 — Custom image with pip-installed plugin

Standard production path. Pip install runs once at image build; the resulting image is immutable and signable. Best for Kubernetes / ECS / Cloud Run / Helm.

```dockerfile
FROM docker.litellm.ai/berriai/litellm:main-latest

ARG CISCO_AI_DEFENSE_LITELLM_VERSION=0.2.0
RUN pip install --no-cache-dir \
        "cisco-ai-defense-litellm[proxy]==${CISCO_AI_DEFENSE_LITELLM_VERSION}"

COPY litellm.yaml /app/config/litellm.yaml

# Materialize the shim alongside litellm.yaml. Not shipped in the wheel
# above — it has to live as a separate file because the LiteLLM proxy
# resolves litellm_settings.callbacks as a file path, not via importlib.
RUN echo 'from cisco_ai_defense_litellm import proxy_handler_instance' \
        > /app/config/cisco_ai_defense.py

EXPOSE 4000
CMD ["litellm", "--config", "/app/config/litellm.yaml", "--host", "0.0.0.0", "--port", "4000"]
```

For Helm (`oci://ghcr.io/berriai/litellm-helm`), build this image and reference it from `image.repository` / `image.tag`. The chart does not run `pip install` at pod start.

#### Pattern 2 — Stock LiteLLM image with pip install on host + mount

For customers who already run the official `docker.litellm.ai/berriai/litellm:main-latest` image and don't want to (or can't) ship a derived image. The same `pip install` runs once on the host or in CI; the resulting package directory is mounted into the stock image at deploy time.

Run both commands once, in the directory that holds your `litellm.yaml`. The first vendors the wheel into `vendor/`; the second creates the shim file (the shim is **not** in the wheel — you create it yourself):

```bash
# 1. Vendor the wheel once on the host (or in CI):
mkdir -p vendor
pip install --no-deps --target=./vendor "cisco-ai-defense-litellm==0.2.0"

# 2. Create the shim next to your litellm.yaml:
cat > cisco_ai_defense.py <<'EOF'
from cisco_ai_defense_litellm import proxy_handler_instance
EOF
```

```yaml
# docker-compose.yml — no derived image required.
services:
  litellm:
    image: docker.litellm.ai/berriai/litellm:main-latest
    entrypoint: []
    command: ["litellm", "--config", "/app/config/litellm.yaml", "--host", "0.0.0.0", "--port", "4000"]
    environment:
      PYTHONPATH: /opt/ai-defense
      LITELLM_MASTER_KEY:           ${LITELLM_MASTER_KEY}
      AI_DEFENSE_CHAT_INSPECT_URL:  ${AI_DEFENSE_CHAT_INSPECT_URL}
      AI_DEFENSE_CHAT_API_KEY:      ${AI_DEFENSE_CHAT_API_KEY}
    volumes:
      - ./vendor:/opt/ai-defense:ro              # pip-installed plugin
      - ./litellm.yaml:/app/config/litellm.yaml:ro
      - ./cisco_ai_defense.py:/app/config/cisco_ai_defense.py:ro
    ports:
      - "4000:4000"
```

`PYTHONPATH=/opt/ai-defense` puts the mounted package on `sys.path`, so the shim's `from cisco_ai_defense_litellm import proxy_handler_instance` resolves. Same wheel as Pattern 1, just unpacked to a host volume instead of a Docker layer. No image rebuild required.

For Helm, the same pattern works by serving `vendor/cisco_ai_defense_litellm/` and `cisco_ai_defense.py` from `ConfigMap`s and adding them via `extraVolumes` / `extraVolumeMounts` (with `PYTHONPATH=/opt/ai-defense` in `env`).

#### Picking between Pattern 1 and Pattern 2

| Concern | Pattern 1 (custom image) | Pattern 2 (stock image + mount) |
|---|---|---|
| Image | Customer-built `FROM litellm + pip install` | Stock `docker.litellm.ai/berriai/litellm:main-latest`, unmodified |
| When pip runs | At image build | Once on host / CI |
| Network at pod start | None | None |
| Customer policy: "no custom images" | Not a fit | Fits |
| Best for | Standard production, Helm | Customers already running the official LiteLLM image |

**Then export the AI Defense endpoint and API key before starting the proxy:**

```bash
export AI_DEFENSE_CHAT_INSPECT_URL="https://<tenant>.ai-defense.cisco.com/inspect/chat"
export AI_DEFENSE_CHAT_API_KEY="<your-api-key>"

litellm --config litellm.yaml --port 4000
```

Once the plugin is loaded, every chat completion and MCP `tools/call` request that flows through this LiteLLM proxy is inspected by AI Defense — both the request before it leaves and the response before it returns. To exclude a route, do not configure that model/MCP server in this LiteLLM instance, or do not load the plugin in the LiteLLM stack that serves it.

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
| `AI_DEFENSE_LOG_LEVEL` | `INFO` | Log level for the `cisco_ai_defense_litellm` logger. |
| `AI_DEFENSE_STDOUT_LOGS` | `false` | When `true`, also emit hook logs to stdout with an `AI_DEFENSE_HOOK` prefix and disable propagation, so verdicts are reliably visible in container logs (e.g. `docker compose logs … \| grep AI_DEFENSE_HOOK`). |
| `AI_DEFENSE_LOG_TEXT` | `false` | When `true`, includes the inspected text content (prompt/response) in log lines. **Keep `false` in production** — enabling it can leak prompt and response data into stdout. |

### Block modes

- `AI_DEFENSE_BLOCK_MODE=message` (default): the hook returns a sanitized refusal message via LiteLLM's standard guardrail surface. The caller sees an HTTP 200 with a "blocked" response — best for end-user chat UIs.
- `AI_DEFENSE_BLOCK_MODE=exception`: the hook raises, so LiteLLM surfaces an error to the caller. Best for service-to-service traffic where you want callers to retry or fail explicitly.

### Inspection scope

There is no per-alias allow-list or deny-list. **If `cisco_ai_defense` is loaded as a LiteLLM callback (Proxy) or installed via `install()` (SDK), every request and response that passes through that LiteLLM stack is inspected.** Disable inspection by removing the plugin from your LiteLLM configuration, not by setting an env var.

> Pattern B (AI Defense Gateway) deployments do **not** need this plugin — the gateway inspects in-line. Don't load the plugin in a stack that already routes to AI Defense Gateway, or you will double-inspect.

---

## Usage Examples

### Programmatic install in a Python application

```python
from cisco_ai_defense_litellm import install, AIDefenseLiteLLMHook

hook: AIDefenseLiteLLMHook = install(
    block_mode="exception",
    scan_outputs=True,
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

Apache-2.0. See [LICENSE](https://github.com/cisco-aispg/ai-defense-litellm/blob/main/LICENSE).

---

## Support

- **Issues and feature requests**: [GitHub Issues](https://github.com/cisco-aispg/ai-defense-litellm/issues)
- **Source code**: [cisco-aispg/ai-defense-litellm](https://github.com/cisco-aispg/ai-defense-litellm)
- **Cisco AI Defense product docs**: [https://www.cisco.com/site/us/en/products/security/ai-defense/index.html](https://www.cisco.com/site/us/en/products/security/ai-defense/index.html)
