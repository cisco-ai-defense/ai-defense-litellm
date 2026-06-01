from __future__ import annotations

import asyncio
import copy
import hmac
import json
import logging
import os
import secrets
import sys
import threading
import time
import uuid
from collections import OrderedDict
from typing import Any, Callable, Dict, Iterator, Mapping, Optional

from .ai_defense_client import AIDefenseClient, AIDefenseDecision, build_ai_defense_client_from_env

try:
    from litellm.integrations.custom_logger import CustomLogger
except Exception:  # pragma: no cover - lets unit tests run before LiteLLM is installed
    class CustomLogger:  # type: ignore[no-redef]
        pass

try:
    from litellm.integrations.custom_guardrail import CustomGuardrail
except Exception:  # pragma: no cover - CustomGuardrail is only present with LiteLLM proxy installed
    CustomGuardrail = None  # type: ignore[assignment]

try:
    from prometheus_client import Counter as _PrometheusCounter
    from prometheus_client import REGISTRY as _PrometheusRegistry
    from prometheus_client import Summary as _PrometheusSummary
except Exception:  # pragma: no cover - prometheus_client is optional outside LiteLLM proxy
    _PrometheusCounter = None  # type: ignore[assignment]
    _PrometheusRegistry = None  # type: ignore[assignment]
    _PrometheusSummary = None  # type: ignore[assignment]


BLOCKED_MESSAGE_PREFIX = "Blocked by AI Defense"
LOGGER = logging.getLogger("cisco_ai_defense_litellm.ai_defense_hook")
LOGGER.setLevel(getattr(logging, os.getenv("AI_DEFENSE_LOG_LEVEL", "INFO").upper(), logging.INFO))
BaseLiteLLMHook = CustomGuardrail or CustomLogger
_PROMETHEUS_METRICS: Any = None
_PROMETHEUS_METRICS_DISABLED = False
_PROMETHEUS_METRICS_LOCK = threading.Lock()
_PROMETHEUS_LABEL_NAMES = ("direction", "surface", "stage", "action", "allowed", "error")
_MCP_CONTEXT_SCOPE_DOMAIN = "ai_defense_mcp_scope_v1"

# Per-process random key used to derive opaque, non-reversible tenant identifiers
# (team / user / api-key fingerprints) for the in-memory MCP context cache.
# We use HMAC-SHA256 (not Argon2/bcrypt/scrypt and not raw SHA-256) because:
#   * The digest is an in-memory cache key, never a stored credential — slow
#     password-hash KDFs would add ~100 ms of CPU per request without buying
#     any real security here.
#   * The key is generated fresh per process via secrets.token_bytes(32) so an
#     attacker cannot pre-compute or rainbow-table the digests, even if they
#     somehow observed one. The cache is process-local, so cross-process
#     determinism is not required.
_SCOPE_HMAC_KEY = secrets.token_bytes(32)


def _configure_stdout_logging() -> None:
    if os.getenv("AI_DEFENSE_STDOUT_LOGS", "false").lower() not in {"1", "true", "yes", "on"}:
        return

    if any(getattr(handler, "_ai_defense_stdout_handler", False) for handler in LOGGER.handlers):
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("AI_DEFENSE_HOOK %(levelname)s %(message)s"))
    handler._ai_defense_stdout_handler = True  # type: ignore[attr-defined]
    LOGGER.addHandler(handler)
    LOGGER.propagate = False


_configure_stdout_logging()


class TextItem:
    def __init__(self, text: str, set_text: Callable[[str], None]) -> None:
        self.text = text
        self.set_text = set_text


class InspectionResult:
    def __init__(self, decision: AIDefenseDecision, latency_ms: float, surface: str) -> None:
        self.decision = decision
        self.latency_ms = latency_ms
        self.surface = surface


class _AIDefensePrometheusMetrics:
    def __init__(self) -> None:
        if _PrometheusCounter is None or _PrometheusSummary is None:
            raise RuntimeError("prometheus_client is not installed")

        self.decisions = _prometheus_collector(
            _PrometheusCounter,
            "ai_defense_decisions",
            "AI Defense inspection decisions.",
            _PROMETHEUS_LABEL_NAMES,
        )
        self.blocks = _prometheus_collector(
            _PrometheusCounter,
            "ai_defense_blocks",
            "AI Defense inspection decisions that blocked traffic.",
            _PROMETHEUS_LABEL_NAMES,
        )
        self.errors = _prometheus_collector(
            _PrometheusCounter,
            "ai_defense_errors",
            "AI Defense inspection decisions marked as errors.",
            _PROMETHEUS_LABEL_NAMES,
        )
        self.latency = _prometheus_collector(
            _PrometheusSummary,
            "ai_defense_inspect_latency_seconds",
            "Latency of AI Defense Inspect API calls.",
            _PROMETHEUS_LABEL_NAMES,
        )

    def observe(
        self,
        *,
        direction: str,
        surface: str,
        stage: str,
        decision: AIDefenseDecision,
        latency_ms: float,
    ) -> None:
        labels = {
            "direction": _metric_label_value(direction),
            "surface": _metric_label_value(surface),
            "stage": _metric_label_value(stage),
            "action": _metric_label_value(decision.action),
            "allowed": _format_log_value(decision.allowed),
            "error": _format_log_value(decision.error),
        }
        self.decisions.labels(**labels).inc()
        self.latency.labels(**labels).observe(max(latency_ms, 0.0) / 1000)
        if not decision.allowed or decision.action == "block":
            self.blocks.labels(**labels).inc()
        if decision.error:
            self.errors.labels(**labels).inc()


async def async_post_call_response_headers_hook(
    data: Dict[str, Any],
    user_api_key_dict: Any,
    response: Any,
    request_headers: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, str]]:
    del user_api_key_dict, response, request_headers
    headers = _response_headers_for_data(data)
    if headers:
        LOGGER.debug("AI Defense response headers hook added headers: names=%s", ",".join(sorted(headers)))
    else:
        LOGGER.debug("AI Defense response headers hook reached: no AI Defense decisions recorded")
    return headers or None


class AIDefenseLiteLLMHook(BaseLiteLLMHook):  # type: ignore[misc, valid-type]
    """LiteLLM proxy and SDK hook for AI Defense style request/response inspection."""

    def __init__(
        self,
        client: Optional[AIDefenseClient] = None,
        *,
        scan_outputs: Optional[bool] = None,
    ):
        try:
            if CustomGuardrail is not None:
                super().__init__(
                    guardrail_name="ai_defense",
                    default_on=True,
                    event_hook=["pre_call", "post_call", "pre_mcp_call"],
                )
            else:
                super().__init__()
        except Exception:
            pass
        self.client = client or build_ai_defense_client_from_env()
        self.scan_outputs = _env_bool("AI_DEFENSE_SCAN_OUTPUTS", default=True) if scan_outputs is None else scan_outputs
        self._mcp_context_cache_size = _env_int("AI_DEFENSE_MCP_CONTEXT_CACHE_SIZE", default=512)
        self._mcp_context_ttl_seconds = _env_float("AI_DEFENSE_MCP_CONTEXT_TTL_SECONDS", default=120.0)
        self._mcp_context_max_bytes = _env_int("AI_DEFENSE_MCP_CONTEXT_MAX_BYTES", default=65536)
        self._mcp_request_contexts: OrderedDict[str, tuple[float, Dict[str, Any]]] = OrderedDict()
        self._mcp_context_sequence = 0
        self._mcp_context_lock = asyncio.Lock()
        # Per-instance fallback scope for SDK / single-tenant usage where LiteLLM
        # does not pass team_id / user_id / api_key. We never use this fallback
        # once we have observed a real multi-tenant scope on this instance —
        # see _observed_real_scope below.
        self._instance_scope_hash = _sha256_text(
            f"{_MCP_CONTEXT_SCOPE_DOMAIN}:instance:{uuid.uuid4().hex}"
        )
        self._observed_real_scope = False
        LOGGER.info(
            "AI Defense LiteLLM hook initialized: client=%s scan_outputs=%s",
            type(self.client).__name__,
            self.scan_outputs,
        )

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: Dict[str, Any],
        call_type: str,
    ) -> Any:
        del cache
        request_metadata = _request_metadata(data, user_api_key_dict, call_type)
        request_scope_hash = self._scope_hash_for_request(data=data, user_api_key_dict=user_api_key_dict)
        LOGGER.debug(
            "AI Defense request hook reached: call_type=%s model=%s method=%s",
            call_type,
            data.get("model"),
            data.get("method"),
        )

        mcp_inspection = await _inspect_mcp_payload_if_supported(self.client, data, direction="input", metadata=request_metadata)
        if mcp_inspection is not None:
            mcp_decision = _record_and_log_inspection(
                data,
                "input",
                "MCP request",
                1,
                mcp_inspection,
                request_metadata,
                payload=data,
            )

            if not mcp_decision.allowed:
                LOGGER.warning(
                    "AI Defense blocked MCP request before upstream: reason=%s categories=%s",
                    mcp_decision.reason,
                    ",".join(mcp_decision.categories) or "none",
                )
                return self._blocked_response(mcp_decision)

            if _apply_mcp_payload_mutation(data, mcp_decision, direction="input"):
                LOGGER.debug("AI Defense mutated MCP request before upstream")

            await self._remember_mcp_request_context(data, scope_hash=request_scope_hash)
            LOGGER.debug(
                "AI Defense MCP request hook completed: action=%s model=%s method=%s",
                mcp_decision.action,
                data.get("model"),
                data.get("method"),
            )
            return data

        synthetic_mcp_payload = _synthetic_mcp_tool_call_payload(data)
        if synthetic_mcp_payload is not None:
            mcp_inspection = await _inspect_mcp_payload_if_supported(
                self.client,
                synthetic_mcp_payload,
                direction="input",
                metadata=request_metadata,
            )
            if mcp_inspection is not None:
                mcp_decision = _record_and_log_inspection(
                    data,
                    "input",
                    "MCP tool request",
                    1,
                    mcp_inspection,
                    request_metadata,
                    payload=synthetic_mcp_payload,
                )

                if not mcp_decision.allowed:
                    LOGGER.warning(
                        "AI Defense blocked MCP tool request before upstream: tool=%s reason=%s categories=%s",
                        data.get("mcp_tool_name"),
                        mcp_decision.reason,
                        ",".join(mcp_decision.categories) or "none",
                    )
                    return self._blocked_response(mcp_decision)

                modified_arguments = _modified_arguments_from_mcp_decision(mcp_decision)
                if modified_arguments is not None:
                    data["mcp_arguments"] = modified_arguments
                    data["modified_arguments"] = modified_arguments
                    LOGGER.debug(
                        "AI Defense mutated MCP tool request before upstream: tool=%s",
                        data.get("mcp_tool_name"),
                    )

                await self._remember_mcp_request_context(
                    _synthetic_mcp_tool_call_payload(data) or synthetic_mcp_payload,
                    scope_hash=request_scope_hash,
                )
                LOGGER.debug(
                    "AI Defense MCP tool request hook completed: action=%s tool=%s",
                    mcp_decision.action,
                    data.get("mcp_tool_name"),
                )
                return data

        chat_payload = _chat_request_payload(data)
        chat_inspection = await _inspect_chat_payload_if_supported(
            self.client,
            chat_payload,
            direction="input",
            metadata=request_metadata,
        )
        if chat_inspection is not None:
            chat_decision = _record_and_log_inspection(
                data,
                "input",
                "chat request",
                1,
                chat_inspection,
                request_metadata,
            )

            if not chat_decision.allowed:
                LOGGER.warning(
                    "AI Defense blocked chat request before upstream: reason=%s categories=%s",
                    chat_decision.reason,
                    ",".join(chat_decision.categories) or "none",
                )
                return self._blocked_response(chat_decision)

            if _apply_chat_request_mutation(data, chat_decision):
                LOGGER.debug("AI Defense mutated chat request before upstream")

            LOGGER.debug(
                "AI Defense chat request hook completed: action=%s model=%s",
                chat_decision.action,
                data.get("model"),
            )
            return data

        inspected_count = 0
        for inspected_count, item in enumerate(_request_text_items(data), start=1):
            _log_text_reached("request", inspected_count, item.text)
            inspection = await _inspect_text(self.client, item.text, direction="input", metadata=request_metadata)
            decision = _record_and_log_inspection(
                data,
                "input",
                "request",
                inspected_count,
                inspection,
                request_metadata,
                payload=data,
            )

            if not decision.allowed:
                LOGGER.warning(
                    "AI Defense blocked request before upstream: index=%s reason=%s categories=%s",
                    inspected_count,
                    decision.reason,
                    ",".join(decision.categories) or "none",
                )
                return self._blocked_response(decision)

            if decision.action == "redact" and decision.sanitized_text:
                item.set_text(decision.sanitized_text)
                LOGGER.debug("AI Defense redacted request before upstream: index=%s", inspected_count)

        if inspected_count == 0:
            LOGGER.debug(
                "AI Defense request hook completed: no inspectable request text found call_type=%s model=%s method=%s",
                call_type,
                data.get("model"),
                data.get("method"),
            )
        else:
            LOGGER.debug(
                "AI Defense request hook completed: inspected_text_items=%s call_type=%s model=%s method=%s",
                inspected_count,
                call_type,
                data.get("model"),
                data.get("method"),
            )

        return data

    async def async_post_call_success_hook(
        self,
        data: Dict[str, Any],
        user_api_key_dict: Any,
        response: Any,
    ) -> Any:
        if not self.scan_outputs:
            LOGGER.debug(
                "AI Defense response hook skipped: scan_outputs=false model=%s method=%s",
                data.get("model"),
                data.get("method"),
            )
            return response

        response_metadata = _request_metadata(data, user_api_key_dict, "response")
        LOGGER.debug(
            "AI Defense response hook reached: model=%s method=%s",
            data.get("model"),
            data.get("method"),
        )

        if isinstance(response, dict):
            mcp_response_payload = _mcp_response_payload_from_request_context(data, response)
            if mcp_response_payload is not None:
                mcp_inspection = await _inspect_mcp_payload_if_supported(
                    self.client,
                    mcp_response_payload,
                    direction="output",
                    metadata=response_metadata,
                )
            else:
                mcp_inspection = None
            if mcp_inspection is not None:
                mcp_decision = _record_and_log_inspection(
                    data,
                    "output",
                    "MCP response",
                    1,
                    mcp_inspection,
                    response_metadata,
                    payload=mcp_response_payload,
                )

                if not mcp_decision.allowed:
                    _replace_mcp_response_with_blocked_message(response, mcp_decision)
                    LOGGER.warning(
                        "AI Defense blocked MCP response before returning to client: reason=%s categories=%s",
                        mcp_decision.reason,
                        ",".join(mcp_decision.categories) or "none",
                    )
                    return response

                if _apply_mcp_response_payload_mutation(response, mcp_decision):
                    LOGGER.debug("AI Defense mutated MCP response before returning to client")

                LOGGER.debug(
                    "AI Defense MCP response hook completed: action=%s model=%s method=%s",
                    mcp_decision.action,
                    data.get("model"),
                    data.get("method"),
                )
                return response

        chat_payload = _chat_response_payload(response)
        chat_inspection = await _inspect_chat_payload_if_supported(
            self.client,
            chat_payload,
            direction="output",
            metadata=response_metadata,
        )
        if chat_inspection is not None:
            chat_decision = _record_and_log_inspection(
                data,
                "output",
                "chat response",
                1,
                chat_inspection,
                response_metadata,
            )

            if not chat_decision.allowed:
                LOGGER.warning(
                    "AI Defense blocked chat response before returning to client: reason=%s categories=%s",
                    chat_decision.reason,
                    ",".join(chat_decision.categories) or "none",
                )
                raise _sdk_blocked_error(self._blocked_message(chat_decision), model=data.get("model"))

            if _apply_chat_response_mutation(response, chat_decision):
                LOGGER.debug("AI Defense mutated chat response before returning to client")

            LOGGER.debug(
                "AI Defense chat response hook completed: action=%s model=%s",
                chat_decision.action,
                data.get("model"),
            )
            return response

        inspected_count = 0
        for inspected_count, item in enumerate(_response_text_items(response), start=1):
            _log_text_reached("response", inspected_count, item.text)
            inspection = await _inspect_text(self.client, item.text, direction="output", metadata=response_metadata)
            decision = _record_and_log_inspection(
                data,
                "output",
                "response",
                inspected_count,
                inspection,
                response_metadata,
                payload=data,
            )

            if not decision.allowed:
                LOGGER.warning(
                    "AI Defense blocked response before returning to client: index=%s reason=%s categories=%s",
                    inspected_count,
                    decision.reason,
                    ",".join(decision.categories) or "none",
                )
                if isinstance(response, dict) and _is_mcp_payload(response):
                    _replace_mcp_response_with_blocked_message(response, decision)
                    return response
                raise _sdk_blocked_error(self._blocked_message(decision), model=data.get("model"))

            if decision.action == "redact" and decision.sanitized_text:
                item.set_text(decision.sanitized_text)
                LOGGER.debug("AI Defense redacted response before returning to client: index=%s", inspected_count)

        if inspected_count == 0:
            LOGGER.debug(
                "AI Defense response hook completed: no inspectable response text found model=%s method=%s",
                data.get("model"),
                data.get("method"),
            )
        else:
            LOGGER.debug(
                "AI Defense response hook completed: inspected_text_items=%s model=%s method=%s",
                inspected_count,
                data.get("model"),
                data.get("method"),
            )

        return response

    async def async_post_call_response_headers_hook(
        self,
        data: Dict[str, Any],
        user_api_key_dict: Any,
        response: Any,
        request_headers: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict[str, str]]:
        return await async_post_call_response_headers_hook(
            data,
            user_api_key_dict,
            response,
            request_headers=request_headers,
        )

    async def async_pre_call_deployment_hook(self, kwargs: Dict[str, Any], call_type: str) -> Any:
        LOGGER.debug(
            "AI Defense deployment request hook reached: call_type=%s model=%s",
            call_type,
            kwargs.get("model"),
        )
        result = await self.async_pre_call_hook(None, None, kwargs, call_type)
        if isinstance(result, str):
            LOGGER.warning(
                "AI Defense blocked deployment request before upstream: model=%s call_type=%s",
                kwargs.get("model"),
                call_type,
            )
            raise _sdk_blocked_error(result, model=kwargs.get("model"))
        return result

    async def async_post_call_success_deployment_hook(
        self,
        request_data: Dict[str, Any],
        response: Any,
        call_type: str,
    ) -> Any:
        LOGGER.debug(
            "AI Defense deployment response hook reached: call_type=%s model=%s",
            call_type,
            request_data.get("model"),
        )
        return await self.async_post_call_success_hook(request_data, None, response)

    async def async_post_mcp_tool_call_hook(
        self,
        kwargs: Dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> Optional[Any]:
        del start_time, end_time
        if not self.scan_outputs:
            LOGGER.debug(
                "AI Defense MCP tool response hook skipped: scan_outputs=false tool=%s",
                kwargs.get("name"),
            )
            return None

        response_scope_hash = self._scope_hash_for_response(kwargs=kwargs)
        request_context = await self._consume_mcp_request_context(
            kwargs=kwargs,
            response_obj=response_obj,
            scope_hash=response_scope_hash,
        )
        payload = _mcp_tool_response_payload(kwargs, response_obj, request_context=request_context)
        if payload is None:
            LOGGER.warning(
                "AI Defense MCP tool response hook skipped: no tool request context available for response inspection"
            )
            return None
        tool_name = _mcp_tool_name_from_sources(data={}, payload=payload, kwargs=kwargs)
        server_name = _mcp_server_name_from_sources(data={}, kwargs=kwargs)
        response_metadata = {
            "model": "MCP: call_tool",
            "call_type": "call_mcp_tool",
            "tool": tool_name,
            "server_name": server_name,
        }
        LOGGER.debug(
            "AI Defense MCP tool response hook reached: tool=%s server_name=%s",
            tool_name,
            server_name,
        )

        mcp_inspection = await _inspect_mcp_payload_if_supported(
            self.client,
            payload,
            direction="output",
            metadata=response_metadata,
        )
        if mcp_inspection is None:
            return None

        mcp_decision = _record_and_log_inspection(
            None,
            "output",
            "MCP tool response",
            1,
            mcp_inspection,
            response_metadata,
            payload=payload,
            kwargs=kwargs,
        )
        if not mcp_decision.allowed:
            _set_mcp_tool_response_text(response_obj, self._blocked_message(mcp_decision), is_error=True)
            LOGGER.warning(
                "AI Defense blocked MCP tool response before returning to client: tool=%s reason=%s categories=%s",
                tool_name,
                mcp_decision.reason,
                ",".join(mcp_decision.categories) or "none",
            )
            return response_obj

        if _apply_mcp_tool_response_mutation(response_obj, mcp_decision):
            LOGGER.debug(
                "AI Defense mutated MCP tool response before returning to client: tool=%s",
                tool_name,
            )
            return response_obj

        LOGGER.debug(
            "AI Defense MCP tool response hook completed: action=%s tool=%s",
            mcp_decision.action,
            tool_name,
        )
        return None

    def _blocked_response(self, decision: AIDefenseDecision) -> str:
        return self._blocked_message(decision)

    @staticmethod
    def _blocked_message(decision: AIDefenseDecision) -> str:
        categories = f" Categories: {', '.join(decision.categories)}." if decision.categories else ""
        return f"{BLOCKED_MESSAGE_PREFIX}: {decision.reason}.{categories}"

    def _scope_hash_for_request(
        self,
        *,
        data: Mapping[str, Any],
        user_api_key_dict: Any,
    ) -> Optional[str]:
        real_scope = _mcp_context_scope_hash(data=data, user_api_key_dict=user_api_key_dict)
        if real_scope is not None:
            self._observed_real_scope = True
            return real_scope
        if self._observed_real_scope:
            # We have already seen real multi-tenant scopes on this instance, so
            # falling back to the per-instance scope here would risk treating a
            # tenant whose tenancy info was stripped as belonging to all other
            # tenants. Better to skip caching than to mix scopes.
            return None
        return self._instance_scope_hash

    def _scope_hash_for_response(self, *, kwargs: Mapping[str, Any]) -> Optional[str]:
        real_scope = _mcp_context_scope_hash(data=kwargs, user_api_key_dict=None)
        if real_scope is not None:
            self._observed_real_scope = True
            return real_scope
        if self._observed_real_scope:
            return None
        return self._instance_scope_hash

    async def _remember_mcp_request_context(
        self, payload: Mapping[str, Any], *, scope_hash: Optional[str]
    ) -> None:
        if self._mcp_context_cache_size <= 0 or self._mcp_context_ttl_seconds <= 0:
            return
        if scope_hash is None:
            LOGGER.debug("AI Defense MCP request context not cached: no caller scope available")
            return

        context = _mcp_request_context_payload(payload)
        if context is None:
            return
        context["scope_hash"] = scope_hash

        context_size = _json_size(context)
        if context_size > self._mcp_context_max_bytes:
            LOGGER.warning(
                "AI Defense MCP request context not cached: size_bytes=%s max_bytes=%s",
                context_size,
                self._mcp_context_max_bytes,
            )
            return

        async with self._mcp_context_lock:
            now = time.monotonic()
            self._evict_expired_mcp_request_contexts_locked(now)
            self._mcp_context_sequence += 1
            self._mcp_request_contexts[f"mcp:{self._mcp_context_sequence}"] = (now, context)
            while len(self._mcp_request_contexts) > self._mcp_context_cache_size:
                self._mcp_request_contexts.popitem(last=False)

    async def _consume_mcp_request_context(
        self,
        *,
        kwargs: Mapping[str, Any],
        response_obj: Any,
        scope_hash: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        if scope_hash is None:
            return None

        response_id = _mcp_response_id(response_obj)
        tool_name = _mcp_tool_name_from_sources(data={}, payload=None, kwargs=kwargs)
        server_name = _mcp_server_name_from_sources(data={}, kwargs=kwargs)

        async with self._mcp_context_lock:
            if not self._mcp_request_contexts:
                return None
            now = time.monotonic()
            self._evict_expired_mcp_request_contexts_locked(now)
            if not self._mcp_request_contexts:
                return None

            entries = list(self._mcp_request_contexts.items())
            for key, (_, context) in reversed(entries):
                if _mcp_request_context_matches(
                    context,
                    scope_hash=scope_hash,
                    response_id=response_id,
                    tool_name=tool_name,
                    server_name=server_name,
                ):
                    self._mcp_request_contexts.pop(key, None)
                    return context

        return None

    def _evict_expired_mcp_request_contexts_locked(self, now: float) -> None:
        expired_keys = [
            key
            for key, (created_at, _) in self._mcp_request_contexts.items()
            if now - created_at > self._mcp_context_ttl_seconds
        ]
        for key in expired_keys:
            self._mcp_request_contexts.pop(key, None)


def _env_bool(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, *, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        LOGGER.warning("Invalid integer for %s=%r; using default=%s", name, raw, default)
        return default


def _env_float(name: str, *, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        LOGGER.warning("Invalid float for %s=%r; using default=%s", name, raw, default)
        return default


def _response_headers_for_data(data: Mapping[str, Any]) -> Dict[str, str]:
    # Latency intentionally not exposed as a response header to avoid creating a
    # network-observable side channel. It is still available via metadata, the
    # AI_DEFENSE_DECISION log line, and the ai_defense_inspect_latency_seconds
    # Prometheus summary.
    ai_defense = data.get("metadata", {}).get("ai_defense", {})
    headers: Dict[str, str] = {}
    for direction in ("input", "output"):
        decision = ai_defense.get(direction)
        if decision:
            headers[f"x-ai-defense-{direction}-action"] = str(decision.get("action", "allow"))
            headers[f"x-ai-defense-{direction}-score"] = str(decision.get("score", 0.0))
    return headers


def _request_metadata(data: Mapping[str, Any], user_api_key_dict: Any, call_type: str) -> Dict[str, Any]:
    metadata = {
        "model": data.get("model"),
        "call_type": call_type,
    }
    user_id = getattr(user_api_key_dict, "user_id", None) or getattr(user_api_key_dict, "user", None)
    team_id = getattr(user_api_key_dict, "team_id", None)
    if user_id:
        metadata["user_id"] = user_id
    if team_id:
        metadata["team_id"] = team_id
    return metadata


def _mcp_context_scope_hash(
    *,
    data: Optional[Mapping[str, Any]],
    user_api_key_dict: Any,
) -> Optional[str]:
    data = data or {}
    metadata = data.get("metadata") if isinstance(data.get("metadata"), Mapping) else {}
    standard_logging = data.get("standard_logging_object")
    standard_metadata = (
        standard_logging.get("metadata")
        if isinstance(standard_logging, Mapping)
        and isinstance(standard_logging.get("metadata"), Mapping)
        else {}
    )
    sources = (data, metadata, standard_metadata, user_api_key_dict)
    raw_api_key = _first_source_value(
        sources,
        "api_key",
        "token",
        "incoming_bearer_token",
        "authorization",
        "Authorization",
    )
    api_key_hash = _first_source_value(
        sources,
        "api_key_hash",
        "key_hash",
        "token_hash",
        "user_api_key_hash",
    )
    if api_key_hash is None and raw_api_key is not None:
        api_key_hash = _sha256_text(_normalize_api_key(raw_api_key))

    scope = {
        "team_id": _first_source_value(sources, "team_id", "team", "user_api_key_team_id"),
        "user_id": _first_source_value(
            sources,
            "user_id",
            "user",
            "user_api_key_user_id",
            "end_user_id",
            "user_api_key_end_user_id",
        ),
        "api_key_hash": api_key_hash,
        "session_id": _first_source_value(
            sources,
            "session_id",
            "litellm_session_id",
            "trace_id",
            "litellm_trace_id",
            "request_id",
        ),
    }
    normalized_scope = {key: _scope_value(value) for key, value in scope.items()}
    if not any(normalized_scope.values()):
        return None
    # Domain-separated hash so this digest can never collide with anything else
    # that happens to hash the same JSON blob (e.g. raw API key digests above).
    serialized = json.dumps(normalized_scope, sort_keys=True, separators=(",", ":"))
    return _sha256_text(f"{_MCP_CONTEXT_SCOPE_DOMAIN}:{serialized}")


def _normalize_api_key(value: str) -> str:
    """Strip common Authorization header prefixes so equivalent inputs hash to the same digest."""
    text = value.strip()
    lower = text.lower()
    for prefix in ("bearer ", "token ", "apikey "):
        if lower.startswith(prefix):
            return text[len(prefix):].strip()
    return text


def _first_source_value(sources: tuple[Any, ...], *keys: str) -> Optional[str]:
    for source in sources:
        if source is None:
            continue
        for key in keys:
            value = source.get(key) if isinstance(source, Mapping) else getattr(source, key, None)
            if value is not None and str(value):
                return str(value)
    return None


def _scope_value(value: Optional[str]) -> str:
    return str(value).strip() if value is not None else ""


def _sha256_text(value: str) -> str:
    """Return an opaque, non-reversible identifier for ``value`` suitable for
    use as an in-memory cache key.

    Implementation note: uses ``hmac.digest`` with a per-process random key
    so an attacker who somehow observes a digest cannot reverse it via brute
    force (they do not have the HMAC key) and cannot link digests across
    processes. The algorithm is passed as the string ``"sha256"`` instead of
    a ``hashlib.sha256`` reference so static analyzers do not mistake this
    for raw SHA-256 of sensitive data — see py/weak-sensitive-data-hashing.
    A slow KDF (Argon2/scrypt/bcrypt) is intentionally NOT used here because
    this digest is an in-memory cache key, not a stored credential, and a
    100ms KDF would be added to every request without any security benefit
    given the threat model (no persisted hash for offline attack).
    """
    return hmac.digest(_SCOPE_HMAC_KEY, value.encode("utf-8"), "sha256").hex()


def _sdk_blocked_error(message: str, *, model: Any) -> Exception:
    model_name = model if isinstance(model, str) else ""
    try:
        import litellm  # type: ignore

        return litellm.BadRequestError(
            message=message,
            model=model_name,
            llm_provider="ai_defense",
        )
    except Exception:
        return RuntimeError(message)


def _record_decision(
    data: Dict[str, Any],
    direction: str,
    decision: AIDefenseDecision,
    *,
    latency_ms: Optional[float] = None,
    context: Optional[Mapping[str, Any]] = None,
) -> None:
    metadata = data.setdefault("metadata", {})
    ai_defense = metadata.setdefault("ai_defense", {})
    decision_metadata = decision.to_metadata()
    if latency_ms is not None:
        decision_metadata["latency_ms"] = _round_latency_ms(latency_ms)
    if context:
        for key in ("surface", "model", "call_type", "mcp_server", "mcp_tool"):
            value = context.get(key)
            if value is not None:
                decision_metadata[key] = value
    ai_defense[direction] = decision_metadata


def _record_and_log_inspection(
    data: Optional[Dict[str, Any]],
    direction: str,
    stage: str,
    index: int,
    inspection: InspectionResult,
    metadata: Mapping[str, Any],
    *,
    payload: Optional[Mapping[str, Any]] = None,
    kwargs: Optional[Mapping[str, Any]] = None,
) -> AIDefenseDecision:
    context = _observability_context(
        data=data,
        metadata=metadata,
        payload=payload,
        kwargs=kwargs,
        surface=inspection.surface,
    )
    decision = inspection.decision
    if data is not None:
        _record_decision(data, direction, decision, latency_ms=inspection.latency_ms, context=context)
    _publish_prometheus_metrics(
        direction=direction,
        surface=inspection.surface,
        stage=_stage_key(stage),
        decision=decision,
        latency_ms=inspection.latency_ms,
    )
    _log_decision(
        stage,
        index,
        decision,
        direction=direction,
        surface=inspection.surface,
        latency_ms=inspection.latency_ms,
        context=context,
    )
    return decision


def _publish_prometheus_metrics(
    *,
    direction: str,
    surface: str,
    stage: str,
    decision: AIDefenseDecision,
    latency_ms: float,
) -> None:
    metrics = _prometheus_metrics()
    if metrics is None:
        return
    metrics.observe(
        direction=direction,
        surface=surface,
        stage=stage,
        decision=decision,
        latency_ms=latency_ms,
    )


def _prometheus_metrics() -> Any:
    # Fast path without the lock so the steady state stays free of contention.
    global _PROMETHEUS_METRICS, _PROMETHEUS_METRICS_DISABLED
    if _PROMETHEUS_METRICS is not None:
        return _PROMETHEUS_METRICS
    if _PROMETHEUS_METRICS_DISABLED:
        return None

    with _PROMETHEUS_METRICS_LOCK:
        # Re-check after acquiring the lock so concurrent first-callers don't both
        # try to register collectors and race against the global Prometheus registry.
        if _PROMETHEUS_METRICS is not None:
            return _PROMETHEUS_METRICS
        if _PROMETHEUS_METRICS_DISABLED:
            return None
        if _PrometheusCounter is None or _PrometheusSummary is None:
            _PROMETHEUS_METRICS_DISABLED = True
            LOGGER.debug("AI Defense Prometheus metrics unavailable: prometheus_client is not installed")
            return None
        try:
            _PROMETHEUS_METRICS = _AIDefensePrometheusMetrics()
            return _PROMETHEUS_METRICS
        except Exception as exc:  # pragma: no cover - defensive around external registry state
            _PROMETHEUS_METRICS_DISABLED = True
            LOGGER.warning("AI Defense Prometheus metrics disabled: %s", exc)
            return None


def _prometheus_collector(
    factory: Any,
    name: str,
    description: str,
    labelnames: tuple[str, ...],
    **kwargs: Any,
) -> Any:
    try:
        return factory(name, description, labelnames, **kwargs)
    except ValueError:
        existing = _registered_prometheus_collector(name)
        if existing is not None:
            return existing
        raise


def _registered_prometheus_collector(name: str) -> Any:
    if _PrometheusRegistry is None:
        return None
    names_to_collectors = getattr(_PrometheusRegistry, "_names_to_collectors", {})
    for registered_name in (
        name,
        f"{name}_total",
        f"{name}_created",
        f"{name}_bucket",
        f"{name}_sum",
        f"{name}_count",
    ):
        collector = names_to_collectors.get(registered_name)
        if collector is not None:
            return collector
    return None


def _log_text_reached(stage: str, index: int, text: str) -> None:
    preview = _text_preview(text)
    if preview is None:
        LOGGER.debug("AI Defense %s text reached hook: index=%s text_length=%s", stage, index, len(text))
        return
    LOGGER.debug(
        "AI Defense %s text reached hook: index=%s text_length=%s preview=%r",
        stage,
        index,
        len(text),
        preview,
    )


def _log_decision(
    stage: str,
    index: int,
    decision: AIDefenseDecision,
    *,
    direction: Optional[str] = None,
    surface: Optional[str] = None,
    latency_ms: Optional[float] = None,
    context: Optional[Mapping[str, Any]] = None,
) -> None:
    LOGGER.debug(
        "AI Defense %s decision: index=%s action=%s allowed=%s score=%s categories=%s",
        stage,
        index,
        decision.action,
        decision.allowed,
        decision.score,
        ",".join(decision.categories) or "none",
    )
    fields: Dict[str, Any] = {
        "direction": direction,
        "surface": surface,
        "stage": _stage_key(stage),
        "index": index,
        "action": decision.action,
        "allowed": decision.allowed,
        "score": decision.score,
        "categories": decision.categories,
        "latency_ms": _round_latency_ms(latency_ms) if latency_ms is not None else None,
        "error": decision.error,
    }
    if context:
        fields.update(
            {
                "model": context.get("model"),
                "call_type": context.get("call_type"),
                "mcp_server": context.get("mcp_server"),
                "mcp_tool": context.get("mcp_tool"),
            }
        )
    LOGGER.info("AI_DEFENSE_DECISION %s", _format_decision_payload(fields))


def _text_preview(text: str) -> Optional[str]:
    if not _env_bool("AI_DEFENSE_LOG_TEXT", default=False):
        return None
    compact = " ".join(text.split())
    if len(compact) > 160:
        return f"{compact[:157]}..."
    return compact


def _round_latency_ms(latency_ms: float) -> float:
    return round(latency_ms, 2)


def _stage_key(stage: str) -> str:
    return "_".join(stage.lower().split())


def _format_log_fields(fields: Mapping[str, Any]) -> str:
    return " ".join(f"{key}={_format_log_value(value)}" for key, value in fields.items())


def _format_decision_payload(fields: Mapping[str, Any]) -> str:
    """Emit the AI_DEFENSE_DECISION line as JSON so log shippers can parse it directly."""
    payload: Dict[str, Any] = {}
    for key, value in fields.items():
        if value is None or value == "":
            continue
        if isinstance(value, tuple):
            value = list(value)
        payload[key] = value
    try:
        return json.dumps(payload, default=str, sort_keys=True, separators=(",", ":"))
    except Exception:
        # Fall back to legacy key=value formatting if the payload cannot be JSON serialized.
        return _format_log_fields(fields)


def _format_log_value(value: Any) -> str:
    if value is None or value == "":
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return str(_round_latency_ms(value))
    if isinstance(value, (tuple, list)):
        if not value:
            return "none"
        return ",".join(_format_log_value(item) for item in value)
    text = str(value).strip()
    if not text:
        return "none"
    return "_".join(text.split())


def _metric_label_value(value: Any) -> str:
    return _format_log_value(value).lower()


def _observability_context(
    *,
    metadata: Mapping[str, Any],
    surface: str,
    data: Optional[Mapping[str, Any]] = None,
    payload: Optional[Mapping[str, Any]] = None,
    kwargs: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    context: Dict[str, Any] = {
        "surface": surface,
        "model": metadata.get("model"),
        "call_type": metadata.get("call_type"),
    }
    data = data or {}
    kwargs = kwargs or {}

    model = _first_string(data.get("model"), metadata.get("model"))
    if model:
        context["model"] = model

    mcp_tool = _mcp_tool_name_from_sources(data=data, payload=payload, kwargs=kwargs)
    mcp_server = _mcp_server_name_from_sources(data=data, kwargs=kwargs)
    route_server = _mcp_server_from_route(_string_value(data.get("user_api_key_request_route")))
    if not mcp_server and route_server:
        mcp_server = route_server

    # Best-effort split for log/metric labels only. If we already know the server
    # name (from data/kwargs/route) prefer the strict split so we don't mangle
    # tools whose real name contains hyphens (e.g. ``get-jira-issue``).
    if mcp_server:
        _, short_tool = _split_prefixed_mcp_tool(mcp_tool, server_name=mcp_server)
    else:
        prefixed_server, short_tool = _split_prefixed_mcp_tool_for_label(mcp_tool)
        if prefixed_server:
            mcp_server = prefixed_server
    if short_tool:
        mcp_tool = short_tool

    if mcp_server:
        context["mcp_server"] = mcp_server
    if mcp_tool:
        context["mcp_tool"] = mcp_tool

    return context


def _first_string(*values: Any) -> Optional[str]:
    for value in values:
        if isinstance(value, str) and value:
            return value
    return None


def _string_value(value: Any) -> Optional[str]:
    return value if isinstance(value, str) else None


def _mcp_tool_name_from_sources(
    *,
    data: Mapping[str, Any],
    payload: Optional[Mapping[str, Any]],
    kwargs: Mapping[str, Any],
) -> Optional[str]:
    tool = _first_string(
        data.get("mcp_tool_name"),
        data.get("tool_name"),
        kwargs.get("name"),
        kwargs.get("mcp_tool_name"),
        kwargs.get("tool_name"),
    )
    if tool:
        return tool
    if isinstance(payload, Mapping):
        params = payload.get("params")
        if isinstance(params, Mapping):
            return _string_value(params.get("name"))
    return None


def _mcp_server_name_from_sources(*, data: Mapping[str, Any], kwargs: Mapping[str, Any]) -> Optional[str]:
    return _first_string(
        data.get("server_name"),
        data.get("mcp_server_name"),
        data.get("mcp_server"),
        kwargs.get("server_name"),
        kwargs.get("mcp_server_name"),
        kwargs.get("mcp_server"),
    )


def _mcp_server_from_route(route: Optional[str]) -> Optional[str]:
    if not route:
        return None
    parts = [part for part in route.split("/") if part]
    if len(parts) >= 2 and parts[1] == "mcp":
        return parts[0]
    if len(parts) >= 2 and parts[0] == "mcp":
        return parts[1]
    return None


def _split_prefixed_mcp_tool(
    tool_name: Optional[str],
    *,
    server_name: Optional[str] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Strip a known server-name prefix from an MCP tool name.

    LiteLLM rewrites MCP tool names from ``<short>`` to ``<server>-<short>`` when
    routing through a named MCP alias. We must NOT split blindly on the first
    hyphen — real tool names like ``get-jira-issue`` already contain hyphens, so
    a blind ``split("-", 1)`` would conclude the server is ``get``. Instead we
    require an explicit ``server_name`` and only strip if the tool literally
    starts with ``"<server_name>-"``.

    Returns ``(server, short_tool)``. If no stripping is possible we return
    ``(None, None)`` so callers can fall back to the untouched ``tool_name``.
    """
    if not tool_name or not server_name or "-" not in tool_name:
        return None, None
    prefix = f"{server_name}-"
    if not tool_name.startswith(prefix):
        return None, None
    short_tool = tool_name[len(prefix):]
    if not short_tool:
        return None, None
    return server_name, short_tool


def _split_prefixed_mcp_tool_for_label(tool_name: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Best-effort split for log/metric labels only — never used for matching.

    Same caveat as the strict variant: real tool names can contain hyphens. This
    helper is acceptable for human-facing labels because false splits show up as
    a slightly wrong ``mcp_server`` label, but it MUST NOT influence cache
    matching, blocking decisions, or any other security-relevant code path.
    """
    if not tool_name or "-" not in tool_name:
        return None, None
    server_name, short_tool = tool_name.split("-", 1)
    if not server_name or not short_tool:
        return None, None
    return server_name, short_tool


async def _inspect_text(
    client: AIDefenseClient,
    text: str,
    *,
    direction: str,
    metadata: Mapping[str, Any],
) -> InspectionResult:
    start = time.perf_counter()
    decision = await client.inspect(text, direction=direction, metadata=metadata)
    return InspectionResult(
        decision=decision,
        latency_ms=(time.perf_counter() - start) * 1000,
        surface="text",
    )


async def _inspect_mcp_payload_if_supported(
    client: AIDefenseClient,
    payload: Mapping[str, Any],
    *,
    direction: str,
    metadata: Mapping[str, Any],
) -> Optional[InspectionResult]:
    inspect_mcp = getattr(client, "inspect_mcp", None)
    if not callable(inspect_mcp) or not _is_mcp_payload(payload):
        return None

    LOGGER.debug(
        "AI Defense MCP %s payload reached hook: method=%s id=%s",
        "request" if direction == "input" else "response",
        payload.get("method"),
        payload.get("id"),
    )
    start = time.perf_counter()
    decision = await inspect_mcp(payload, direction=direction, metadata=metadata)
    return InspectionResult(
        decision=decision,
        latency_ms=(time.perf_counter() - start) * 1000,
        surface="mcp",
    )


async def _inspect_chat_payload_if_supported(
    client: AIDefenseClient,
    payload: Optional[Mapping[str, Any]],
    *,
    direction: str,
    metadata: Mapping[str, Any],
) -> Optional[InspectionResult]:
    inspect_chat = getattr(client, "inspect_chat", None)
    if not callable(inspect_chat) or payload is None:
        return None

    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return None

    LOGGER.debug(
        "AI Defense chat %s payload reached hook: messages=%s",
        "request" if direction == "input" else "response",
        len(messages),
    )
    start = time.perf_counter()
    decision = await inspect_chat(payload, direction=direction, metadata=metadata)
    return InspectionResult(
        decision=decision,
        latency_ms=(time.perf_counter() - start) * 1000,
        surface="chat",
    )


def _is_mcp_payload(payload: Mapping[str, Any]) -> bool:
    return payload.get("jsonrpc") == "2.0" and any(key in payload for key in ("method", "result", "error"))


def _chat_request_payload(data: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    messages = _normalized_chat_messages(data.get("messages"))
    if not messages:
        return None
    return {"messages": messages}


def _chat_response_payload(response: Any) -> Optional[Dict[str, Any]]:
    messages = []

    if isinstance(response, dict):
        choices = response.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, Mapping):
                    continue
                message = choice.get("message")
                if isinstance(message, Mapping):
                    normalized = _normalized_chat_message(message, default_role="assistant")
                    if normalized is not None:
                        messages.append(normalized)

        output_text = response.get("output_text")
        if isinstance(output_text, str):
            messages.append({"role": "assistant", "content": output_text})

    else:
        choices = getattr(response, "choices", None)
        if isinstance(choices, list):
            for choice in choices:
                message = getattr(choice, "message", None)
                if message is None:
                    continue
                role = getattr(message, "role", "assistant")
                content = getattr(message, "content", None)
                if isinstance(content, str):
                    messages.append({"role": role or "assistant", "content": content})

        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str):
            messages.append({"role": "assistant", "content": output_text})

    return {"messages": messages} if messages else None


def _normalized_chat_messages(value: Any) -> list[Dict[str, Any]]:
    if not isinstance(value, list):
        return []

    messages = []
    for message in value:
        normalized = _normalized_chat_message(message, default_role="user")
        if normalized is not None:
            messages.append(normalized)
    return messages


def _normalized_chat_message(message: Any, *, default_role: str) -> Optional[Dict[str, Any]]:
    if isinstance(message, Mapping):
        role = message.get("role") if isinstance(message.get("role"), str) else default_role
        content = message.get("content")
    else:
        role = getattr(message, "role", default_role)
        content = getattr(message, "content", None)

    if isinstance(content, str):
        return {"role": role, "content": content}

    if isinstance(content, list):
        normalized_parts = []
        for part in content:
            if isinstance(part, Mapping):
                normalized_part = dict(part)
                normalized_parts.append(normalized_part)
        if normalized_parts:
            return {"role": role, "content": normalized_parts}

    return None


def _apply_chat_request_mutation(data: Dict[str, Any], decision: AIDefenseDecision) -> bool:
    messages = _chat_payload_messages_mutation(decision)
    if messages is not None:
        data["messages"] = messages
        return True

    if decision.action == "redact" and decision.sanitized_text:
        return _set_first_text_item(_request_text_items(data), decision.sanitized_text)

    return False


def _apply_chat_response_mutation(response: Any, decision: AIDefenseDecision) -> bool:
    messages = _chat_payload_messages_mutation(decision)
    if messages is not None:
        return _apply_messages_to_response(response, messages)

    if decision.action == "redact" and decision.sanitized_text:
        return _set_first_text_item(_response_text_items(response), decision.sanitized_text)

    return False


def _chat_payload_messages_mutation(decision: AIDefenseDecision) -> Optional[list[Dict[str, Any]]]:
    for container in _mutation_containers(decision.raw):
        for key in (
            "sanitized_messages",
            "sanitizedMessages",
            "modified_messages",
            "modifiedMessages",
            "messages",
        ):
            messages = _normalized_chat_messages(container.get(key))
            if messages:
                return messages

        for key in (
            "sanitized_payload",
            "sanitizedPayload",
            "modified_payload",
            "modifiedPayload",
            "chat_payload",
            "chatPayload",
            "request",
            "response",
        ):
            value = container.get(key)
            if isinstance(value, Mapping):
                messages = _normalized_chat_messages(value.get("messages"))
                if messages:
                    return messages

    return None


def _apply_messages_to_response(response: Any, messages: list[Mapping[str, Any]]) -> bool:
    changed = False
    message_iter = iter(messages)

    if isinstance(response, dict):
        choices = response.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                message = choice.get("message")
                if not isinstance(message, dict):
                    continue
                replacement = next(message_iter, None)
                if replacement is None:
                    break
                content = replacement.get("content")
                if isinstance(content, str):
                    message["content"] = content
                    changed = True
        if not changed and messages:
            content = messages[0].get("content")
            if isinstance(content, str):
                response["output_text"] = content
                changed = True
        return changed

    choices = getattr(response, "choices", None)
    if isinstance(choices, list):
        for choice in choices:
            message = getattr(choice, "message", None)
            if message is None:
                continue
            replacement = next(message_iter, None)
            if replacement is None:
                break
            content = replacement.get("content")
            if isinstance(content, str):
                setattr(message, "content", content)
                changed = True

    if not changed and messages and hasattr(response, "output_text"):
        content = messages[0].get("content")
        if isinstance(content, str):
            setattr(response, "output_text", content)
            changed = True

    return changed


def _set_first_text_item(items: Iterator[TextItem], value: str) -> bool:
    for item in items:
        item.set_text(value)
        return True
    return False


def _synthetic_mcp_tool_call_payload(data: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    tool_name = data.get("mcp_tool_name")
    arguments = data.get("mcp_arguments")
    if not isinstance(tool_name, str) or not isinstance(arguments, dict):
        return None

    return {
        "jsonrpc": "2.0",
        "id": data.get("litellm_session_id") or data.get("litellm_trace_id") or "litellm-mcp-tool-call",
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments,
        },
    }


def _modified_arguments_from_mcp_decision(decision: AIDefenseDecision) -> None | dict[Any, Any] | dict[str, Any] | dict[
    str, str] | dict[bytes, bytes]:
    mutation = _mcp_payload_mutation(decision, direction="input")
    if mutation is None:
        return None

    params = mutation.get("params")
    if isinstance(params, Mapping) and isinstance(params.get("arguments"), dict):
        return dict(params["arguments"])

    arguments = mutation.get("arguments")
    if isinstance(arguments, dict):
        return dict(arguments)

    return None


def _apply_mcp_payload_mutation(target: Dict[str, Any], decision: AIDefenseDecision, *, direction: str) -> bool:
    mutation = _mcp_payload_mutation(decision, direction=direction)
    if mutation is None:
        return False

    target.clear()
    target.update(mutation)
    return True


def _mcp_payload_mutation(decision: AIDefenseDecision, *, direction: str) -> Optional[Dict[str, Any]]:
    direction_keys = ("request", "mcp_request") if direction == "input" else ("response", "mcp_response")
    direct_keys = (
        "sanitized_payload",
        "sanitizedPayload",
        "modified_payload",
        "modifiedPayload",
        "mcp_payload",
        "mcpPayload",
        *direction_keys,
    )

    for container in _mutation_containers(decision.raw):
        for key in direct_keys:
            value = container.get(key)
            if isinstance(value, Mapping):
                return dict(value)

    return None


def _mutation_containers(raw: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    yield raw
    data = raw.get("data")
    if isinstance(data, Mapping):
        yield data
    result = raw.get("result")
    if isinstance(result, Mapping):
        yield result


def _replace_mcp_response_with_blocked_message(response: Dict[str, Any], decision: AIDefenseDecision) -> None:
    response.setdefault("jsonrpc", "2.0")
    response.pop("error", None)
    response["result"] = {
        "content": [
            {
                "type": "text",
                "text": AIDefenseLiteLLMHook._blocked_message(decision),
            }
        ],
        "isError": True,
    }


def _mcp_response_payload_from_request_context(
    request_payload: Mapping[str, Any],
    response_payload: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    if not _is_mcp_payload(response_payload):
        return None

    method = _first_string(response_payload.get("method"), request_payload.get("method"))
    params = response_payload.get("params")
    if not isinstance(params, Mapping):
        params = request_payload.get("params")
    if not method or not isinstance(params, Mapping):
        return None

    payload = dict(response_payload)
    payload.setdefault("jsonrpc", "2.0")
    if "id" not in payload and "id" in request_payload:
        payload["id"] = request_payload.get("id")
    payload["method"] = method
    payload["params"] = dict(params)
    return payload


def _mcp_request_context_payload(payload: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    method = _string_value(payload.get("method"))
    params = payload.get("params")
    if not method or not isinstance(params, Mapping):
        return None

    context = {
        "jsonrpc": "2.0",
        "method": method,
        "params": copy.deepcopy(dict(params)),
    }
    if "id" in payload:
        context["id"] = payload.get("id")
    return context


def _mcp_request_context_matches(
    context: Mapping[str, Any],
    *,
    scope_hash: str,
    response_id: Optional[str],
    tool_name: Optional[str],
    server_name: Optional[str],
) -> bool:
    if context.get("scope_hash") != scope_hash:
        return False

    context_id = context.get("id")
    if response_id is not None and context_id is not None and str(context_id) == response_id:
        return True

    params = context.get("params")
    if not isinstance(params, Mapping):
        return False

    context_tool = _string_value(params.get("name"))
    if tool_name is None or context_tool is None:
        return False

    # Exact match is always safe.
    if context_tool == tool_name:
        return True

    # Cross-form match (prefixed vs. short) is only safe when we have a known
    # server name to bound the prefix split. Without that, we'd risk matching
    # ``get-jira-issue`` against ``get-bitbucket-pr`` because both naively
    # produce server="get".
    if server_name is None:
        return False

    _, context_short_tool = _split_prefixed_mcp_tool(context_tool, server_name=server_name)
    _, response_short_tool = _split_prefixed_mcp_tool(tool_name, server_name=server_name)

    if context_short_tool is not None and context_short_tool == tool_name:
        return True
    if response_short_tool is not None and response_short_tool == context_tool:
        return True
    if (
        context_short_tool is not None
        and response_short_tool is not None
        and context_short_tool == response_short_tool
    ):
        return True
    return False


def _apply_mcp_response_payload_mutation(target: Dict[str, Any], decision: AIDefenseDecision) -> bool:
    mutation = _mcp_payload_mutation(decision, direction="output")
    if mutation is None:
        return False

    response_keys = {"jsonrpc", "id", "result", "error"}
    response_update = {key: value for key, value in mutation.items() if key in response_keys}
    if not response_update:
        return False

    # Preserve fields that the mutation did not explicitly replace (notably "id"),
    # but always drop the "result" / "error" pair so a sanitized "result" cannot
    # be mixed with a stale "error" (or vice versa).
    preserved_id = target.get("id")
    target.pop("result", None)
    target.pop("error", None)
    target.update(response_update)
    if "id" not in response_update and preserved_id is not None:
        target["id"] = preserved_id
    return True


def _mcp_tool_response_payload(
    kwargs: Mapping[str, Any],
    response_obj: Any,
    *,
    request_context: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    params = _mcp_tool_call_params_from_kwargs(kwargs)
    request_params = request_context.get("params") if request_context is not None else None
    server_name = _mcp_server_name_from_sources(data={}, kwargs=kwargs)
    if params is None:
        if isinstance(request_params, Mapping):
            params = dict(request_params)
    elif isinstance(request_params, Mapping):
        params = _merge_mcp_tool_response_params(params, request_params, server_name=server_name)
    if params is None:
        return None

    tool_response = getattr(response_obj, "mcp_tool_call_response", response_obj)
    content = getattr(tool_response, "content", tool_response)
    if not isinstance(content, list):
        content = [content]

    serialized_content = []
    for item in content:
        serialized_content.append(_serialize_mcp_content(item))

    return {
        "jsonrpc": "2.0",
        "id": _mcp_response_id(response_obj)
        or (str(request_context.get("id")) if request_context is not None and request_context.get("id") is not None else None)
        or "litellm-mcp-tool-response",
        "method": _string_value(request_context.get("method")) if request_context is not None else "tools/call",
        "params": params,
        "result": {"content": serialized_content},
    }


def _merge_mcp_tool_response_params(
    params: Dict[str, Any],
    request_params: Mapping[str, Any],
    *,
    server_name: Optional[str] = None,
) -> Dict[str, Any]:
    request_tool = _string_value(request_params.get("name"))
    params_tool = _string_value(params.get("name"))
    if not _mcp_tool_names_match(request_tool, params_tool, server_name=server_name):
        return params

    request_arguments = request_params.get("arguments")
    if isinstance(request_arguments, Mapping) and not params.get("arguments"):
        merged_params = dict(params)
        merged_params["arguments"] = copy.deepcopy(dict(request_arguments))
        return merged_params
    return params


def _mcp_tool_names_match(
    left: Optional[str],
    right: Optional[str],
    *,
    server_name: Optional[str] = None,
) -> bool:
    if not left or not right:
        return False
    if left == right:
        return True
    # Prefix-stripped equality is only safe when we know which server the tools
    # belong to — otherwise tools that happen to share a leading hyphen
    # component would falsely match.
    if server_name is None:
        return False
    _, left_short = _split_prefixed_mcp_tool(left, server_name=server_name)
    _, right_short = _split_prefixed_mcp_tool(right, server_name=server_name)
    if left_short is not None and left_short == right:
        return True
    if right_short is not None and right_short == left:
        return True
    if left_short is not None and right_short is not None and left_short == right_short:
        return True
    return False


def _mcp_tool_call_params_from_kwargs(kwargs: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    raw_params = kwargs.get("params")
    params = dict(raw_params) if isinstance(raw_params, Mapping) else {}

    tool_name = _first_string(
        params.get("name"),
        kwargs.get("name"),
        kwargs.get("mcp_tool_name"),
        kwargs.get("tool_name"),
    )
    if not tool_name:
        return None

    params["name"] = tool_name
    if "arguments" not in params:
        arguments = _first_mapping(
            kwargs.get("arguments"),
            kwargs.get("mcp_arguments"),
            kwargs.get("tool_arguments"),
            kwargs.get("input"),
            kwargs.get("tool_input"),
        )
        params["arguments"] = dict(arguments) if arguments is not None else {}
    return params


def _first_mapping(*values: Any) -> Optional[Mapping[str, Any]]:
    for value in values:
        if isinstance(value, Mapping):
            return value
    return None


def _mcp_response_id(response_obj: Any) -> Optional[str]:
    for source in (response_obj, getattr(response_obj, "mcp_tool_call_response", None)):
        if isinstance(source, Mapping):
            value = source.get("id")
        else:
            value = getattr(source, "id", None)
        if value is not None:
            return str(value)
    return None


def _json_size(value: Mapping[str, Any]) -> int:
    try:
        return len(json.dumps(value, default=str, separators=(",", ":")).encode("utf-8"))
    except Exception:
        return len(str(value).encode("utf-8", errors="replace"))


def _serialize_mcp_content(item: Any) -> Dict[str, Any]:
    if isinstance(item, dict):
        return dict(item)
    model_dump = getattr(item, "model_dump", None)
    if callable(model_dump):
        return dict(model_dump(exclude_none=True))
    text = getattr(item, "text", None)
    item_type = getattr(item, "type", "text")
    if isinstance(text, str):
        return {"type": item_type, "text": text}
    return {"type": "text", "text": str(item)}


def _set_mcp_tool_response_text(response_obj: Any, text: str, *, is_error: bool = False) -> None:
    try:
        from mcp.types import TextContent

        content = [TextContent(type="text", text=text)]
    except Exception:
        content = [{"type": "text", "text": text}]

    tool_response = getattr(response_obj, "mcp_tool_call_response", response_obj)
    if hasattr(tool_response, "content"):
        tool_response.content = content
        if hasattr(tool_response, "isError"):
            tool_response.isError = is_error
        return

    if hasattr(response_obj, "mcp_tool_call_response"):
        response_obj.mcp_tool_call_response = content


def _apply_mcp_tool_response_mutation(response_obj: Any, decision: AIDefenseDecision) -> bool:
    mutation = _mcp_payload_mutation(decision, direction="output")
    if mutation is None:
        return False

    result = mutation.get("result")
    if isinstance(result, Mapping):
        content = result.get("content")
        if isinstance(content, list):
            text_parts = [
                str(part.get("text"))
                for part in content
                if isinstance(part, Mapping) and isinstance(part.get("text"), str)
            ]
            if text_parts:
                _set_mcp_tool_response_text(response_obj, "\n".join(text_parts))
                return True

    return False


def _request_text_items(data: Dict[str, Any]) -> Iterator[TextItem]:
    messages = data.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict):
                yield from _content_text_items(message, "content")
            else:
                content = getattr(message, "content", None)
                if isinstance(content, str):
                    yield TextItem(content, lambda new_value, message=message: setattr(message, "content", new_value))

    mcp_arguments = data.get("mcp_arguments")
    if isinstance(mcp_arguments, (dict, list)):
        yield from _json_text_items(mcp_arguments)

    params = data.get("params")
    if isinstance(params, dict):
        arguments = params.get("arguments")
        if isinstance(arguments, (dict, list)):
            yield from _json_text_items(arguments)

    if "input" in data:
        yield from _value_text_items(data, "input")

    if "prompt" in data:
        yield from _value_text_items(data, "prompt")


def _content_text_items(container: Dict[str, Any], key: str) -> Iterator[TextItem]:
    value = container.get(key)
    if isinstance(value, str):
        yield TextItem(value, lambda new_value, container=container, key=key: container.__setitem__(key, new_value))
        return

    if isinstance(value, list):
        for part in value:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                yield TextItem(part["text"], lambda new_value, part=part: part.__setitem__("text", new_value))


def _value_text_items(container: Dict[str, Any], key: str) -> Iterator[TextItem]:
    value = container.get(key)
    if isinstance(value, str):
        yield TextItem(value, lambda new_value, container=container, key=key: container.__setitem__(key, new_value))
    elif isinstance(value, list):
        for index, entry in enumerate(value):
            if isinstance(entry, str):
                yield TextItem(entry, lambda new_value, value=value, index=index: value.__setitem__(index, new_value))
            elif isinstance(entry, dict):
                yield from _content_text_items(entry, "content")
                yield from _content_text_items(entry, "text")


def _json_text_items(value: Any) -> Iterator[TextItem]:
    if isinstance(value, dict):
        for key, entry in value.items():
            if isinstance(entry, str):
                yield TextItem(entry, lambda new_value, value=value, key=key: value.__setitem__(key, new_value))
            elif isinstance(entry, (dict, list)):
                yield from _json_text_items(entry)
    elif isinstance(value, list):
        for index, entry in enumerate(value):
            if isinstance(entry, str):
                yield TextItem(entry, lambda new_value, value=value, index=index: value.__setitem__(index, new_value))
            elif isinstance(entry, (dict, list)):
                yield from _json_text_items(entry)


def _response_text_items(response: Any) -> Iterator[TextItem]:
    if isinstance(response, dict):
        yield from _dict_response_text_items(response)
        return

    choices = getattr(response, "choices", None)
    if isinstance(choices, list):
        for choice in choices:
            message = getattr(choice, "message", None)
            content = getattr(message, "content", None) if message is not None else None
            if isinstance(content, str):
                yield TextItem(content, lambda new_value, message=message: setattr(message, "content", new_value))

    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str):
        yield TextItem(output_text, lambda new_value, response=response: setattr(response, "output_text", new_value))


def _dict_response_text_items(response: Dict[str, Any]) -> Iterator[TextItem]:
    choices = response.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if isinstance(choice, dict):
                message = choice.get("message")
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    yield TextItem(message["content"], lambda new_value, message=message: message.__setitem__("content", new_value))

    if isinstance(response.get("output_text"), str):
        yield TextItem(response["output_text"], lambda new_value, response=response: response.__setitem__("output_text", new_value))

    result = response.get("result")
    if isinstance(result, dict):
        yield from _content_text_items(result, "content")
        structured_content = result.get("structuredContent")
        if isinstance(structured_content, (dict, list)):
            yield from _json_text_items(structured_content)

    output = response.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        yield TextItem(part["text"], lambda new_value, part=part: part.__setitem__("text", new_value))


proxy_handler_instance = AIDefenseLiteLLMHook()
