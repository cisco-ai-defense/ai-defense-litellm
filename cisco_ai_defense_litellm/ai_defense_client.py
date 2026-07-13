# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0from __future__ import annotations

from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Mapping, Optional, Protocol


class AIDefenseDecision:
    def __init__(
            self,
            allowed: bool,
            action: str = "allow",
            reason: str = "No policy violations detected",
            score: float = 0.0,
            categories: tuple[str, ...] = (),
            sanitized_text: Optional[str] = None,
            error: bool = False,
            raw: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.allowed = allowed
        self.action = action
        self.reason = reason
        self.score = score
        self.categories = categories
        self.sanitized_text = sanitized_text
        self.error = error
        self.raw = raw or {}

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "action": self.action,
            "reason": self.reason,
            "score": self.score,
            "categories": list(self.categories),
            "error": self.error,
        }


class AIDefenseClient(Protocol):
    async def inspect(
            self,
            text: str,
            *,
            direction: str,
            metadata: Optional[Mapping[str, Any]] = None,
    ) -> AIDefenseDecision:
        ...


class UnconfiguredAIDefenseClient:
    """Error client used when no AI Defense Inspect endpoint is configured."""

    async def inspect(
            self,
            text: str,
            *,
            direction: str,
            metadata: Optional[Mapping[str, Any]] = None,
    ) -> AIDefenseDecision:
        del text, direction, metadata
        return AIDefenseDecision(
            allowed=False,
            action="error",
            reason="AI Defense Inspect endpoint not configured",
            categories=("ai_defense_unavailable",),
            error=True,
        )


class CiscoMCPAIDefenseClient:
    """HTTP adapter for Cisco AI Defense's MCP inspection endpoint."""

    def __init__(
            self,
            inspect_url: str,
            *,
            api_key: str,
            api_key_header: str = "X-Cisco-AI-Defense-API-Key",
            timeout_seconds: float = 2.0,
    ):
        self.inspect_url = inspect_url
        self.api_key = api_key
        self.api_key_header = api_key_header
        self.timeout_seconds = timeout_seconds

    async def inspect(
            self,
            text: str,
            *,
            direction: str,
            metadata: Optional[Mapping[str, Any]] = None,
    ) -> AIDefenseDecision:
        return await self.inspect_mcp(
            _text_to_mcp_payload(text, direction=direction),
            direction=direction,
            metadata=metadata,
        )

    async def inspect_mcp(
            self,
            payload: Mapping[str, Any],
            *,
            direction: str,
            metadata: Optional[Mapping[str, Any]] = None,
    ) -> AIDefenseDecision:
        del metadata
        try:
            return await asyncio.to_thread(self._inspect_mcp_sync, payload)
        except Exception as exc:
            return AIDefenseDecision(
                allowed=False,
                action="error",
                reason=f"AI Defense unavailable: {exc}",
                score=0.0,
                categories=("ai_defense_unavailable",),
                error=True,
            )

    def _inspect_mcp_sync(self, payload: Mapping[str, Any]) -> AIDefenseDecision:
        request_body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            self.api_key_header: self.api_key,
        }
        request = urllib.request.Request(
            self.inspect_url,
            data=request_body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            try:
                return decision_from_mapping(json.loads(body))
            except Exception:
                pass
            raise RuntimeError(f"HTTP {exc.code}: {body}") from exc

        return decision_from_mapping(json.loads(body))


class CiscoChatAIDefenseClient:
    """HTTP adapter for Cisco AI Defense's chat inspection endpoint."""

    def __init__(
            self,
            inspect_url: str,
            *,
            api_key: str,
            api_key_header: str = "X-Cisco-AI-Defense-API-Key",
            timeout_seconds: float = 2.0,
    ):
        self.inspect_url = inspect_url
        self.api_key = api_key
        self.api_key_header = api_key_header
        self.timeout_seconds = timeout_seconds

    async def inspect(
            self,
            text: str,
            *,
            direction: str,
            metadata: Optional[Mapping[str, Any]] = None,
    ) -> AIDefenseDecision:
        role = "assistant" if direction == "output" else "user"
        return await self.inspect_chat(
            {"messages": [{"role": role, "content": text}]},
            direction=direction,
            metadata=metadata,
        )

    async def inspect_chat(
            self,
            payload: Mapping[str, Any],
            *,
            direction: str,
            metadata: Optional[Mapping[str, Any]] = None,
    ) -> AIDefenseDecision:
        try:
            return await asyncio.to_thread(self._inspect_chat_sync, payload)
        except Exception as exc:
            return AIDefenseDecision(
                allowed=False,
                action="error",
                reason=f"AI Defense unavailable: {exc}",
                score=0.0,
                categories=("ai_defense_unavailable",),
                error=True,
            )

    def _inspect_chat_sync(self, payload: Mapping[str, Any]) -> AIDefenseDecision:
        request_body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            self.api_key_header: self.api_key,
        }
        request = urllib.request.Request(
            self.inspect_url,
            data=request_body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            try:
                return decision_from_mapping(json.loads(body))
            except Exception:
                pass
            raise RuntimeError(f"HTTP {exc.code}: {body}") from exc

        return decision_from_mapping(json.loads(body))


class CiscoCompositeAIDefenseClient:
    """Cisco AI Defense adapter using chat and MCP inspection endpoints together."""

    def __init__(
            self,
            *,
            chat_client: Optional[CiscoChatAIDefenseClient] = None,
            mcp_client: Optional[CiscoMCPAIDefenseClient] = None,
    ):
        if chat_client is None and mcp_client is None:
            raise ValueError("At least one AI Defense client must be configured")
        self.chat_client = chat_client
        self.mcp_client = mcp_client

    async def inspect(
            self,
            text: str,
            *,
            direction: str,
            metadata: Optional[Mapping[str, Any]] = None,
    ) -> AIDefenseDecision:
        if self.chat_client is not None:
            return await self.chat_client.inspect(text, direction=direction, metadata=metadata)
        if self.mcp_client is not None:
            return await self.mcp_client.inspect(text, direction=direction, metadata=metadata)
        raise RuntimeError("No AI Defense inspection client configured")

    async def inspect_chat(
            self,
            payload: Mapping[str, Any],
            *,
            direction: str,
            metadata: Optional[Mapping[str, Any]] = None,
    ) -> AIDefenseDecision:
        if self.chat_client is None:
            return await self.inspect(_messages_to_text(payload.get("messages")), direction=direction, metadata=metadata)
        return await self.chat_client.inspect_chat(payload, direction=direction, metadata=metadata)

    async def inspect_mcp(
            self,
            payload: Mapping[str, Any],
            *,
            direction: str,
            metadata: Optional[Mapping[str, Any]] = None,
    ) -> AIDefenseDecision:
        if self.mcp_client is None:
            return await self.inspect(json.dumps(payload), direction=direction, metadata=metadata)
        return await self.mcp_client.inspect_mcp(payload, direction=direction, metadata=metadata)


def decision_from_mapping(payload: Mapping[str, Any]) -> AIDefenseDecision:
    jsonrpc_error = _jsonrpc_error_payload(payload)
    if jsonrpc_error is not None:
        reason = _jsonrpc_error_reason(jsonrpc_error)
        return AIDefenseDecision(
            allowed=False,
            action="error",
            reason=reason,
            score=0.0,
            categories=("ai_defense_error",),
            error=True,
            raw=dict(payload),
        )

    decision_payload = _decision_payload(payload)
    action = _decision_action(decision_payload)
    allowed = _decision_allowed(decision_payload, action)
    categories = _decision_categories(decision_payload)
    score = _decision_score(decision_payload)
    sanitized_text = decision_payload.get("sanitized_text") or decision_payload.get("sanitizedText")
    if isinstance(categories, str):
        categories = (categories,)
    return AIDefenseDecision(
        allowed=allowed,
        action=action,
        reason=str(
            decision_payload.get("reason")
            or decision_payload.get("message")
            or decision_payload.get("description")
            or "AI Defense decision received"
        ),
        score=score,
        categories=tuple(str(category) for category in categories),
        sanitized_text=sanitized_text if isinstance(sanitized_text, str) else None,
        error=bool(decision_payload.get("error") or decision_payload.get("is_error") or decision_payload.get("isError")),
        raw=dict(payload),
    )


def build_ai_defense_client_from_env() -> AIDefenseClient:
    chat_inspect_url = os.getenv("AI_DEFENSE_CHAT_INSPECT_URL")
    mcp_inspect_url = os.getenv("AI_DEFENSE_MCP_INSPECT_URL")
    if chat_inspect_url or mcp_inspect_url:
        timeout = float(os.getenv("AI_DEFENSE_TIMEOUT_SECONDS", "5.0"))
        default_api_key_header = os.getenv("AI_DEFENSE_API_KEY_HEADER", "X-Cisco-AI-Defense-API-Key")
        chat_api_key = os.getenv("AI_DEFENSE_CHAT_API_KEY", "")
        mcp_api_key = os.getenv("AI_DEFENSE_MCP_API_KEY", "")
        chat_client = (
            CiscoChatAIDefenseClient(
                chat_inspect_url,
                api_key=chat_api_key,
                api_key_header=default_api_key_header,
                timeout_seconds=timeout,
            )
            if chat_inspect_url
            else None
        )
        mcp_client = (
            CiscoMCPAIDefenseClient(
                mcp_inspect_url,
                api_key=mcp_api_key,
                api_key_header=default_api_key_header,
                timeout_seconds=timeout,
            )
            if mcp_inspect_url
            else None
        )
        if chat_client and mcp_client:
            return CiscoCompositeAIDefenseClient(chat_client=chat_client, mcp_client=mcp_client)
        return chat_client or mcp_client  # type: ignore[return-value]

    return UnconfiguredAIDefenseClient()


def _text_to_mcp_payload(text: str, *, direction: str) -> Dict[str, Any]:
    if direction == "output":
        return {
            "jsonrpc": "2.0",
            "id": "litellm-ai-defense-output",
            "result": {"content": [{"type": "text", "text": text}]},
        }

    return {
        "jsonrpc": "2.0",
        "id": "litellm-ai-defense-input",
        "method": "tools/call",
        "params": {
            "name": "litellm_text_input",
            "arguments": {"text": text},
        },
    }


def _messages_to_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    parts = []
    for message in messages:
        if isinstance(message, Mapping):
            content = message.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                parts.extend(str(part.get("text")) for part in content if isinstance(part, Mapping) and isinstance(part.get("text"), str))
    return "\n".join(parts)


def _decision_action(payload: Mapping[str, Any]) -> str:
    raw_action = payload.get("action")
    if raw_action is None:
        return "allow"

    action = str(raw_action).strip().lower()
    if action in {"deny", "denied", "blocked", "block", "reject", "rejected", "unsafe", "malicious"}:
        return "block"
    if action in {"sanitize", "sanitized", "redacted"}:
        return "redact"
    return action


def _decision_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if _has_decision_fields(payload):
        return payload

    for key in ("result", "data", "inspection", "ai_defense", "aiDefense"):
        value = payload.get(key)
        if isinstance(value, Mapping) and _has_decision_fields(value):
            return value

    result = payload.get("result")
    if isinstance(result, Mapping):
        for key in ("data", "inspection", "ai_defense", "aiDefense"):
            value = result.get(key)
            if isinstance(value, Mapping) and _has_decision_fields(value):
                return value

    return payload


def _has_decision_fields(payload: Mapping[str, Any]) -> bool:
    return any(
        key in payload
        for key in (
            "action",
            "score",
            "risk_score",
            "confidence",
            "categories",
            "violations",
            "threats",
            "policies",
            "reason",
            "sanitized_text",
            "sanitizedText",
            "sanitized_payload",
        )
    )


def _decision_allowed(payload: Mapping[str, Any], action: str) -> bool:
    del payload
    return action not in {"block", "error"}


def _decision_categories(payload: Mapping[str, Any]) -> Any:
    return payload.get("categories") or payload.get("violations") or payload.get("threats") or payload.get("policies") or ()


def _decision_score(payload: Mapping[str, Any]) -> float:
    raw_score = payload.get("score")
    if raw_score is None:
        raw_score = payload.get("risk_score")
    if raw_score is None:
        raw_score = payload.get("confidence")
    try:
        return float(raw_score or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _jsonrpc_error_payload(payload: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    error = payload.get("error")
    if isinstance(error, Mapping):
        return error
    result = payload.get("result")
    if isinstance(result, Mapping):
        error = result.get("error")
        if isinstance(error, Mapping):
            return error
    return None


def _jsonrpc_error_reason(error: Mapping[str, Any]) -> str:
    message = error.get("message")
    if isinstance(message, str) and message:
        return f"AI Defense returned an error: {message}"
    code = error.get("code")
    if code is not None:
        return f"AI Defense returned an error: {code}"
    return "AI Defense returned an error"