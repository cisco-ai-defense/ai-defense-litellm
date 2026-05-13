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
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Protocol, Tuple


DEFAULT_BLOCK_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("prompt_injection", r"\b(ignore|bypass|override)\b.{0,80}\b(previous|prior|system|developer)\b.{0,80}\b(instruction|message|prompt|policy)s?\b"),
    ("secret_exfiltration", r"\b(exfiltrate|leak|dump|print|reveal|steal)\b.{0,80}\b(secret|token|credential|password|api[_ -]?key)s?\b"),
    ("tool_abuse", r"\b(call|use|invoke)\b.{0,80}\btool\b.{0,80}\b(without|bypass|ignore)\b.{0,80}\bapproval\b"),
)

SECRET_VALUE_PATTERN = re.compile(
    r"(?P<name>\b(?:api[_-]?key|secret|password|token|bearer)\b\s*[:=]\s*)(?P<value>[A-Za-z0-9_\-./+=]{6,})",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AIDefenseDecision:
    allowed: bool
    action: str = "allow"
    reason: str = "No policy violations detected"
    score: float = 0.0
    categories: Tuple[str, ...] = ()
    sanitized_text: Optional[str] = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "action": self.action,
            "reason": self.reason,
            "score": self.score,
            "categories": list(self.categories),
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


class RuleBasedAIDefenseClient:
    """Local deterministic client used by unit tests and offline smoke checks."""

    def __init__(self, block_patterns: Iterable[Tuple[str, str]] = DEFAULT_BLOCK_PATTERNS):
        self._block_patterns = tuple((category, re.compile(pattern, re.IGNORECASE | re.DOTALL)) for category, pattern in block_patterns)

    async def inspect(
        self,
        text: str,
        *,
        direction: str,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> AIDefenseDecision:
        del direction, metadata
        if not text:
            return AIDefenseDecision(allowed=True)

        matched_categories = [category for category, pattern in self._block_patterns if pattern.search(text)]
        if matched_categories:
            return AIDefenseDecision(
                allowed=False,
                action="block",
                reason="Potential prompt-injection or secret-exfiltration request detected",
                score=0.95,
                categories=tuple(matched_categories),
            )

        sanitized = self.redact(text)
        if sanitized != text:
            return AIDefenseDecision(
                allowed=True,
                action="redact",
                reason="Sensitive value redacted before forwarding",
                score=0.65,
                categories=("secret",),
                sanitized_text=sanitized,
            )

        return AIDefenseDecision(allowed=True)

    @staticmethod
    def redact(text: str) -> str:
        return SECRET_VALUE_PATTERN.sub(lambda match: f"{match.group('name')}[REDACTED]", text)


class CiscoMCPAIDefenseClient:
    """HTTP adapter for Cisco AI Defense's MCP inspection endpoint."""

    def __init__(
        self,
        inspect_url: str,
        *,
        api_key: str,
        api_key_header: str = "X-Cisco-AI-Defense-API-Key",
        timeout_seconds: float = 2.0,
        fail_open: bool = False,
    ):
        self.inspect_url = inspect_url
        self.api_key = api_key
        self.api_key_header = api_key_header
        self.timeout_seconds = timeout_seconds
        self.fail_open = fail_open

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
            if self.fail_open:
                return AIDefenseDecision(
                    allowed=True,
                    action="allow",
                    reason=f"AI Defense unavailable, fail-open enabled: {exc}",
                    score=0.0,
                    categories=("ai_defense_unavailable",),
                )
            return AIDefenseDecision(
                allowed=False,
                action="block",
                reason=f"AI Defense unavailable, fail-closed: {exc}",
                score=1.0,
                categories=("ai_defense_unavailable",),
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
        fail_open: bool = False,
    ):
        self.inspect_url = inspect_url
        self.api_key = api_key
        self.api_key_header = api_key_header
        self.timeout_seconds = timeout_seconds
        self.fail_open = fail_open

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
        del direction, metadata
        try:
            return await asyncio.to_thread(self._inspect_chat_sync, payload)
        except Exception as exc:
            if self.fail_open:
                return AIDefenseDecision(
                    allowed=True,
                    action="allow",
                    reason=f"AI Defense unavailable, fail-open enabled: {exc}",
                    score=0.0,
                    categories=("ai_defense_unavailable",),
                )
            return AIDefenseDecision(
                allowed=False,
                action="block",
                reason=f"AI Defense unavailable, fail-closed: {exc}",
                score=1.0,
                categories=("ai_defense_unavailable",),
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
        raw=dict(payload),
    )


def build_ai_defense_client_from_env() -> AIDefenseClient:
    chat_inspect_url = os.getenv("AI_DEFENSE_CHAT_INSPECT_URL")
    mcp_inspect_url = os.getenv("AI_DEFENSE_MCP_INSPECT_URL")
    if chat_inspect_url or mcp_inspect_url:
        timeout = float(os.getenv("AI_DEFENSE_TIMEOUT_SECONDS", "2.0"))
        fail_open = os.getenv("AI_DEFENSE_FAIL_OPEN", "false").lower() in {"1", "true", "yes"}
        default_api_key_header = os.getenv("AI_DEFENSE_API_KEY_HEADER", "X-Cisco-AI-Defense-API-Key")
        chat_api_key = os.getenv("AI_DEFENSE_CHAT_API_KEY", "")
        mcp_api_key = os.getenv("AI_DEFENSE_MCP_API_KEY", "")
        chat_client = (
            CiscoChatAIDefenseClient(
                chat_inspect_url,
                api_key=chat_api_key,
                api_key_header=default_api_key_header,
                timeout_seconds=timeout,
                fail_open=fail_open,
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
                fail_open=fail_open,
            )
            if mcp_inspect_url
            else None
        )
        if chat_client and mcp_client:
            return CiscoCompositeAIDefenseClient(chat_client=chat_client, mcp_client=mcp_client)
        return chat_client or mcp_client  # type: ignore[return-value]

    return RuleBasedAIDefenseClient()


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
    raw_action = payload.get("action") or payload.get("decision") or payload.get("verdict") or payload.get("status")
    if raw_action is None:
        if payload.get("blocked"):
            return "block"
        if payload.get("allowed") is False or payload.get("safe") is False or payload.get("is_safe") is False:
            return "block"
        if payload.get("sanitized_text") or payload.get("sanitizedText") or payload.get("sanitized_payload"):
            return "redact"
        return "allow"

    action = str(raw_action).lower()
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
            "allowed",
            "blocked",
            "safe",
            "is_safe",
            "decision",
            "verdict",
            "status",
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
    if "allowed" in payload:
        return bool(payload["allowed"])
    if "blocked" in payload:
        return not bool(payload["blocked"])
    if "safe" in payload:
        return bool(payload["safe"])
    if "is_safe" in payload:
        return bool(payload["is_safe"])
    return action != "block"


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
