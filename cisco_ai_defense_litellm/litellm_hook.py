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

import logging
import os
import sys
from dataclasses import dataclass
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


BLOCKED_MESSAGE_PREFIX = "Blocked by AI Defense"
LOGGER = logging.getLogger("cisco_ai_defense_litellm.ai_defense_hook")
LOGGER.setLevel(getattr(logging, os.getenv("AI_DEFENSE_LOG_LEVEL", "INFO").upper(), logging.INFO))
BaseLiteLLMHook = CustomGuardrail or CustomLogger


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


@dataclass
class TextItem:
    text: str
    set_text: Callable[[str], None]


async def async_post_call_response_headers_hook(
        data: Dict[str, Any],
    user_api_key_dict: Any,
    response: Any,
    request_headers: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, str]]:
    del user_api_key_dict, response, request_headers
    headers = _response_headers_for_data(data)
    if headers:
        LOGGER.info("AI Defense response headers hook added headers: names=%s", ",".join(sorted(headers)))
    else:
        LOGGER.info("AI Defense response headers hook reached: no AI Defense decisions recorded")
    return headers or None


class AIDefenseLiteLLMHook(BaseLiteLLMHook):  # type: ignore[misc, valid-type]
    """LiteLLM proxy and SDK hook for AI Defense style request/response inspection."""

    def __init__(
        self,
        client: Optional[AIDefenseClient] = None,
        *,
        block_mode: Optional[str] = None,
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
        self.block_mode = (block_mode or os.getenv("AI_DEFENSE_BLOCK_MODE", "message")).lower()
        self.scan_outputs = _env_bool("AI_DEFENSE_SCAN_OUTPUTS", default=True) if scan_outputs is None else scan_outputs
        LOGGER.info(
            "AI Defense LiteLLM hook initialized: client=%s block_mode=%s scan_outputs=%s",
            type(self.client).__name__,
            self.block_mode,
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
        LOGGER.info(
            "AI Defense request hook reached: call_type=%s model=%s method=%s",
            call_type,
            data.get("model"),
            data.get("method"),
        )

        mcp_decision = await _inspect_mcp_payload_if_supported(self.client, data, direction="input", metadata=request_metadata)
        if mcp_decision is not None:
            _record_decision(data, "input", mcp_decision)
            _log_decision("MCP request", 1, mcp_decision)

            if not mcp_decision.allowed:
                LOGGER.warning(
                    "AI Defense blocked MCP request before upstream: reason=%s categories=%s",
                    mcp_decision.reason,
                    ",".join(mcp_decision.categories) or "none",
                )
                return self._blocked_response(mcp_decision)

            if _apply_mcp_payload_mutation(data, mcp_decision, direction="input"):
                LOGGER.info("AI Defense mutated MCP request before upstream")

            LOGGER.info(
                "AI Defense MCP request hook completed: action=%s model=%s method=%s",
                mcp_decision.action,
                data.get("model"),
                data.get("method"),
            )
            return data

        synthetic_mcp_payload = _synthetic_mcp_tool_call_payload(data)
        if synthetic_mcp_payload is not None:
            mcp_decision = await _inspect_mcp_payload_if_supported(
                self.client,
                synthetic_mcp_payload,
                direction="input",
                metadata=request_metadata,
            )
            if mcp_decision is not None:
                _record_decision(data, "input", mcp_decision)
                _log_decision("MCP tool request", 1, mcp_decision)

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
                    LOGGER.info(
                        "AI Defense mutated MCP tool request before upstream: tool=%s",
                        data.get("mcp_tool_name"),
                    )

                LOGGER.info(
                    "AI Defense MCP tool request hook completed: action=%s tool=%s",
                    mcp_decision.action,
                    data.get("mcp_tool_name"),
                )
                return data

        chat_payload = _chat_request_payload(data)
        chat_decision = await _inspect_chat_payload_if_supported(
            self.client,
            chat_payload,
            direction="input",
            metadata=request_metadata,
        )
        if chat_decision is not None:
            _record_decision(data, "input", chat_decision)
            _log_decision("chat request", 1, chat_decision)

            if not chat_decision.allowed:
                LOGGER.warning(
                    "AI Defense blocked chat request before upstream: reason=%s categories=%s",
                    chat_decision.reason,
                    ",".join(chat_decision.categories) or "none",
                )
                return self._blocked_response(chat_decision)

            if _apply_chat_request_mutation(data, chat_decision):
                LOGGER.info("AI Defense mutated chat request before upstream")

            LOGGER.info(
                "AI Defense chat request hook completed: action=%s model=%s",
                chat_decision.action,
                data.get("model"),
            )
            return data

        inspected_count = 0
        for inspected_count, item in enumerate(_request_text_items(data), start=1):
            _log_text_reached("request", inspected_count, item.text)
            decision = await self.client.inspect(item.text, direction="input", metadata=request_metadata)
            _record_decision(data, "input", decision)
            _log_decision("request", inspected_count, decision)

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
                LOGGER.info("AI Defense redacted request before upstream: index=%s", inspected_count)

        if inspected_count == 0:
            LOGGER.info(
                "AI Defense request hook completed: no inspectable request text found call_type=%s model=%s method=%s",
                call_type,
                data.get("model"),
                data.get("method"),
            )
        else:
            LOGGER.info(
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
            LOGGER.info(
                "AI Defense response hook skipped: scan_outputs=false model=%s method=%s",
                data.get("model"),
                data.get("method"),
            )
            return response

        response_metadata = _request_metadata(data, user_api_key_dict, "response")
        LOGGER.info(
            "AI Defense response hook reached: model=%s method=%s",
            data.get("model"),
            data.get("method"),
        )

        if isinstance(response, dict):
            mcp_decision = await _inspect_mcp_payload_if_supported(self.client, response, direction="output", metadata=response_metadata)
            if mcp_decision is not None:
                _record_decision(data, "output", mcp_decision)
                _log_decision("MCP response", 1, mcp_decision)

                if not mcp_decision.allowed:
                    _replace_mcp_response_with_blocked_message(response, mcp_decision)
                    LOGGER.warning(
                        "AI Defense blocked MCP response before returning to client: reason=%s categories=%s",
                        mcp_decision.reason,
                        ",".join(mcp_decision.categories) or "none",
                    )
                    return response

                if _apply_mcp_payload_mutation(response, mcp_decision, direction="output"):
                    LOGGER.info("AI Defense mutated MCP response before returning to client")

                LOGGER.info(
                    "AI Defense MCP response hook completed: action=%s model=%s method=%s",
                    mcp_decision.action,
                    data.get("model"),
                    data.get("method"),
                )
                return response

        chat_payload = _chat_response_payload(response)
        chat_decision = await _inspect_chat_payload_if_supported(
            self.client,
            chat_payload,
            direction="output",
            metadata=response_metadata,
        )
        if chat_decision is not None:
            _record_decision(data, "output", chat_decision)
            _log_decision("chat response", 1, chat_decision)

            if not chat_decision.allowed:
                _replace_response_text_with_blocked_message(response, chat_decision)
                LOGGER.warning(
                    "AI Defense blocked chat response before returning to client: reason=%s categories=%s",
                    chat_decision.reason,
                    ",".join(chat_decision.categories) or "none",
                )
                return response

            if _apply_chat_response_mutation(response, chat_decision):
                LOGGER.info("AI Defense mutated chat response before returning to client")

            LOGGER.info(
                "AI Defense chat response hook completed: action=%s model=%s",
                chat_decision.action,
                data.get("model"),
            )
            return response

        inspected_count = 0
        for inspected_count, item in enumerate(_response_text_items(response), start=1):
            _log_text_reached("response", inspected_count, item.text)
            decision = await self.client.inspect(item.text, direction="output", metadata=response_metadata)
            _record_decision(data, "output", decision)
            _log_decision("response", inspected_count, decision)

            if not decision.allowed:
                item.set_text(self._blocked_message(decision))
                LOGGER.warning(
                    "AI Defense blocked response before returning to client: index=%s reason=%s categories=%s",
                    inspected_count,
                    decision.reason,
                    ",".join(decision.categories) or "none",
                )
                break

            if decision.action == "redact" and decision.sanitized_text:
                item.set_text(decision.sanitized_text)
                LOGGER.info("AI Defense redacted response before returning to client: index=%s", inspected_count)

        if inspected_count == 0:
            LOGGER.info(
                "AI Defense response hook completed: no inspectable response text found model=%s method=%s",
                data.get("model"),
                data.get("method"),
            )
        else:
            LOGGER.info(
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
        LOGGER.info(
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
        LOGGER.info(
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
            LOGGER.info(
                "AI Defense MCP tool response hook skipped: scan_outputs=false tool=%s",
                kwargs.get("name"),
            )
            return None

        payload = _mcp_tool_response_payload(response_obj)
        response_metadata = {
            "model": "MCP: call_tool",
            "call_type": "call_mcp_tool",
            "tool": kwargs.get("name"),
            "server_name": kwargs.get("server_name"),
        }
        LOGGER.info(
            "AI Defense MCP tool response hook reached: tool=%s server_name=%s",
            kwargs.get("name"),
            kwargs.get("server_name"),
        )

        mcp_decision = await _inspect_mcp_payload_if_supported(
            self.client,
            payload,
            direction="output",
            metadata=response_metadata,
        )
        if mcp_decision is None:
            return None

        _log_decision("MCP tool response", 1, mcp_decision)
        if not mcp_decision.allowed:
            _set_mcp_tool_response_text(response_obj, self._blocked_message(mcp_decision), is_error=True)
            LOGGER.warning(
                "AI Defense blocked MCP tool response before returning to client: tool=%s reason=%s categories=%s",
                kwargs.get("name"),
                mcp_decision.reason,
                ",".join(mcp_decision.categories) or "none",
            )
            return response_obj

        if _apply_mcp_tool_response_mutation(response_obj, mcp_decision):
            LOGGER.info(
                "AI Defense mutated MCP tool response before returning to client: tool=%s",
                kwargs.get("name"),
            )
            return response_obj

        LOGGER.info(
            "AI Defense MCP tool response hook completed: action=%s tool=%s",
            mcp_decision.action,
            kwargs.get("name"),
        )
        return None

    def _blocked_response(self, decision: AIDefenseDecision) -> str:
        if self.block_mode == "exception":
            try:
                from fastapi import HTTPException
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(self._blocked_message(decision)) from exc
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "blocked_by_ai_defense",
                    "reason": decision.reason,
                    "categories": list(decision.categories),
                    "score": decision.score,
                },
            )
        return self._blocked_message(decision)

    @staticmethod
    def _blocked_message(decision: AIDefenseDecision) -> str:
        categories = f" Categories: {', '.join(decision.categories)}." if decision.categories else ""
        return f"{BLOCKED_MESSAGE_PREFIX}: {decision.reason}.{categories}"


def _env_bool(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _response_headers_for_data(data: Mapping[str, Any]) -> Dict[str, str]:
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


def _record_decision(data: Dict[str, Any], direction: str, decision: AIDefenseDecision) -> None:
    metadata = data.setdefault("metadata", {})
    ai_defense = metadata.setdefault("ai_defense", {})
    ai_defense[direction] = decision.to_metadata()


def _log_text_reached(stage: str, index: int, text: str) -> None:
    preview = _text_preview(text)
    if preview is None:
        LOGGER.info("AI Defense %s text reached hook: index=%s text_length=%s", stage, index, len(text))
        return
    LOGGER.info(
        "AI Defense %s text reached hook: index=%s text_length=%s preview=%r",
        stage,
        index,
        len(text),
        preview,
    )


def _log_decision(stage: str, index: int, decision: AIDefenseDecision) -> None:
    LOGGER.info(
        "AI Defense %s decision: index=%s action=%s allowed=%s score=%s categories=%s",
        stage,
        index,
        decision.action,
        decision.allowed,
        decision.score,
        ",".join(decision.categories) or "none",
    )


def _text_preview(text: str) -> Optional[str]:
    if not _env_bool("AI_DEFENSE_LOG_TEXT", default=False):
        return None
    compact = " ".join(text.split())
    if len(compact) > 160:
        return f"{compact[:157]}..."
    return compact


async def _inspect_mcp_payload_if_supported(
    client: AIDefenseClient,
    payload: Mapping[str, Any],
    *,
    direction: str,
    metadata: Mapping[str, Any],
) -> Optional[AIDefenseDecision]:
    inspect_mcp = getattr(client, "inspect_mcp", None)
    if not callable(inspect_mcp) or not _is_mcp_payload(payload):
        return None

    LOGGER.info(
        "AI Defense MCP %s payload reached hook: method=%s id=%s",
        "request" if direction == "input" else "response",
        payload.get("method"),
        payload.get("id"),
    )
    return await inspect_mcp(payload, direction=direction, metadata=metadata)


async def _inspect_chat_payload_if_supported(
    client: AIDefenseClient,
    payload: Optional[Mapping[str, Any]],
    *,
    direction: str,
    metadata: Mapping[str, Any],
) -> Optional[AIDefenseDecision]:
    inspect_chat = getattr(client, "inspect_chat", None)
    if not callable(inspect_chat) or payload is None:
        return None

    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return None

    LOGGER.info(
        "AI Defense chat %s payload reached hook: messages=%s",
        "request" if direction == "input" else "response",
        len(messages),
    )
    return await inspect_chat(payload, direction=direction, metadata=metadata)


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


def _replace_response_text_with_blocked_message(response: Any, decision: AIDefenseDecision) -> None:
    blocked_message = AIDefenseLiteLLMHook._blocked_message(decision)
    changed = False
    for item in _response_text_items(response):
        item.set_text(blocked_message)
        changed = True

    if not changed and isinstance(response, dict):
        response["output_text"] = blocked_message


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
        ]
    }


def _mcp_tool_response_payload(response_obj: Any) -> Dict[str, Any]:
    tool_response = getattr(response_obj, "mcp_tool_call_response", response_obj)
    content = getattr(tool_response, "content", tool_response)
    if not isinstance(content, list):
        content = [content]

    serialized_content = []
    for item in content:
        serialized_content.append(_serialize_mcp_content(item))

    return {
        "jsonrpc": "2.0",
        "id": "litellm-mcp-tool-response",
        "result": {"content": serialized_content},
    }


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
