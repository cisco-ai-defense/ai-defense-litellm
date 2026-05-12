from __future__ import annotations

from types import ModuleType
from typing import Optional

from .litellm_hook import AIDefenseLiteLLMHook

_INSTALLED_HOOK: Optional[AIDefenseLiteLLMHook] = None


def create_hook(**kwargs: object) -> AIDefenseLiteLLMHook:
    return AIDefenseLiteLLMHook(**kwargs)


def install(*, litellm_module: Optional[ModuleType] = None, **kwargs: object) -> AIDefenseLiteLLMHook:
    global _INSTALLED_HOOK
    litellm = litellm_module or _import_litellm()

    if _INSTALLED_HOOK is not None and _hook_registered(litellm, _INSTALLED_HOOK):
        return _INSTALLED_HOOK

    hook = create_hook(**kwargs)
    callback_manager = getattr(litellm, "logging_callback_manager", None)
    if callback_manager is None or not hasattr(callback_manager, "add_litellm_callback"):
        callbacks = getattr(litellm, "callbacks", None)
        if not isinstance(callbacks, list):
            raise RuntimeError("LiteLLM SDK callback registry is not available")
        callbacks.append(hook)
    else:
        callback_manager.add_litellm_callback(hook)

    _INSTALLED_HOOK = hook
    return hook


def uninstall(*, litellm_module: Optional[ModuleType] = None) -> bool:
    global _INSTALLED_HOOK
    litellm = litellm_module or _import_litellm()
    hook = _INSTALLED_HOOK
    if hook is None:
        return False

    removed = False
    for list_name in (
        "callbacks",
        "input_callback",
        "success_callback",
        "failure_callback",
        "_async_input_callback",
        "_async_success_callback",
        "_async_failure_callback",
    ):
        callback_list = getattr(litellm, list_name, None)
        if not isinstance(callback_list, list):
            continue
        while hook in callback_list:
            callback_list.remove(hook)
            removed = True

    _INSTALLED_HOOK = None
    return removed


def is_installed(*, litellm_module: Optional[ModuleType] = None) -> bool:
    litellm = litellm_module or _import_litellm()
    hook = _INSTALLED_HOOK
    return hook is not None and _hook_registered(litellm, hook)


def _hook_registered(litellm: ModuleType, hook: AIDefenseLiteLLMHook) -> bool:
    callbacks = getattr(litellm, "callbacks", None)
    return isinstance(callbacks, list) and hook in callbacks


def _import_litellm() -> ModuleType:
    import litellm  # type: ignore

    return litellm
