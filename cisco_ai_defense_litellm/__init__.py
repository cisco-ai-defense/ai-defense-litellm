"""Cisco AI Defense integration for LiteLLM proxy and SDK deployments."""

from .litellm_hook import AIDefenseLiteLLMHook, proxy_handler_instance
from .sdk import create_hook, install, is_installed, uninstall

__all__ = [
    "__version__",
    "AIDefenseLiteLLMHook",
    "proxy_handler_instance",
    "create_hook",
    "install",
    "is_installed",
    "uninstall",
]

__version__ = "0.2.0"
