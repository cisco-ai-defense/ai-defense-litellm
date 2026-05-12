import unittest
from types import SimpleNamespace

from cisco_ai_defense_litellm.sdk import install, is_installed, uninstall


class FakeCallbackManager:
    def __init__(self, module):
        self.module = module

    def add_litellm_callback(self, callback):
        self.module.callbacks.append(callback)


class LiteLLMSDKInstallTests(unittest.TestCase):
    def setUp(self):
        self.fake_litellm = SimpleNamespace(
            callbacks=[],
            input_callback=[],
            success_callback=[],
            failure_callback=[],
            _async_input_callback=[],
            _async_success_callback=[],
            _async_failure_callback=[],
        )
        self.fake_litellm.logging_callback_manager = FakeCallbackManager(self.fake_litellm)
        uninstall(litellm_module=self.fake_litellm)

    def tearDown(self):
        uninstall(litellm_module=self.fake_litellm)

    def test_install_registers_single_hook(self):
        first = install(litellm_module=self.fake_litellm)
        second = install(litellm_module=self.fake_litellm)

        self.assertIs(first, second)
        self.assertEqual(len(self.fake_litellm.callbacks), 1)
        self.assertTrue(is_installed(litellm_module=self.fake_litellm))

    def test_uninstall_removes_registered_hook(self):
        install(litellm_module=self.fake_litellm)

        removed = uninstall(litellm_module=self.fake_litellm)

        self.assertTrue(removed)
        self.assertFalse(self.fake_litellm.callbacks)
        self.assertFalse(is_installed(litellm_module=self.fake_litellm))


if __name__ == "__main__":
    unittest.main()
