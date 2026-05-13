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

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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
