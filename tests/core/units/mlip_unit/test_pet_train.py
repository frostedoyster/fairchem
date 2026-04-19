"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from tests.core.testing_utils import launch_main


def test_pet_train_from_cli(fake_uma_dataset, torch_deterministic):
    launch_main(
        [
            "--config",
            "tests/core/units/mlip_unit/test_mlip_train_pet.yaml",
            f"datasets.data_root_dir={fake_uma_dataset}",
        ]
    )
