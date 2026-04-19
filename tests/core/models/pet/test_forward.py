"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import torch

from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.datasets.common_structures import get_fcc_crystal_by_num_atoms
from fairchem.core.models.base import HydraModelV2
from fairchem.core.models.pet.PET import (
    PETBackbone,
    PETDirectForceHead,
    PETEnergyHead,
)


def test_pet_dummy_forward_and_backward_cpu():
    data = AtomicData.from_ase(get_fcc_crystal_by_num_atoms(4))
    data["batch"] = torch.zeros(data["pos"].shape[0], dtype=torch.long)
    data["natoms"] = torch.tensor([data["pos"].shape[0]])

    backbone = PETBackbone()
    energy_head = PETEnergyHead(backbone)
    force_head = PETDirectForceHead(backbone)
    model = HydraModelV2(
        backbone,
        {
            "energy": energy_head,
            "forces": force_head,
        },
    )

    out = model(data)

    expected_energy = data["pos"].sum().reshape(1)
    assert torch.allclose(out["energy"]["energy"], expected_energy)
    assert out["forces"]["forces"].shape == data["pos"].shape

    loss = out["energy"]["energy"].sum() + out["forces"]["forces"].sum()
    loss.backward()

    assert energy_head.energy_scale.grad is not None
    assert force_head.force_scale.grad is not None
