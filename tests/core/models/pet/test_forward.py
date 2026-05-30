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
    for num_neighbors_adaptive in (None, 8.0):
        data = AtomicData.from_ase(
            get_fcc_crystal_by_num_atoms(4),
            r_edges=True,
            radius=6.0,
            max_neigh=100,
        )
        data["batch"] = torch.zeros(data["pos"].shape[0], dtype=torch.long)
        data["natoms"] = torch.tensor([data["pos"].shape[0]])

        backbone = PETBackbone(
            cutoff=6.0,
            cutoff_width=0.5,
            num_neighbors_adaptive=num_neighbors_adaptive,
            d_pet=32,
            node_to_edge_ratio=2,
            num_gnn_layers=1,
            num_attention_layers=1,
            num_heads=4,
            regress_forces=False,
            regress_stress=False,
        )
        model = HydraModelV2(
            backbone,
            {
                "energy": PETEnergyHead(backbone),
                "forces": PETDirectForceHead(backbone),
            },
        )

        out = model(data)

        assert out["energy"]["energy"].shape == torch.Size([1])
        assert out["forces"]["forces"].shape == data["pos"].shape

        loss = out["energy"]["energy"].sum() + out["forces"]["forces"].sum()
        loss.backward()

        assert any(p.grad is not None for p in model.parameters())
