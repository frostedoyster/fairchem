"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import math
import numpy as np
import torch
from ase import Atoms

if TYPE_CHECKING:
    from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.models.utils.irreps import cg_change_mat, irreps_sum


def _get_molecule_cell(data_object: AtomicData):
    # create an Atoms object and center molecule in cell
    mol = Atoms(
        numbers=data_object.atomic_numbers,
        positions=data_object.pos,
    )
    # largest radius cutoff is ~12A include that and a safety factor of 10
    mol.center(vacuum=(12.0 * 10.0))
    mol.pbc = [True, True, True]

    positions = np.array(mol.get_positions(), copy=True)
    # pbc = np.array(mol.pbc, copy=True)
    cell = np.array(mol.get_cell(complete=True), copy=True)

    atomic_numbers = torch.Tensor(mol.get_atomic_numbers())
    positions = torch.from_numpy(positions).float()
    cell = torch.from_numpy(cell).view(1, 3, 3).float()
    natoms = positions.shape[0]
    assert data_object.natoms == natoms

    return atomic_numbers, positions, cell


def common_transform(data_object: AtomicData, config) -> AtomicData:
    data_object.dataset = config["dataset_name"]

    if not hasattr(data_object, "charge"):
        data_object.charge = 0
    if not hasattr(data_object, "spin"):
        data_object.spin = 0
    ensure_tensor(data_object, "energy")
    return data_object


def ensure_tensor(data_object, keys):
    # ensure dataset_energy is a tensor
    if isinstance(keys, str):
        keys = [keys]
    for key in keys:
        if hasattr(data_object, key):
            if not torch.is_tensor(getattr(data_object, key)):
                setattr(
                    data_object,
                    key,
                    torch.tensor(getattr(data_object, key), dtype=torch.float),
                )
            setattr(data_object, key, getattr(data_object, key).view(-1).float())
    return data_object


def ani1x_transform(data_object: AtomicData, config) -> AtomicData:
    # make periodic with molecule centered in large cell
    atomic_numbers, positions, cell = _get_molecule_cell(data_object)
    data_object.atomic_numbers = atomic_numbers
    data_object.pos = positions
    data_object.cell = cell

    # add fixed
    data_object.fixed = torch.zeros(data_object.natoms, dtype=torch.float)

    # common transforms
    data_object = common_transform(data_object, config)

    # ensure ani1x_energy is a tensor
    return data_object


def trans1x_transform(data_object: AtomicData, config) -> AtomicData:
    # make periodic with molecule centered in large cell
    atomic_numbers, positions, cell = _get_molecule_cell(data_object)
    data_object.atomic_numbers = atomic_numbers
    data_object.pos = positions
    data_object.cell = cell

    # add fixed and cell
    data_object.fixed = torch.zeros(data_object.natoms, dtype=torch.float)

    # common transforms
    data_object = common_transform(data_object, config)

    # ensure trans1x_energy is a tensor

    return data_object


def spice_transform(data_object: AtomicData, config) -> AtomicData:
    # make periodic with molecule centered in large cell
    atomic_numbers, positions, cell = _get_molecule_cell(data_object)
    data_object.atomic_numbers = atomic_numbers
    data_object.pos = positions
    data_object.cell = cell

    # add fixed
    data_object.fixed = torch.zeros(data_object.natoms, dtype=torch.float)

    # common transforms
    data_object = common_transform(data_object, config)
    # this is necessary for SPICE maceoff split to work with GemNet-OC
    data_object.tags = torch.full(data_object.tags.shape, 2, dtype=torch.long)

    # ensure spice_energy is a tensor

    return data_object


def qmof_transform(data_object: AtomicData, config) -> AtomicData:
    # add fixed and cell
    data_object.fixed = torch.zeros(data_object.natoms, dtype=torch.float)

    # common transforms
    data_object = common_transform(data_object, config)

    return data_object


def qm9_transform(data_object: AtomicData, config) -> AtomicData:
    # make periodic with molecule centered in large cell
    atomic_numbers, positions, cell = _get_molecule_cell(data_object)
    data_object.atomic_numbers = atomic_numbers
    data_object.pos = positions
    data_object.cell = cell

    # add fixed
    data_object.fixed = torch.zeros(data_object.natoms, dtype=torch.float)

    # common transforms
    data_object = common_transform(data_object, config)

    return data_object


def omol_transform(data_object: AtomicData, config) -> AtomicData:
    # make periodic with molecule centered in large cell
    atomic_numbers, positions, cell = _get_molecule_cell(data_object)
    data_object.atomic_numbers = atomic_numbers
    data_object.pos = positions
    data_object.cell = cell
    assert hasattr(
        data_object, "charge"
    ), "no charge in omol dataset set a2g_args: {r_energy: True, r_forces: True, r_data_keys: ['spin', 'charge']}"
    assert hasattr(
        data_object, "spin"
    ), "no spin in omol dataset set a2g_args: {r_energy: True, r_forces: True, r_data_keys: ['spin', 'charge']}"

    # add fixed
    data_object.fixed = torch.zeros(data_object.natoms, dtype=torch.float)

    return common_transform(data_object, config)


def stress_reshape_transform(data_object: AtomicData, config) -> AtomicData:
    for k in data_object.keys():  # noqa: SIM118
        if "stress" in k and ("iso" not in k and "aniso" not in k):
            data_object[k] = data_object[k].reshape(1, 9)
    return data_object


def _random_rotation_matrix(
    num_transforms: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Sample a Haar-uniform 3D rotation matrix."""
    u1, u2, u3 = torch.rand(num_transforms, 3, device=device, dtype=dtype).unbind(-1)
    two_pi = 2.0 * torch.pi

    qx = torch.sqrt(1.0 - u1) * torch.sin(two_pi * u2)
    qy = torch.sqrt(1.0 - u1) * torch.cos(two_pi * u2)
    qz = torch.sqrt(u1) * torch.sin(two_pi * u3)
    qw = torch.sqrt(u1) * torch.cos(two_pi * u3)

    return torch.stack(
        [
            torch.stack(
                [
                    1.0 - 2.0 * (qy * qy + qz * qz),
                    2.0 * (qx * qy - qz * qw),
                    2.0 * (qx * qz + qy * qw),
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    2.0 * (qx * qy + qz * qw),
                    1.0 - 2.0 * (qx * qx + qz * qz),
                    2.0 * (qy * qz - qx * qw),
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    2.0 * (qx * qz - qy * qw),
                    2.0 * (qy * qz + qx * qw),
                    1.0 - 2.0 * (qx * qx + qy * qy),
                ],
                dim=-1,
            ),
        ],
        dim=-2,
    )


def _get_o3_transformation(data_object: AtomicData, config) -> torch.Tensor:
    device = data_object.pos.device
    dtype = data_object.pos.dtype
    num_transforms = data_object.num_graphs

    if "matrix" in config:
        transformation = torch.as_tensor(config["matrix"], device=device, dtype=dtype)
        if transformation.shape == (3, 3):
            transformation = transformation.expand(num_transforms, -1, -1)
        elif transformation.shape != (num_transforms, 3, 3):
            raise ValueError(
                "random_o3_transform matrix must have shape [3, 3] or "
                f"[{num_transforms}, 3, 3]."
            )
        return transformation

    if config.get("rotation", True):
        transformation = _random_rotation_matrix(num_transforms, device, dtype)
    else:
        transformation = torch.eye(3, device=device, dtype=dtype).expand(
            num_transforms, -1, -1
        )

    if config.get("inversion", True):
        inversion_probability = float(config.get("inversion_probability", 0.5))
        if not 0.0 <= inversion_probability <= 1.0:
            raise ValueError("inversion_probability must be between 0 and 1.")
        inversion_mask = (
            torch.rand(num_transforms, 1, 1, device=device) < inversion_probability
        )
        transformation = torch.where(inversion_mask, -transformation, transformation)

    return transformation


def _rotate_rows(values: torch.Tensor, transformation: torch.Tensor) -> torch.Tensor:
    return torch.bmm(values.unsqueeze(-2), transformation.transpose(-1, -2)).squeeze(
        -2
    )


def _get_value_transformations(
    data_object: AtomicData, values: torch.Tensor, transformation: torch.Tensor, key: str
) -> torch.Tensor:
    if values.shape[0] == data_object.pos.shape[0]:
        return transformation[data_object.batch]
    if values.shape[0] == data_object.num_graphs:
        return transformation
    raise ValueError(
        f"random_o3_transform could not map '{key}' with leading dimension "
        f"{values.shape[0]} to either atoms ({data_object.pos.shape[0]}) or "
        f"graphs ({data_object.num_graphs})."
    )


def _rotate_vectors(data_object: AtomicData, keys, transformation: torch.Tensor) -> None:
    for key in keys:
        if key in data_object:
            values = data_object[key]
            if not torch.is_tensor(values) or values.shape[-1] != 3:
                raise ValueError(
                    f"random_o3_transform expected '{key}' to be a tensor with "
                    f"last dimension 3, got {type(values)} with shape "
                    f"{getattr(values, 'shape', None)}."
                )
            value_transformation = _get_value_transformations(
                data_object, values, transformation, key
            )
            data_object[key] = _rotate_rows(values, value_transformation)


def _rotate_rank2_tensors(
    data_object: AtomicData, keys, transformation: torch.Tensor
) -> None:
    for key in keys:
        if key not in data_object:
            continue

        values = data_object[key]
        if not torch.is_tensor(values):
            raise ValueError(f"random_o3_transform expected '{key}' to be a tensor.")

        original_shape = values.shape
        if values.shape[-2:] == (3, 3):
            matrices = values
        elif values.shape[-1:] == (9,):
            matrices = values.reshape(values.shape[:-1] + (3, 3))
        else:
            raise ValueError(
                f"random_o3_transform expected '{key}' to have shape [..., 3, 3] "
                f"or [..., 9], got {values.shape}."
            )

        value_transformation = _get_value_transformations(
            data_object, matrices, transformation, key
        )
        rotated = torch.bmm(
            torch.bmm(value_transformation, matrices),
            value_transformation.transpose(-1, -2),
        )
        data_object[key] = rotated.reshape(original_shape)


def random_o3_transform(data_object: AtomicData, config) -> AtomicData:
    """Apply random rotation and optional inversion augmentation.

    This mirrors the Cartesian part of metatrain's RotationalAugmenter. Configure
    it only on training datasets. It updates geometry fields and matching
    Cartesian targets, while leaving scalar targets and graph connectivity
    unchanged.
    """
    transformation = _get_o3_transformation(data_object, config)

    data_object.pos = _rotate_rows(data_object.pos, transformation[data_object.batch])
    if "cell" in data_object:
        data_object.cell = torch.bmm(data_object.cell, transformation.transpose(-1, -2))

    vector_keys = list(config.get("vector_keys", ["forces"]))
    for dens_vector_key in ("force_data", "noise_vec"):
        if dens_vector_key in data_object and dens_vector_key not in vector_keys:
            vector_keys.append(dens_vector_key)
    rank2_keys = config.get("rank2_keys", ["stress"])
    _rotate_vectors(data_object, vector_keys, transformation)
    _rotate_rank2_tensors(data_object, rank2_keys, transformation)

    return data_object


def dens_transform(data_object: AtomicData, config) -> AtomicData:
    """Apply DeNS-style position noise and store a separate denoising target.

    This is intended as a collate-time transform so noise is resampled every epoch.
    Graphs selected for DeNS get:
    - noisy positions on a subset of atoms
    - original forces copied to ``force_data`` for model-side force encoding
    - atomwise added noise stored in ``noise_vec`` for DeNS loss computation
    - optional masking of graph-level targets such as energy/stress via ``inf``
    """

    if "batch" in data_object:
        batch = data_object.batch.long()
    else:
        batch = torch.zeros(
            data_object.pos.shape[0], device=data_object.pos.device, dtype=torch.long
        )
    num_graphs = data_object.num_graphs if "natoms" in data_object else 1
    num_atoms = int(data_object.pos.shape[0])
    device = data_object.pos.device
    dtype = data_object.pos.dtype

    graph_probability = float(config.get("graph_probability", 1.0))
    atom_probability = float(config.get("atom_probability", 1.0))
    noise_std = float(config.get("noise_std", 0.01))
    respect_fixed = bool(config.get("respect_fixed", True))
    mask_energy = bool(config.get("mask_energy", True))
    mask_stress = bool(config.get("mask_stress", True))
    max_force_norm = config.get("max_force_norm")
    max_corrupted_fraction = float(config.get("max_corrupted_fraction", 1.0))

    if not 0.0 <= graph_probability <= 1.0:
        raise ValueError("dens_transform graph_probability must be between 0 and 1.")
    if not 0.0 <= atom_probability <= 1.0:
        raise ValueError("dens_transform atom_probability must be between 0 and 1.")
    if noise_std < 0.0:
        raise ValueError("dens_transform noise_std must be non-negative.")
    if max_force_norm is not None and float(max_force_norm) < 0.0:
        raise ValueError("dens_transform max_force_norm must be non-negative.")
    if not 0.0 <= max_corrupted_fraction <= 1.0:
        raise ValueError(
            "dens_transform max_corrupted_fraction must be between 0 and 1."
        )

    dens_batch_mask = torch.rand(num_graphs, device=device) < graph_probability
    if max_force_norm is not None and "forces" in data_object:
        force_norms = torch.linalg.vector_norm(data_object.forces, dim=-1)
        per_graph_force_max = torch.full(
            (num_graphs,),
            0.0,
            device=device,
            dtype=force_norms.dtype,
        )
        per_graph_force_max.scatter_reduce_(
            0,
            batch,
            force_norms,
            reduce="amax",
            include_self=True,
        )
        dens_batch_mask = dens_batch_mask & (per_graph_force_max <= float(max_force_norm))

    eligible_atoms = dens_batch_mask[batch]
    if respect_fixed and "fixed" in data_object:
        eligible_atoms = eligible_atoms & (data_object.fixed == 0)

    noise_mask = eligible_atoms
    if atom_probability < 1.0:
        noise_mask = noise_mask & (torch.rand(num_atoms, device=device) < atom_probability)

    if max_corrupted_fraction < 1.0:
        for graph_idx in torch.nonzero(dens_batch_mask, as_tuple=False).flatten():
            graph_eligible = torch.nonzero(
                eligible_atoms & (batch == graph_idx), as_tuple=False
            ).flatten()
            if graph_eligible.numel() == 0:
                continue
            max_corrupted_atoms = max(
                1,
                math.floor(graph_eligible.numel() * max_corrupted_fraction),
            )
            graph_noisy = torch.nonzero(
                noise_mask & (batch == graph_idx), as_tuple=False
            ).flatten()
            if graph_noisy.numel() > max_corrupted_atoms:
                keep_order = torch.randperm(graph_noisy.numel(), device=device)[
                    :max_corrupted_atoms
                ]
                keep_mask = torch.zeros(
                    graph_noisy.numel(), device=device, dtype=torch.bool
                )
                keep_mask[keep_order] = True
                noise_mask[graph_noisy[~keep_mask]] = False

    if dens_batch_mask.any():
        for graph_idx in torch.nonzero(dens_batch_mask, as_tuple=False).flatten():
            graph_eligible = eligible_atoms & (batch == graph_idx)
            if graph_eligible.any() and not (noise_mask & (batch == graph_idx)).any():
                noise_mask[torch.nonzero(graph_eligible, as_tuple=False)[0, 0]] = True

    dens_batch_mask = torch.bincount(batch[noise_mask], minlength=num_graphs) > 0
    data_object.noise_mask = noise_mask
    data_object.dens_batch_mask = dens_batch_mask
    data_object.denoising_pos_forward = dens_batch_mask

    if "forces" in data_object:
        force_data = data_object.forces.clone()
    else:
        force_data = torch.zeros((num_atoms, 3), device=device, dtype=dtype)
    data_object.force_data = force_data

    noise = torch.randn_like(data_object.pos) * noise_std
    noise = noise * noise_mask.unsqueeze(-1).to(dtype)
    data_object.pos = data_object.pos + noise
    data_object.noise_vec = noise

    if mask_energy and "energy" in data_object:
        data_object.energy = data_object.energy.clone()
        data_object.energy[dens_batch_mask] = torch.inf

    if mask_stress and "stress" in data_object:
        data_object.stress = data_object.stress.clone()
        data_object.stress[dens_batch_mask] = torch.inf

    return data_object


def asedb_transform(data_object: AtomicData, config) -> AtomicData:
    data_object.dataset = config["dataset_name"]
    data_object.sid = str(
        data_object.sid.item() if torch.is_tensor(data_object) else data_object.sid
    )
    return data_object


class DataTransforms:
    def __init__(self, config) -> None:
        self.config = config

    def __call__(self, data_object):
        if not self.config:
            return data_object

        for transform_fn in self.config:
            # TODO: Normalization information used in the trainers. Ignore here for now
            # TODO: if we dont use them here, these should not be defined as "transforms" in the config
            # TODO: add them as another entry under dataset, maybe "standardize"?
            if transform_fn in ("normalizer", "element_references"):
                continue

            data_object = eval(transform_fn)(data_object, self.config[transform_fn])

        return data_object


def decompose_tensor(data_object, config) -> AtomicData:
    tensor_key = config["tensor"]
    rank = config["rank"]

    if tensor_key not in data_object:
        return data_object

    if rank != 2:
        raise NotImplementedError

    tensor_decomposition = torch.einsum(
        "ab, cb->ca",
        cg_change_mat(rank),
        data_object[tensor_key].reshape(1, irreps_sum(rank)),
    )

    for decomposition_key in config["decomposition"]:
        irrep_dim = config["decomposition"][decomposition_key]["irrep_dim"]
        data_object[decomposition_key] = tensor_decomposition[
            :,
            max(0, irreps_sum(irrep_dim - 1)) : irreps_sum(irrep_dim),
        ]

    return data_object
