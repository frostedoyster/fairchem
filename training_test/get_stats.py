import sys
import torch
import tqdm
import yaml

from fairchem.core.datasets import AseDBDataset


ds = AseDBDataset(config=dict(src=sys.argv[1]))
unique_elements = set()
for atoms in tqdm.tqdm(ds):
    unique_elements.update([int(a) for a in atoms["atomic_numbers"]])
unique_elements = sorted(unique_elements)

print("Unique elements in dataset:", unique_elements)

atomic_number_to_index = torch.full((max(unique_elements) + 1,), fill_value=1e10, dtype=torch.long)
for i, element in enumerate(unique_elements):
    atomic_number_to_index[element] = i
index_to_atomic_number = torch.tensor(unique_elements)

XTX = torch.zeros((len(unique_elements), len(unique_elements)))
XTY = torch.zeros(len(unique_elements))

for atoms in tqdm.tqdm(ds):
    atomic_numbers = atoms["atomic_numbers"]
    energy = atoms["energy"]
    natoms = len(atomic_numbers)
    x = torch.index_add(torch.zeros(len(unique_elements)), 0, atomic_number_to_index[atomic_numbers], torch.ones(natoms)) / natoms
    y = energy / natoms
    XTX += torch.outer(x, x)
    XTY += x * y

c = torch.linalg.solve(XTX, XTY)

energy_sum_of_squares = 0.0
energy_count = 0
force_sum_of_squares = 0.0
force_count = 0
stress_sum_of_squares = 0.0
stress_count = 0

for i in range(len(ds)):
    atoms = ds[i]
    atomic_numbers = atoms["atomic_numbers"]
    energy = atoms["energy"]
    natoms = len(atomic_numbers)
    x = torch.index_add(torch.zeros(len(unique_elements)), 0, atomic_number_to_index[atomic_numbers], torch.ones(natoms)) / natoms
    y = energy / natoms
    energy_pred = x @ c
    energy_sum_of_squares += torch.square(energy_pred - y).item()
    energy_count += 1

    forces = atoms["forces"]
    force_sum_of_squares += float(torch.sum(forces ** 2))
    force_count += forces.numel()

    stress = atoms["stress"].flatten()
    mask = torch.logical_not(torch.isinf(stress))
    stress = stress[mask]
    stress_sum_of_squares += float(torch.sum(stress ** 2))
    stress_count += stress.numel()

energy_rmse = (energy_sum_of_squares / energy_count) ** 0.5
force_rmse = (force_sum_of_squares / force_count) ** 0.5
stress_rmse = (stress_sum_of_squares / stress_count) ** 0.5

element_references = [0.0] * (max(unique_elements) + 1)
for element, coefficient in zip(unique_elements, c.tolist()):
    element_references[element] = coefficient

with open(sys.argv[2] if len(sys.argv) > 2 else "dataset_stats.yaml", "w") as f:
    yaml.safe_dump(
        {
            "energy": {
                "per_atom_residual_rmsd": energy_rmse,
                "element_references": element_references,
            },
            "forces": {"rmsd": force_rmse},
            "stress": {"rmsd": stress_rmse},
        },
        f,
        sort_keys=False,
    )
