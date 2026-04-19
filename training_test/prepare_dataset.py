from __future__ import annotations

from pathlib import Path

import numpy as np
from ase.db import connect
from ase.io import iread


def main() -> None:
    root = Path(__file__).resolve().parent
    xyz_path = root / "ethanol_train_0.xyz"
    db_path = root / "ethanol_train_0.db"
    metadata_path = root / "metadata.npz"

    if not xyz_path.is_file():
        raise FileNotFoundError(f"Missing input dataset: {xyz_path}")

    if db_path.exists():
        db_path.unlink()

    natoms = []
    with connect(db_path) as db:
        for frame_idx, atoms in enumerate(iread(xyz_path, index=":", format="extxyz")):
            atoms.info["sid"] = str(xyz_path)
            atoms.info["fid"] = frame_idx
            db.write(atoms)
            natoms.append(len(atoms))

    np.savez(metadata_path, natoms=np.asarray(natoms, dtype=np.int32))
    print(f"Wrote {len(natoms)} structures to {db_path}")
    print(f"Wrote atom-count metadata to {metadata_path}")


if __name__ == "__main__":
    main()
