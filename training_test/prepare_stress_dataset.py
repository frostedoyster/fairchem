from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect
from ase.io import iread


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert mad-test.xyz to one ASE DB, using inf stress targets as masks "
            "for frames without stress."
        )
    )
    parser.add_argument(
        "xyz",
        type=Path,
        help="Input extended XYZ file.",
        required=True,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "mad-test_mixed_db",
        help="Output directory containing data.db and metadata.npz.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output data.db and metadata.npz.",
    )
    args = parser.parse_args()

    xyz_path = args.xyz.resolve()
    out_dir = args.out_dir.resolve()
    db_path = out_dir / "data.db"
    metadata_path = out_dir / "metadata.npz"

    if not xyz_path.is_file():
        raise FileNotFoundError(f"Missing input dataset: {xyz_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        db_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
    elif db_path.exists() or metadata_path.exists():
        raise FileExistsError(
            f"{out_dir} already contains data.db or metadata.npz. "
            "Use --overwrite to replace them."
        )

    natoms = []
    stress_count = 0
    missing_stress_count = 0
    with connect(db_path) as db:
        for frame_idx, atoms in enumerate(iread(xyz_path, index=":", format="extxyz")):
            results = getattr(atoms.calc, "results", {}) if atoms.calc else {}
            if "stress" not in results:
                calc_kwargs = {
                    key: results[key]
                    for key in ("energy", "forces")
                    if key in results
                }
                calc_kwargs["stress"] = np.full((3, 3), np.inf)
                atoms.calc = SinglePointCalculator(atoms, **calc_kwargs)
                missing_stress_count += 1
            else:
                stress_count += 1

            db.write(atoms, sid=str(xyz_path), fid=frame_idx)
            natoms.append(len(atoms))

    np.savez(metadata_path, natoms=np.asarray(natoms, dtype=np.int32))
    print(f"Wrote {len(natoms)} structures to {db_path}")
    print(f"Finite stress targets: {stress_count}")
    print(f"Inf stress masks: {missing_stress_count}")
    print(f"Wrote atom-count metadata to {metadata_path}")


if __name__ == "__main__":
    main()
