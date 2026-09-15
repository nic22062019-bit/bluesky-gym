#!/usr/bin/env python3
"""
UTM / bluesky-gym: fix numpy>=2 incompatibility in bluesky-simulator.

Bug: in bluesky/traffic/windfield.py the final return does
    return float(vnorth), float(veast)
where vnorth/veast are 1-element 1-D numpy arrays. With numpy>=2, float()
on a 1-D array raises "only 0-dimensional arrays can be converted to Python
scalars", which breaks conflict generation (creconfs) in bluesky-gym envs
(HorizontalCREnv, SectorCREnv, MergeEnv, ...).

Fix: use .item() (numpy scalar -> python float), valid for both 0-d and 1-d.
This script locates the installed windfield.py inside a Python environment and
applies the fix idempotently.

Usage:
    python patch_numpy2_bluesky.py            # patch current interpreter's site-packages
    VENV=/root/bluesky_venv python patch_numpy2_bluesky.py   # patch a specific venv
"""
import os, sys, glob

VENV = os.environ.get("VENV", "")


def find_windfield():
    cands = []
    if VENV:
        base = os.path.join(VENV, "lib")
        cands += glob.glob(base + "/python*/site-packages/bluesky/traffic/windfield.py")
    else:
        import bluesky  # noqa
        cands.append(
            os.path.join(os.path.dirname(bluesky.__file__), "traffic", "windfield.py")
        )
    return cands


def patch(path):
    src = open(path).read()
    old = "            return float(vnorth),float(veast)"
    new = "            return float(vnorth.item()),float(veast.item())"
    if old in src:
        open(path, "w").write(src.replace(old, new))
        return "PATCHED"
    return "ALREADY_OK / no-op"


def main():
    done = []
    for p in find_windfield():
        status = patch(p)
        done.append(f"{p} -> {status}")
        print(f"* {p} -> {status}")
    if not done:
        print("windfield.py not found; run inside an env with bluesky-simulator installed")
        sys.exit(1)


if __name__ == "__main__":
    main()