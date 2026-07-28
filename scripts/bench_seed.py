"""Replay recorded keyframes through the CDT seeder alone, on CPU.

No CUDA, no rasterizer, no TriangleModel -- `seed_patch_cdt` only touches
`model.vertices` and `model._triangle_indices`, so a two-attribute shim stands in
for the real model. That makes the seeder's cost and the weld-table health
measurable without a GPU box.

    MAPPER_TIME=1 python scripts/bench_seed.py --records_dir <dir> --n 30

Occlusion culling is off here: it needs the rendered depth, which needs the GPU.
"""
import argparse
import importlib.util
import os
import sys
import time

import types

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "mapper"))
sys.path.insert(0, os.path.join(_ROOT, "interface"))

# Register a bare `scene` package and load the two modules by path:
# scene/__init__.py drags in the whole 3DGS dataset stack, and global_mesh
# imports `scene.patch`, so both must resolve to the same module object.
_pkg = types.ModuleType("scene")
_pkg.__path__ = [os.path.join(_ROOT, "mapper", "scene")]
sys.modules.setdefault("scene", _pkg)
for _n in ("patch", "global_mesh"):
    _spec = importlib.util.spec_from_file_location(
        "scene." + _n, os.path.join(_ROOT, "mapper", "scene", _n + ".py"))
    _m = importlib.util.module_from_spec(_spec)
    sys.modules["scene." + _n] = _m
    _spec.loader.exec_module(_m)
patch = sys.modules["scene.patch"]
GlobalMesh = sys.modules["scene.global_mesh"].GlobalMesh

from utils.timing import frame_end, report  # noqa: E402
import slam_interface                       # noqa: E402


class Keyframe:
    def __init__(self, record):
        self.index = record.index
        self.depth = torch.from_numpy(record.depth).float()
        self.cov = torch.from_numpy(record.depth_cov).float()
        self.rgb = torch.from_numpy(record.rgb).float() / 255.0
        self.K = torch.from_numpy(record.intrinsics).float()
        pose = record.pose
        if record.pose_frame == slam_interface.POSE_FRAME_W2C:
            pose = np.linalg.inv(pose)
        self.c2w = torch.from_numpy(pose).float()


def _log_timing(line):
    """frame_end returns "" when MAPPER_TIME is off; don't print blank lines."""
    if line:
        print("  " + line)


class ShimModel:
    """What patch.py actually reads off a TriangleModel."""
    def __init__(self, mesh):
        self.vertices = mesh.vertices
        self._triangle_indices = torch.from_numpy(mesh.faces)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records_dir", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--csv", default="keyframe_debug/bench_seed.csv")
    args = ap.parse_args()

    mesh = GlobalMesh(device="cpu")
    n_patches = 0
    worst, total = 0.0, 0.0
    for i, record in enumerate(slam_interface.iter_records(args.records_dir)):
        if i >= args.n:
            break
        kf = Keyframe(record)
        model = ShimModel(mesh) if not mesh.is_empty() else None
        loops = mesh.boundary_loops() if model else None
        seed_version = mesh.version

        t0 = time.perf_counter()
        p, _tile, ids = patch.seed_patch_cdt(kf, model, model_loops=loops,
                                             debug=False)
        dt = time.perf_counter() - t0
        worst = max(worst, dt)
        total += dt

        if p is None:
            print(f"kf {record.index}: no patch  ({dt:.2f}s)")
            _log_timing(frame_end(record.index, args.csv))
            continue

        # the seam must be identity, not a re-measurement
        m = ids >= 0
        if m.any() and model is not None:
            err = (p.vertices[torch.from_numpy(np.where(m)[0])]
                   - model.vertices[torch.from_numpy(ids[m])]).abs().max().item()
            assert err == 0.0, f"kf {record.index}: weld vertex off by {err}"

        rec = mesh.weld(p, ids, kf_index=record.index, seed_version=seed_version,
                        splits=getattr(p, "edge_splits", None))
        n_patches += 1

        print(f"kf {record.index}: {rec.n_faces:6d} tris  {int(m.sum()):5d} weldable"
              f"  +{rec.n_vertices:5d} new  {mesh.stats()}  {dt:6.2f}s")
        _log_timing(frame_end(record.index, args.csv))

    print()
    print(report())
    print(f"\n{mesh.stats()}")
    print(f"{n_patches} patches, worst {worst:.2f}s, mean "
          f"{total / max(n_patches, 1):.2f}s/keyframe")


if __name__ == "__main__":
    main()
