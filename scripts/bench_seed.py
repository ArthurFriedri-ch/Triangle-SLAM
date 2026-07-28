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

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "mapper"))
sys.path.insert(0, os.path.join(_ROOT, "interface"))

# import patch.py by path: scene/__init__.py drags in the whole 3DGS stack
_spec = importlib.util.spec_from_file_location(
    "patch_bench", os.path.join(_ROOT, "mapper", "scene", "patch.py"))
patch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(patch)

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


class ShimModel:
    """What patch.py actually reads off a TriangleModel."""
    def __init__(self, vertices, faces):
        self.vertices = vertices
        self._triangle_indices = faces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records_dir", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--csv", default="keyframe_debug/bench_seed.csv")
    args = ap.parse_args()

    patches, loops, verts, faces, off = [], [], [], [], 0
    worst, total = 0.0, 0.0
    for i, record in enumerate(slam_interface.iter_records(args.records_dir)):
        if i >= args.n:
            break
        kf = Keyframe(record)
        model = ShimModel(torch.cat(verts), torch.cat(faces)) if patches else None

        t0 = time.perf_counter()
        p, _tile, ids = patch.seed_patch_cdt(kf, model, model_loops=loops or None,
                                             debug=False)
        dt = time.perf_counter() - t0
        worst = max(worst, dt)
        total += dt

        if p is None:
            print(f"kf {record.index}: no patch  ({dt:.2f}s)")
            print("  " + frame_end(record.index, args.csv))
            continue

        # the seam must be identity, not a re-measurement
        m = ids >= 0
        if m.any() and model is not None:
            err = (p.vertices[torch.from_numpy(np.where(m)[0])]
                   - model.vertices[torch.from_numpy(ids[m])]).abs().max().item()
            assert err == 0.0, f"kf {record.index}: weld vertex off by {err}"

        patches.append(p)
        verts.append(p.vertices)
        faces.append(p.faces + off)
        for r in p.boundary_loops():
            loops.append(r + off)
        off += len(p.vertices)

        print(f"kf {record.index}: {len(p.faces):6d} tris  {int(m.sum()):5d} weldable"
              f"  map={off:7d} verts  {dt:6.2f}s")
        print("  " + frame_end(record.index, args.csv))

    print()
    print(report())
    print(f"\n{len(patches)} patches, worst {worst:.2f}s, mean "
          f"{total / max(len(patches), 1):.2f}s/keyframe")


if __name__ == "__main__":
    main()
