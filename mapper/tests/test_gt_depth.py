"""Invariants for the ground-truth depth loader.

Run directly (`python mapper/tests/test_gt_depth.py`) or under pytest.
"""
import os
import shutil
import sys
import tempfile
from types import SimpleNamespace

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_MAPPER = os.path.dirname(_HERE)
sys.path.insert(0, _MAPPER)

from utils.gt_depth import (GtDepthSource, PNG_SCALE_REPLICA,  # noqa: E402
                            depth_dir_for)

H, W = 64, 96
K_REPLICA = np.array([[308.0, 0, 307.7], [0, 308.0, 172.0], [0, 0, 1.0]])
# the global factor a monocular reconstruction is undetermined by
TRUE_SCALE = 2.45


def _metric_depth(i):
    """A distinctive depth map per frame, so a wrong pairing shows up."""
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    return (1.5 + 0.5 * np.sin(0.15 * x + 0.4 * i)
            + 0.4 * np.cos(0.11 * y - 0.3 * i)).astype(np.float32)


def _make(n=8, names="depth{:06d}.png", scale=PNG_SCALE_REPLICA,
          reverse=False, indices=None):
    """Write PNGs plus matching fake records. Returns (dir, records)."""
    d = tempfile.mkdtemp()
    indices = indices if indices is not None else list(range(n))
    metrics = [_metric_depth(i) for i in range(n)]
    order = list(reversed(metrics)) if reverse else metrics
    for k, (idx, m) in enumerate(zip(indices, order)):
        png = np.clip(m * scale, 0, 65535).astype(np.uint16)
        cv2.imwrite(os.path.join(d, names.format(idx)), png)
    records = [SimpleNamespace(index=idx,
                               depth=metrics[k] / TRUE_SCALE,   # SLAM scale
                               intrinsics=K_REPLICA)
               for k, idx in enumerate(indices)]
    return d, records


def test_recovers_scale_and_lands_in_pose_frame():
    """GT depth must come back in the record's scale, not in metres.

    The SLAM is monocular, so its poses carry an arbitrary global factor.
    Handing back metres would put geometry TRUE_SCALE times too far out and
    patches from different keyframes would not overlap.
    """
    d, recs = _make()
    try:
        src = GtDepthSource(d).calibrate(recs)
        assert abs(src.scale - TRUE_SCALE) < 0.05 * TRUE_SCALE, \
            f"estimated scale {src.scale:.3f}, wanted {TRUE_SCALE}"
        got = src.depth_for(recs[3], 3)
        m = (got > 0) & (recs[3].depth > 0)
        rel = np.abs(got[m] - recs[3].depth[m]) / recs[3].depth[m]
        assert rel.max() < 0.02, f"depth off by up to {rel.max():.1%} of the record"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_matches_stem_by_record_index():
    """Records are indexed by original frame number, not keyframe order."""
    idx = [0, 2, 5, 16, 20, 25, 29, 32]
    d, recs = _make(n=len(idx), indices=idx)
    try:
        src = GtDepthSource(d).calibrate(recs)
        assert src.strategy == "stem:depth{i:06d}", src.strategy
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_falls_back_to_ordinal_for_timestamp_names():
    """TUM-style names carry no index, so pairing is positional."""
    idx = [0, 2, 5, 16, 20, 25, 29, 32]
    names = "13050314{:02d}.374112.png".format
    d = tempfile.mkdtemp()
    try:
        metrics = [_metric_depth(i) for i in range(len(idx))]
        for k, m in enumerate(metrics):
            cv2.imwrite(os.path.join(d, names(50 + k)),
                        np.clip(m * PNG_SCALE_REPLICA, 0, 65535).astype(np.uint16))
        recs = [SimpleNamespace(index=i, depth=metrics[k] / TRUE_SCALE,
                                intrinsics=K_REPLICA)
                for k, i in enumerate(idx)]
        src = GtDepthSource(d).calibrate(recs)
        assert src.strategy == "ordinal", src.strategy
        assert abs(src.scale - TRUE_SCALE) < 0.05 * TRUE_SCALE
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_rejects_a_misaligned_folder():
    """Silently pairing the wrong depth with a keyframe is the worst outcome."""
    d, recs = _make(n=8, names="13050314{:02d}.374112.png", reverse=True,
                    indices=list(range(50, 58)))
    try:
        try:
            GtDepthSource(d).calibrate(recs)
        except RuntimeError as e:
            assert "correlat" in str(e), str(e)
            return
        raise AssertionError("accepted a scrambled depth folder")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_wrong_png_scale_is_absorbed_by_alignment():
    """A mis-specified png_scale must not distort geometry, only the reported metres.

    Both factors are global, so the estimated alignment soaks up the error and
    the depth still lands in the record's frame.
    """
    d, recs = _make()
    try:
        src = GtDepthSource(d, png_scale=1000.0).calibrate(recs)
        got = src.depth_for(recs[2], 2)
        m = (got > 0) & (recs[2].depth > 0)
        rel = np.abs(got[m] - recs[2].depth[m]) / recs[2].depth[m]
        assert rel.max() < 0.02, f"off by {rel.max():.1%} despite alignment"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_depth_dir_for():
    d = tempfile.mkdtemp()
    try:
        assert depth_dir_for(os.path.join(d, "nope_records")) is None
        rec = os.path.join(d, "seq_records")
        os.makedirs(rec + "_depth")
        assert depth_dir_for(rec) == rec + "_depth"
        assert depth_dir_for(rec + os.sep) == rec + "_depth"
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    bad = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            bad += 1
            print(f"  FAIL  {name}: {e}")
    print(f"\n{len(fns) - bad}/{len(fns)} passed")
    sys.exit(1 if bad else 0)
