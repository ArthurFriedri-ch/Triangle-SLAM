"""Ground-truth depth PNGs as a drop-in replacement for the SLAM depth.

Records carry depth estimated by DROID-SLAM running *monocular* (see
`third_party/nerf-slam/datasets/tum_dataset.py`: it reads rgb.txt only). A
monocular reconstruction is determined only up to a global scale, and the poses
in the record share that scale. Measured on office0, where the records already
carry a populated `gt_depth`:

    median(gt_metres) / median(droid_depth) = 2.4497 +- 0.0345  (n=9 keyframes)

Constant to 1.4% across the sequence, which is the signature of a single global
scale factor rather than per-frame error. So metric depth cannot be substituted
directly: back-projected geometry would sit 2.45x further out than the camera
baselines imply and patches from different keyframes would not overlap at all.

This module therefore estimates that factor once and divides the GT depth by it,
which keeps depth and poses in the same frame while still buying the thing that
actually matters -- the *shape* of the depth map, which DROID gets badly wrong on
some frames (correlation 0.20 with GT on office0 kf_000064).

Because the alignment factor is estimated rather than assumed, an incorrect
`png_scale` is absorbed into it and the geometry is still self-consistent; only
the reported "metres" would be off.
"""
import glob
import os
import re

import cv2
import numpy as np

# depth_metres = png_value / png_scale
PNG_SCALE_TUM = 5000.0        # TUM RGB-D convention
PNG_SCALE_REPLICA = 6553.5    # Replica convention

# TUM freiburg1 radial-tangential distortion, copied from
# third_party/nerf-slam/datasets/tum_dataset.py:21. The loader undistorts the
# RGB with these, so raw TUM depth PNGs must be undistorted the same way or they
# sit several pixels off at the image edges.
TUM_FR1_DIST = np.array([0.262383, -0.953104, -0.005358, 0.002628, 1.163314])
TUM_FR1_FX = 517.306408

_STEM_PATTERNS = ("kf_{i:06d}", "{i:06d}", "depth{i:06d}", "depth_{i:06d}",
                  "frame{i:06d}", "frame_{i:06d}", "{i}")


def _looks_like_tum(K):
    return abs(float(K[0, 0]) - TUM_FR1_FX) < 5.0


def default_png_scale(K):
    return PNG_SCALE_TUM if _looks_like_tum(K) else PNG_SCALE_REPLICA


class GtDepthSource:
    """Maps records to ground-truth depth PNGs in a sibling `*_depth` folder."""

    def __init__(self, depth_dir, png_scale=None, align=True, undistort="auto",
                 max_depth=50.0):
        self.dir = depth_dir
        self.png_scale = png_scale
        self.align = align
        self.undistort = undistort
        self.max_depth = max_depth

        self.files = sorted(
            f for ext in ("png", "PNG")
            for f in glob.glob(os.path.join(depth_dir, f"*.{ext}")))
        if not self.files:
            raise FileNotFoundError(f"no PNGs in {depth_dir}")

        self._stems = {os.path.splitext(os.path.basename(f))[0]: f
                       for f in self.files}
        self.strategy = None      # "stem:<pattern>" | "index" | "ordinal"
        self.scale = 1.0          # metres -> record/pose frame
        self._map_corr = None
        self._undistort_maps = None
        self._report = []

    # -- loading -----------------------------------------------------------

    def _raw(self, path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"could not read {path}")
        if img.ndim == 3:
            img = img[..., 0]
        return img.astype(np.float32)

    def _path_for(self, record, ordinal):
        if self.strategy is None:
            return None
        if self.strategy.startswith("stem:"):
            pat = self.strategy[5:]
            return self._stems.get(pat.format(i=int(record.index)))
        idx = int(record.index) if self.strategy == "index" else ordinal
        return self.files[idx] if 0 <= idx < len(self.files) else None

    def _undistort_for(self, K, shape):
        if self._undistort_maps is None:
            H, W = shape
            self._undistort_maps = cv2.initUndistortRectifyMap(
                K, TUM_FR1_DIST, None, K, (W, H), cv2.CV_32FC1)
        m1, m2 = self._undistort_maps
        return m1, m2

    def depth_for(self, record, ordinal=None):
        """GT depth for `record`, in the record's own (pose) scale. 0 = invalid."""
        path = self._path_for(record, ordinal)
        if path is None:
            return None
        d = self._raw(path) / self.png_scale                 # metres

        if d.shape != record.depth.shape:
            d = cv2.resize(d, (record.depth.shape[1], record.depth.shape[0]),
                           interpolation=cv2.INTER_NEAREST)

        want = self.undistort
        if want == "auto":
            want = _looks_like_tum(record.intrinsics)
        if want:
            # nearest-neighbour: linear would blend across depth discontinuities
            # and smear the 0 = "no return" pixels into real geometry
            m1, m2 = self._undistort_for(np.asarray(record.intrinsics, np.float64),
                                         d.shape)
            d = cv2.remap(d, m1, m2, cv2.INTER_NEAREST, borderValue=0.0)

        d[~np.isfinite(d)] = 0.0
        if self.max_depth:
            d[d > self.max_depth] = 0.0
        return (d / self.scale).astype(np.float32)           # into pose scale

    # -- calibration -------------------------------------------------------

    def calibrate(self, records, min_corr=0.5):
        """Pick the index->file mapping and the global scale, then verify.

        The mapping is chosen by correlating candidate GT frames against the
        record's own depth: a correct pairing correlates strongly (~0.95 on
        office0), a misaligned one does not. This is what makes the feature
        safe when the PNGs carry original dataset filenames whose relation to
        the record index is not otherwise recoverable.
        """
        if not records:
            raise ValueError("calibrate needs at least one record")
        K = records[0].intrinsics
        if self.png_scale is None:
            self.png_scale = default_png_scale(K)
            self._report.append(
                f"png_scale={self.png_scale} "
                f"({'TUM' if _looks_like_tum(K) else 'Replica'}, from intrinsics fx="
                f"{float(K[0, 0]):.1f})")

        cands = [f"stem:{p}" for p in _STEM_PATTERNS] + ["index", "ordinal"]
        best, best_corr = None, -np.inf
        saved_scale, self.scale = self.scale, 1.0
        for strat in cands:
            self.strategy = strat
            corrs = []
            for ordinal, rec in enumerate(records):
                try:
                    g = self.depth_for(rec, ordinal)
                except Exception:
                    g = None
                if g is None:
                    continue
                d = rec.depth
                m = (g > 0) & (d > 0) & (d < self.max_depth) & np.isfinite(d)
                if m.sum() < 500:
                    continue
                corrs.append(np.corrcoef(g[m], d[m])[0, 1])
            if len(corrs) >= max(1, len(records) // 2):
                c = float(np.median(corrs))
                if c > best_corr:
                    best, best_corr = strat, c
        self.strategy, self._map_corr = best, best_corr
        self.scale = saved_scale

        if best is None:
            raise RuntimeError(
                f"could not match any record to a PNG in {self.dir}. "
                f"Files look like: {[os.path.basename(f) for f in self.files[:3]]}")
        if best_corr < min_corr:
            raise RuntimeError(
                f"best index->file mapping ({best}) correlates only "
                f"{best_corr:.2f} with the record depth, below {min_corr}. The "
                f"PNGs are probably misaligned with the keyframes. Files look "
                f"like: {[os.path.basename(f) for f in self.files[:3]]}")
        self._report.append(f"mapping={best} (depth correlation {best_corr:.3f})")

        # global scale: monocular SLAM fixes geometry only up to one factor
        self.scale = 1.0
        if self.align:
            ratios = []
            for ordinal, rec in enumerate(records):
                g = self.depth_for(rec, ordinal)
                if g is None:
                    continue
                d = rec.depth
                m = (g > 0) & (d > 0) & (d < self.max_depth) & np.isfinite(d)
                if m.sum() < 500:
                    continue
                ratios.append(float(np.median(g[m] / d[m])))
            if not ratios:
                raise RuntimeError("no overlapping pixels to estimate the scale")
            self.scale = float(np.median(ratios))
            spread = float(np.std(ratios) / max(self.scale, 1e-9))
            self._report.append(
                f"scale={self.scale:.4f} metres per record unit "
                f"(spread {spread:.1%} over {len(ratios)} keyframes)")
            if spread > 0.10:
                self._report.append(
                    "  WARNING: scale is not constant across keyframes. Either "
                    "the mapping is wrong or the trajectory drifts in scale; "
                    "geometry from different keyframes will not line up.")
        return self

    def summary(self):
        head = f"[gt-depth] {len(self.files)} PNGs in {self.dir}"
        return "\n".join([head] + [f"[gt-depth]   {r}" for r in self._report])


def depth_dir_for(records_dir):
    """`<records_dir>` -> `<records_dir>_depth`, or None if absent."""
    d = os.path.normpath(records_dir).rstrip(os.sep) + "_depth"
    return d if os.path.isdir(d) else None
