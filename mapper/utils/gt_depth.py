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

    def __init__(self, depth_dir, png_scale=None, align="per_frame",
                 undistort="auto", max_depth=50.0, force_strategy=None):
        self.dir = depth_dir
        self.png_scale = png_scale
        self.force_strategy = force_strategy
        if align is True:
            align = "per_frame"
        elif align is False:
            align = "none"
        if align not in ("per_frame", "global", "none"):
            raise ValueError(f"align must be per_frame/global/none, got {align!r}")
        self.align = align
        self.scale_spread = None
        self.last_frame_scale = None
        self.undistort = undistort
        self.max_depth = max_depth

        self.files = sorted(
            f for ext in ("png", "PNG")
            for f in glob.glob(os.path.join(depth_dir, f"*.{ext}")))
        if not self.files:
            raise FileNotFoundError(f"no PNGs in {depth_dir}")

        self._stems = {os.path.splitext(os.path.basename(f))[0]: f
                       for f in self.files}
        self._assoc = self._load_association()
        self.strategy = None      # "assoc" | "stem:<pattern>" | "index" | "ordinal"
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

    @staticmethod
    def _read_tum_index(path):
        ts, names = [], []
        with open(path) as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln or ln.startswith("#"):
                    continue
                parts = ln.split()
                if len(parts) < 2:
                    continue
                ts.append(float(parts[0]))
                names.append(os.path.basename(parts[1]))
        return np.asarray(ts, np.float64), names

    def _load_association(self):
        """TUM rgb.txt + depth.txt, the only correct pairing for that dataset.

        The RGB and depth streams are captured at different instants and have
        different lengths (freiburg1_room: 1362 vs 1360), so pairing by position
        drifts. `record.index` is a row index into rgb.txt -- see
        `third_party/nerf-slam/datasets/tum_dataset.py`, which builds its frame
        list from rgb.txt alone -- so the record's timestamp is rgb_ts[index],
        and the depth frame is whichever is nearest in time.
        """
        # Only look beside the folder when it is literally a TUM `depth/`
        # directory. Searching the parent unconditionally is unsafe: for
        # `data/office0_records_depth` the parent is `data/`, shared by every
        # dataset, so one stray rgb.txt there would silently repair-pair
        # Replica -- which is synthetic and needs no association at all.
        bases = [self.dir]
        if os.path.basename(os.path.normpath(self.dir)) == "depth":
            bases.append(os.path.dirname(os.path.normpath(self.dir)))
        for base in bases:
            rgb_txt = os.path.join(base, "rgb.txt")
            dep_txt = os.path.join(base, "depth.txt")
            if os.path.isfile(rgb_txt) and os.path.isfile(dep_txt):
                rgb_ts, _ = self._read_tum_index(rgb_txt)
                dep_ts, dep_names = self._read_tum_index(dep_txt)
                if len(rgb_ts) and len(dep_ts):
                    o = np.argsort(dep_ts)
                    return {"rgb_ts": rgb_ts, "dep_ts": dep_ts[o],
                            "dep_names": [dep_names[i] for i in o],
                            "src": base, "max_dt": 0.0}
        return None

    def _assoc_path(self, record, max_dt=0.05):
        a = self._assoc
        k = int(record.index)
        if a is None or not (0 <= k < len(a["rgb_ts"])):
            return None
        t = a["rgb_ts"][k]
        j = int(np.searchsorted(a["dep_ts"], t))
        best, bd = None, np.inf
        for c in (j - 1, j):
            if 0 <= c < len(a["dep_ts"]):
                d = abs(a["dep_ts"][c] - t)
                if d < bd:
                    best, bd = c, d
        if best is None or bd > max_dt:
            return None
        a["max_dt"] = max(a["max_dt"], bd)
        return self._stems.get(os.path.splitext(a["dep_names"][best])[0])

    def _path_for(self, record, ordinal):
        if self.strategy is None:
            return None
        if self.strategy == "assoc":
            return self._assoc_path(record)
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

        s = self.scale
        if self.align == "per_frame":
            # Monocular scale drifts along the trajectory (measured on
            # freiburg1_room: 1.73 at the start to 1.45 at the end, correlation
            # -0.707 with keyframe index), so one global factor cannot hold.
            # Pose k and the SLAM depth at k come out of the same bundle
            # adjustment and agree locally, so matching this keyframe's own
            # factor is what keeps back-projected geometry consistent with its
            # pose -- while still taking the shape from ground truth.
            local = self._frame_scale(d, record.depth)
            if local is not None:
                s = local
        # metres per record unit for THIS keyframe, so downstream evaluation can
        # report in metric units even under per-frame alignment
        self.last_frame_scale = s
        return (d / s).astype(np.float32)                    # into pose scale

    def _frame_scale(self, metric, slam_depth, min_px=500):
        m = ((metric > 0) & (slam_depth > 0) & (slam_depth < self.max_depth)
             & np.isfinite(slam_depth))
        if m.sum() < min_px:
            return None
        return float(np.median(metric[m] / slam_depth[m]))

    # -- calibration -------------------------------------------------------

    def _score(self, records):
        """Median correlation of the current mapping against the record depth."""
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
        if len(corrs) < max(1, len(records) // 2):
            return None
        return float(np.median(corrs))

    def calibrate(self, records, min_corr=0.5, max_scale_spread=0.10):
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

        saved_scale, self.scale = self.scale, 1.0
        if self.force_strategy:
            self.strategy = self.force_strategy
            best_corr = self._score(records)
            if best_corr is None:
                raise RuntimeError(
                    f"forced mapping {self.force_strategy} matched no records "
                    f"in {self.dir}")
        elif self._assoc is not None:
            # The dataset ships its own pairing. Do not put that in a contest
            # against positional guesses: consecutive depth frames at 30 Hz are
            # nearly identical, and against a noisy monocular depth estimate the
            # correlation ceiling is only ~0.74, so a one-frame offset barely
            # moves the score. Measured on freiburg1_room with a probe spanning
            # the sequence, the wrong positional mapping actually scored *higher*
            # (0.744) than the correct timestamp association (0.728), while the
            # streams demonstrably diverge for 1014 of 1362 rows.
            self.strategy = "assoc"
            best_corr = self._score(records)
        else:
            # A filename that encodes the record index is an explicit pairing,
            # so accept it structurally the way `assoc` is accepted. Replica is
            # synthetic with rgb and depth rendered 1:1, and depth{index}.png
            # says so outright -- there is nothing to arbitrate. Correlation
            # must not decide it: here it compares a clean synthetic depth map
            # against monocular SLAM depth, and a positional mapping can score
            # higher than the correct one, exactly as it did on TUM.
            self.strategy = None
            for p in _STEM_PATTERNS:
                cand = f"stem:{p}"
                self.strategy = cand
                if all(self._path_for(r, i) is not None
                       for i, r in enumerate(records)):
                    break
                self.strategy = None
            if self.strategy is not None:
                best_corr = self._score(records)
            else:
                # nothing structural to go on; positional guesses get scored
                best, best_corr = None, -np.inf
                for strat in ("index", "ordinal"):
                    self.strategy = strat
                    c = self._score(records)
                    if c is not None and c > best_corr:
                        best, best_corr = strat, c
                self.strategy = best
        self._map_corr = best_corr
        self.scale = saved_scale

        if self.strategy is None or best_corr is None:
            raise RuntimeError(
                f"could not match any record to a PNG in {self.dir}. "
                f"Files look like: {[os.path.basename(f) for f in self.files[:3]]}")
        # A hard floor applies only to *guessed* pairings. `assoc` and a
        # filename carrying the record index are explicit statements about which
        # depth belongs to which keyframe, and that evidence outranks a
        # correlation against noisy monocular depth -- which we have twice seen
        # rank a wrong mapping above the right one. A low score there still
        # means something is off, but it is not the pairing.
        if best_corr < min_corr and not self.force_strategy:
            raise RuntimeError(
                f"mapping {self.strategy} correlates only {best_corr:.2f} with "
                f"the record depth, below {min_corr}. Even an explicit-looking "
                f"pairing can be wrong: office0's records embed a gt_depth that "
                f"matches depth<index>.png only for the first few keyframes and "
                f"then drifts, so the record index is not that dataset's frame "
                f"number. Refusing rather than seeding from depth belonging to "
                f"another frame. Files look like: "
                f"{[os.path.basename(f) for f in self.files[:3]]}. Override with "
                f"--gt_depth_map if you know the pairing is right.")
        self._report.append(
            f"mapping={self.strategy} (depth correlation {best_corr:.3f})")
        if self.strategy == "assoc":
            self._report.append(
                f"  timestamp association from {self._assoc['src']}; "
                f"worst rgb->depth gap {self._assoc['max_dt'] * 1e3:.1f} ms")

        # global scale: monocular SLAM fixes geometry only up to one factor
        self.scale = 1.0
        if self.align != "none":
            # measure unscaled, or per-frame alignment would divide the very
            # ratio we are trying to read and always report 1.0
            align, self.align = self.align, "global"
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
            self.align = align
            if not ratios:
                raise RuntimeError("no overlapping pixels to estimate the scale")
            self.scale = float(np.median(ratios))
            spread = float(np.std(ratios) / max(self.scale, 1e-9))
            self._report.append(
                f"scale={self.scale:.4f} metres per record unit "
                f"(spread {spread:.1%} over {len(ratios)} keyframes), "
                f"align={self.align}")
            if self.align == "per_frame":
                self._report.append(
                    "  each keyframe uses its own factor, so trajectory scale "
                    "drift does not misplace patches; the global figure above "
                    "is reported for reference only")
            self.scale_spread = spread
            if spread > max_scale_spread and self.align != "per_frame":
                # Spread is a sharper health check than correlation: a correct
                # pairing yields one global factor, a wrong one yields noise.
                # On freiburg1_room the correct association gives 2.5% and the
                # wrong positional mapping 7.5%, while their correlations differ
                # by less than 0.02.
                self._report.append(
                    f"  WARNING: scale varies {spread:.1%} across keyframes, "
                    f"above {max_scale_spread:.0%}. Either the depth is paired "
                    f"with the wrong keyframes or the trajectory drifts in "
                    f"scale; geometry from different keyframes will not line up.")
        return self

    def summary(self):
        head = f"[gt-depth] {len(self.files)} PNGs in {self.dir}"
        return "\n".join([head] + [f"[gt-depth]   {r}" for r in self._report])


def depth_dir_for(records_dir):
    """`<records_dir>` -> `<records_dir>_depth`, or None if absent."""
    d = os.path.normpath(records_dir).rstrip(os.sep) + "_depth"
    return d if os.path.isdir(d) else None
