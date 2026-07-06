# interface/slam_interface/record.py
import io as _io
from dataclasses import dataclass
from typing import Optional
import numpy as np

SCHEMA_VERSION = 1

# Stored in every record so each file is self-describing — kills the
# pose-frame / units ambiguity that silently misaligns downstream.
POSE_FRAME_C2W = "c2w"   # pose maps camera -> world
POSE_FRAME_W2C = "w2c"   # pose maps world  -> camera
DEPTH_UNITS_M  = "m"


@dataclass
class Record:
    index: int                          # frame index (from kf_idx_to_f_idx)
    pose: np.ndarray                    # (4,4) float64
    depth: np.ndarray                   # (H,W) float32, metric, 0 = invalid
    depth_cov: np.ndarray              # (H,W) float32, depth-space variance
    intrinsics: np.ndarray             # (3,3) float64, K at depth resolution
    rgb: np.ndarray                    # (H,W,3) uint8, 0..255
    pose_frame: str = POSE_FRAME_C2W
    depth_units: str = DEPTH_UNITS_M
    pose_cov: Optional[np.ndarray] = None   # (6,6) float64, optional
    gt_depth: Optional[np.ndarray] = None   # (H,W) float32, optional (Replica)
    schema_version: int = SCHEMA_VERSION

    def validate(self):
        H, W = self.depth.shape
        if self.pose.shape != (4, 4):
            raise ValueError(f"pose must be (4,4), got {self.pose.shape}")
        if self.intrinsics.shape != (3, 3):
            raise ValueError(f"intrinsics must be (3,3), got {self.intrinsics.shape}")
        if self.depth_cov.shape != (H, W):
            raise ValueError(f"depth_cov {self.depth_cov.shape} != depth {(H, W)}")
        if self.rgb.shape != (H, W, 3):
            raise ValueError(f"rgb must be {(H, W, 3)}, got {self.rgb.shape}")
        if self.pose_frame not in (POSE_FRAME_C2W, POSE_FRAME_W2C):
            raise ValueError(f"unknown pose_frame {self.pose_frame!r}")
        if not np.isfinite(self.pose).all():
            raise ValueError("pose contains non-finite values")
        # The load-bearing check: covariance must be finite and non-negative.
        if not np.isfinite(self.depth_cov).all():
            raise ValueError("depth_cov contains non-finite values")
        if (self.depth_cov < 0).any():
            raise ValueError("depth_cov contains negative variance")

    # ---- codec: the single (de)serialization used by BOTH io.py and ipc.py ----
    def _to_arrays(self):
        d = {
            "index": np.asarray(self.index, np.int64),
            "pose": np.asarray(self.pose, np.float64),
            "depth": np.asarray(self.depth, np.float32),
            "depth_cov": np.asarray(self.depth_cov, np.float32),
            "intrinsics": np.asarray(self.intrinsics, np.float64),
            "rgb": np.asarray(self.rgb, np.uint8),
            "pose_frame": np.asarray(self.pose_frame),      # <U dtype, pickle-free
            "depth_units": np.asarray(self.depth_units),
            "schema_version": np.asarray(self.schema_version, np.int64),
        }
        if self.pose_cov is not None:
            d["pose_cov"] = np.asarray(self.pose_cov, np.float64)
        if self.gt_depth is not None:
            d["gt_depth"] = np.asarray(self.gt_depth, np.float32)
        return d

    @classmethod
    def _from_arrays(cls, npz):
        f = set(npz.files)
        return cls(
            index=int(npz["index"]),
            pose=npz["pose"], depth=npz["depth"], depth_cov=npz["depth_cov"],
            intrinsics=npz["intrinsics"], rgb=npz["rgb"],
            pose_frame=str(npz["pose_frame"]), depth_units=str(npz["depth_units"]),
            schema_version=int(npz["schema_version"]),
            pose_cov=npz["pose_cov"] if "pose_cov" in f else None,
            gt_depth=npz["gt_depth"] if "gt_depth" in f else None,
        )

    def to_bytes(self) -> bytes:
        buf = _io.BytesIO()
        np.savez_compressed(buf, **self._to_arrays())
        return buf.getvalue()

    @classmethod
    def from_bytes(cls, b: bytes) -> "Record":
        with np.load(_io.BytesIO(b), allow_pickle=False) as npz:  # pickle-free = safe
            return cls._from_arrays(npz)