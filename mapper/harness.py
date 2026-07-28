# mapper/harness.py
import argparse
import itertools
import os

import torch

from scene.patch import seed_patch
import slam_interface
from scene.cameras import Camera
import numpy as np, math
from triangle_renderer import render, TriangleModel
from arguments import PipelineParams
from argparse import ArgumentParser
from utils.timing import tic, frame_end, report
from utils.gt_depth import GtDepthSource, depth_dir_for
import torchvision
import matplotlib.pyplot as cm

# build a real PipelineParams with defaults (render reads pipe.debug, pipe.convert_SHs_python)
_pipe = PipelineParams(ArgumentParser()).extract(
    ArgumentParser().parse_args([]))
_pipe.debug = True
_pipe.convert_SHs_python = False
_BG = torch.tensor([0., 0., 0.], device="cuda")  # black background

background = torch.tensor([0., 0., 0.], device="cuda")  # black bg

def _log(msg):
    """Print only when there is something to print (timing may be disabled)."""
    if msg:
        print(msg)


ALPHA_SEEN = 0.5  # novelty: below this accumulated alpha => unseen => seed
FIXED_SIGMA = 0.001  # static sigma for all triangles; no annealing today


class Keyframe:
    """Record converted into the mapper's own (torch) world."""

    def __init__(self, record, device, depth=None):
        self.index = record.index
        # `depth` overrides the record's SLAM depth (ground-truth PNGs). It has
        # already been divided into the record's scale by GtDepthSource.
        self.is_gt_depth = depth is not None
        self.depth = torch.from_numpy(record.depth if depth is None
                                      else depth).float().to(device)
        self.cov = torch.from_numpy(record.depth_cov).float().to(device)
        self.rgb = torch.from_numpy(record.rgb).float().to(device) / 255.0
        self.K = torch.from_numpy(record.intrinsics).float().to(device)
        pose = record.pose
        if record.pose_frame == slam_interface.POSE_FRAME_W2C:
            pose = np.linalg.inv(pose)  # want camera->world
        self.c2w = torch.from_numpy(pose).float().to(device)


class TriangleMapper:
    def __init__(self, device="cuda:0", debug=True, use_occlusion=True,
                 gt_depth=None):
        self.out_dir = "keyframe_debug/"
        self.device = device
        self.patches = []
        self.debug = debug                  # seam tile + age view; both debug-only
        self.use_occlusion = use_occlusion
        self.gt_depth = gt_depth            # GtDepthSource or None
        self._n_seen = 0
        self._model_cache = {}   # (colors_mode, n_patches) -> TriangleModel

    def is_empty(self):
        return len(self.patches) == 0

    def _build_model(self, colors_mode="rgb"):
        """Aggregate all patches into a TriangleModel for rendering.

        Cached on (mode, patch count): this used to run three times per
        keyframe, each time re-allocating every nn.Parameter and redoing RGB2SH
        over the whole map.
        """
        key = (colors_mode, len(self.patches))
        if key in self._model_cache:
            return self._model_cache[key]
        with tic("build_model"):
            tm = TriangleModel(sh_degree=0)
            verts, faces, colors, off = [], [], [], 0
            for p in self.patches:
                verts.append(p.vertices)
                faces.append(p.faces + off)
                off += len(p.vertices)
                colors.append(p.age_colors if colors_mode == "age" else p.colors)
            tm.populate_triangle_model(torch.cat(verts), torch.cat(faces),
                                       torch.cat(colors), FIXED_SIGMA)
            with torch.no_grad():
                tm.vertex_weight.fill_(20.0)
        # drop entries from earlier keyframes, keep both colour modes of this one
        self._model_cache = {k: v for k, v in self._model_cache.items()
                             if k[1] == len(self.patches)}
        self._model_cache[key] = tm
        return tm

    def _model_boundary_loops(self):
        """Global boundary rings, from each patch's cached rings plus its offset.

        Never re-derived from the global face array -- that was an O(3T) pass
        per keyframe over a mesh that grows linearly.
        """
        loops, off = [], 0
        for p in self.patches:
            for ring in p.boundary_loops():
                loops.append(ring + off)
            off += len(p.vertices)
        return loops

    def _render(self, kf, colors_mode="rgb"):
        tm = self._build_model(colors_mode=colors_mode)
        view = make_view(kf, kf.rgb.shape[1], kf.rgb.shape[0])  # W, H
        with tic("render", cuda=True):
            return render(view, tm, _pipe, _BG)

    def render_coverage(self, kf):
        """True accumulated alpha of the existing map, seen from kf."""
        pkg = self._render(kf, colors_mode="rgb")
        alpha = pkg["rend_alpha"].squeeze(0)  # (H, W), accumulated alpha
        return alpha

    def integrate(self, record):
        gt = None
        if self.gt_depth is not None:
            gt = self.gt_depth.depth_for(record, self._n_seen)
            if gt is None:
                print(f"kf {record.index}: no ground-truth depth found, "
                      f"falling back to the record's SLAM depth")
        self._n_seen += 1

        kf = Keyframe(record, self.device, depth=gt)
        # With ground-truth depth the covariance describes a depth map we are no
        # longer using, so validity is what bounds the region instead.
        invalid_mask = (kf.depth <= 0) if kf.is_gt_depth else None

        tm, loops, rendered_depth = None, None, None
        if self.patches:
            tm = self._build_model(colors_mode="rgb")
            loops = self._model_boundary_loops()
            if self.use_occlusion:
                pkg = self._render(kf, colors_mode="rgb")   # cache hit on tm
                rendered_depth = pkg["surf_depth"].squeeze(0)   # (H,W) camera depth
                self._check_depth_alignment(kf, rendered_depth, pkg["rend_alpha"])

        with tic("seed_patch"):
            patch, image, ids = seed_patch(kf, tm, model_loops=loops,
                                           rendered_depth=rendered_depth,
                                           invalid_mask=invalid_mask,
                                           debug=self.debug)
        if patch is None:
            print(f"kf {record.index}: no patch seeded, skipping")
            _log(frame_end(record.index, os.path.join(self.out_dir, "timings.csv")))
            return

        save_patch_npz(patch, os.path.join(self.out_dir, f"kf{kf.index:04d}.npz"), ids)

        patch.age = record.index
        color = age_to_color(record.index, device=self.device)  # (3,)
        patch.age_colors = color.unsqueeze(0).expand(patch.vertices.shape[0], 3).contiguous()  # (V,3)
        self.patches.append(patch)

        # self.optimize() ; self.anneal() ; self.densify()

        self.dump_views(kf, self.out_dir, image)
        n_weld = int((ids >= 0).sum()) if ids is not None else 0
        print(f"kf {record.index}: {len(patch.faces)} tris, {len(self.patches)} patches, "
              f"{n_weld} weldable verts")
        _log(frame_end(record.index, os.path.join(self.out_dir, "timings.csv")))

    def _check_depth_alignment(self, kf, rendered_depth, alpha):
        """One-shot sanity check that surf_depth is comparable to kf.depth.

        The occlusion test assumes both are camera-space metres. If the
        rasterizer's convention differs, that assumption fails silently and
        badly, so measure the disagreement once and say so.
        """
        if getattr(self, "_depth_checked", False):
            return
        with torch.no_grad():
            m = (alpha.squeeze(0) > 0.5) & (rendered_depth > 0) & (kf.depth > 0)
            if m.sum() < 100:
                return          # too little overlap to judge; try again next frame
            self._depth_checked = True
            diff = (rendered_depth[m] - kf.depth[m]).abs().median().item()
            scale = kf.depth[m].median().item()
            print(f"[depth-check] median |surf_depth - kf.depth| = {diff:.4f} m "
                  f"over {int(m.sum())} covered px (median depth {scale:.2f} m)")
            if diff > 0.25 * max(scale, 1e-6):
                print("[depth-check] WARNING: surf_depth does not look like "
                      "camera-space metres. Occlusion culling is probably "
                      "misfiring; pass rendered_depth=None to disable it.")

    def render_age_view(self, kf, path):
        """Deliverable 4: render the whole map age-colored from kf's viewpoint."""
        pkg = self._render(kf, colors_mode="age")
        torchvision.utils.save_image(pkg["render"], path)

    def dump_views(self, kf, out_dir, patch_debug=None):
        os.makedirs(out_dir, exist_ok=True)
        idx = kf.index

        pkg_rgb = self._render(kf, colors_mode="rgb")
        alpha = pkg_rgb["rend_alpha"].clamp(0, 1)          # (1,H,W)
        gt = kf.rgb.permute(2, 0, 1)                       # (3,H,W)

        tiles = [gt, pkg_rgb["render"].clamp(0, 1)]
        if self.debug:                                     # age view is debug-only
            tiles.append(self._render(kf, colors_mode="age")["render"].clamp(0, 1))
        tiles.append(alpha.repeat(3, 1, 1))

        if patch_debug is not None:
            # patch_debug = debug_render_pslg(valid, vertices, segments, holes, points)
            tiles.append(patch_debug.to(gt.device))

        panel = torch.cat(tiles, dim=2)                    # concat along width
        torchvision.utils.save_image(
            panel, os.path.join(out_dir, f"kf{idx:04d}_panel.png"))


def source_from_dir(d):
    return slam_interface.iter_records(d)


def source_from_socket(port):
    return slam_interface.ipc.RecordReceiver(port=port)


def age_to_color(age, max_age=50, device="cuda"):
    """Map a keyframe age (index) to an RGB color via a colormap."""
    t = float(age) / max(1, max_age)  # normalize to [0,1]
    t = min(max(t, 0.0), 1.0)
    rgba = cm.get_cmap("turbo")(t)  # returns (r,g,b,a) in [0,1]
    return torch.tensor(rgba[:3], dtype=torch.float32, device=device)  # drop alpha -> (3,)


def make_view(kf, W, H):
    w2c = np.linalg.inv(kf.c2w.cpu().numpy())
    R = w2c[:3, :3].T  # Camera expects R transposed (3DGS convention) — verify in cameras.py
    T = w2c[:3, 3]
    fx, fy = kf.K[0, 0].item(), kf.K[1, 1].item()
    FoVx = 2 * math.atan(W / (2 * fx))
    FoVy = 2 * math.atan(H / (2 * fy))
    return Camera(colmap_id=0, R=R, T=T, FoVx=FoVx, FoVy=FoVy,
                  image=torch.zeros(3, H, W),  # dummy; render doesn't need real pixels
                  gt_alpha_mask=None, image_name=f"kf{kf.index}", uid=kf.index)


def save_patch_npz(patch, path, ids=None):
    """Write a single patch to a self-contained .npz for the polyscope viewer."""
    def np_(x):
        return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

    data = {
        "vertices": np_(patch.vertices).astype(np.float64),
        "faces":    np_(patch.faces).astype(np.int64),
    }
    if ids is not None:
        data["weld_ids"] = np_(ids).astype(np.int64)
    if hasattr(patch, "colors"):
        data["colors"] = np.clip(np_(patch.colors), 0, 1).astype(np.float64)
    if hasattr(patch, "age_colors"):
        data["age_colors"] = np.clip(np_(patch.age_colors), 0, 1).astype(np.float64)
    if hasattr(patch, "age"):
        data["age"] = np.asarray(getattr(patch, "age"))

    np.savez_compressed(path, **data)
    print(f"[save_patch_npz] wrote {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--records_dir")
    p.add_argument("--port", type=int)
    p.add_argument("--max_kf", type=int, default=None)
    p.add_argument("--fast", action="store_true",
                   help="skip the seam debug tile and the age render")
    p.add_argument("--no_occlusion", action="store_true",
                   help="ignore rendered depth; treat all projected model area as covered")
    p.add_argument("--no_gt_depth", action="store_true",
                   help="use the record's SLAM depth even if <records_dir>_depth exists")
    p.add_argument("--gt_depth_dir", default=None,
                   help="override the ground-truth depth folder")
    p.add_argument("--gt_depth_scale", type=float, default=None,
                   help="PNG units per metre (default: 5000 TUM / 6553.5 Replica)")
    p.add_argument("--gt_depth_no_align", action="store_true",
                   help="do NOT rescale ground-truth depth into the pose frame. "
                        "The SLAM is monocular, so its poses carry an arbitrary "
                        "global scale; without alignment the geometry will not "
                        "match the camera baselines.")
    p.add_argument("--gt_depth_probe", type=int, default=8,
                   help="keyframes used to calibrate the mapping and scale")
    args = p.parse_args()

    gt = None
    if not args.no_gt_depth and args.records_dir:
        gt_dir = args.gt_depth_dir or depth_dir_for(args.records_dir)
        if gt_dir is None:
            print(f"[gt-depth] no {args.records_dir}_depth folder; "
                  f"using the record's SLAM depth")
        else:
            probe = list(itertools.islice(
                source_from_dir(args.records_dir), args.gt_depth_probe))
            gt = GtDepthSource(gt_dir, png_scale=args.gt_depth_scale,
                               align=not args.gt_depth_no_align).calibrate(probe)
            print(gt.summary())

    mapper = TriangleMapper(debug=not args.fast,
                            use_occlusion=not args.no_occlusion,
                            gt_depth=gt)
    source = source_from_socket(args.port) if args.port else source_from_dir(args.records_dir)

    for i, record in enumerate(source):
        if args.max_kf is not None and i >= args.max_kf:
            break
        mapper.integrate(record)

    print(f"Done. {len(mapper.patches)} patches from {len(mapper.patches)} keyframes.")
    _log(report())
