# mapper/harness.py
import argparse
import os

import torch

from scene.patch import seed_patch
import slam_interface
from scene.cameras import Camera
import numpy as np, math
from triangle_renderer import render, TriangleModel
from arguments import PipelineParams
from argparse import ArgumentParser
import torchvision
import matplotlib.pyplot as cm

# build a real PipelineParams with defaults (render reads pipe.debug, pipe.convert_SHs_python)
_pipe = PipelineParams(ArgumentParser()).extract(
    ArgumentParser().parse_args([]))
_pipe.debug = True
_pipe.convert_SHs_python = False
_BG = torch.tensor([0., 0., 0.], device="cuda")  # black background

background = torch.tensor([0., 0., 0.], device="cuda")  # black bg

ALPHA_SEEN = 0.5  # novelty: below this accumulated alpha => unseen => seed
FIXED_SIGMA = 0.001  # static sigma for all triangles; no annealing today


class Keyframe:
    """Record converted into the mapper's own (torch) world."""

    def __init__(self, record, device):
        self.index = record.index
        self.depth = torch.from_numpy(record.depth).float().to(device)
        self.cov = torch.from_numpy(record.depth_cov).float().to(device)
        self.rgb = torch.from_numpy(record.rgb).float().to(device) / 255.0
        self.K = torch.from_numpy(record.intrinsics).float().to(device)
        pose = record.pose
        if record.pose_frame == slam_interface.POSE_FRAME_W2C:
            pose = np.linalg.inv(pose)  # want camera->world
        self.c2w = torch.from_numpy(pose).float().to(device)


class TriangleMapper:
    def __init__(self, device="cuda:0"):
        self.out_dir = "keyframe_debug/"
        self.device = device
        self.patches = []

    def is_empty(self):
        return len(self.patches) == 0

    def _build_model(self, colors_mode="rgb"):
        """Aggregate all patches into a fresh TriangleModel for rendering."""
        tm = TriangleModel(sh_degree=0)
        verts, faces, colors, off = [], [], [], 0
        for p in self.patches:
            verts.append(p.vertices)
            faces.append(p.faces + off)
            off += len(p.vertices)
            colors.append(p.age_colors if colors_mode == "age" else p.colors)
        tm.populate_triangle_model(torch.cat(verts), torch.cat(faces), torch.cat(colors), FIXED_SIGMA)
        with torch.no_grad():
            tm.vertex_weight.fill_(20.0)
        return tm

    def _render(self, kf, colors_mode="rgb"):
        tm = self._build_model(colors_mode=colors_mode)
        view = make_view(kf, kf.rgb.shape[1], kf.rgb.shape[0])  # W, H
        return render(view, tm, _pipe, _BG)

    def render_coverage(self, kf):
        """True accumulated alpha of the existing map, seen from kf."""
        pkg = self._render(kf, colors_mode="rgb")
        alpha = pkg["rend_alpha"].squeeze(0)  # (H, W), accumulated alpha
        return alpha

    def integrate(self, record):
        kf = Keyframe(record, self.device)
        if self.patches:
            tm = self._build_model(colors_mode="rgb")
        else:
            tm = None
        patch, image, ids = seed_patch(kf, tm)
        save_patch_npz(patch, os.path.join(self.out_dir, f"kf{kf.index:04d}.npz"))

        patch.age = record.index
        color = age_to_color(record.index, device=self.device)  # (3,)
        patch.age_colors = color.unsqueeze(0).expand(patch.vertices.shape[0], 3).contiguous()  # (V,3)
        self.patches.append(patch)

        # self.optimize() ; self.anneal() ; self.densify()

        self.dump_views(kf, self.out_dir, image)
        print(f"kf {record.index}: {len(patch.faces)} tris, {len(self.patches)} patches")

    def render_age_view(self, kf, path):
        """Deliverable 4: render the whole map age-colored from kf's viewpoint."""
        pkg = self._render(kf, colors_mode="age")
        torchvision.utils.save_image(pkg["render"], path)

    def dump_views(self, kf, out_dir, patch_debug=None):
        os.makedirs(out_dir, exist_ok=True)
        idx = kf.index

        pkg_rgb = self._render(kf, colors_mode="rgb")
        pkg_age = self._render(kf, colors_mode="age")
        alpha = pkg_rgb["rend_alpha"].clamp(0, 1)          # (1,H,W)
        gt = kf.rgb.permute(2, 0, 1)                       # (3,H,W)

        tiles = [gt, pkg_rgb["render"].clamp(0, 1),
                 pkg_age["render"].clamp(0, 1),
                 alpha.repeat(3, 1, 1)]

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


def save_patch_npz(patch, path):
    """Write a single patch to a self-contained .npz for the polyscope viewer."""
    def np_(x):
        return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

    data = {
        "vertices": np_(patch.vertices).astype(np.float64),
        "faces":    np_(patch.faces).astype(np.int64),
    }
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
    args = p.parse_args()

    mapper = TriangleMapper()
    source = source_from_socket(args.port) if args.port else source_from_dir(args.records_dir)

    for i, record in enumerate(source):
        if args.max_kf is not None and i >= args.max_kf:
            break
        mapper.integrate(record)

    print(f"Done. {len(mapper.patches)} patches from {len(mapper.patches)} keyframes.")
