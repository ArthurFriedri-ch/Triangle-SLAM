# mapper/harness.py
import argparse
import os
import time

import torch

from scene.patch import (seed_patch, boundary_halfedges, OCCLUSION_REL,
                        OCCLUSION_MIN_AREA, MIN_REGION_FRAC)
import slam_interface
from scene.cameras import Camera
import numpy as np, math
from triangle_renderer import render, TriangleModel
from arguments import PipelineParams, OptimizationParams
from argparse import ArgumentParser
from utils.timing import tic, count, frame_end, report, totals
from utils.gt_depth import GtDepthSource, depth_dir_for
from utils.point_utils import depth_to_normal
from scene.global_mesh import GlobalMesh
import evaluate as ev
import torchvision
import matplotlib.pyplot as cm

# Defaults for the TS+ parameter groups. Both groups must be registered on the
# SAME parser and extracted from the namespace that parser produces:
# `ParamGroup.extract` copies whatever it finds in the *parsed* namespace, so
# building on one parser and parsing another returns an empty GroupParams. That
# went unnoticed for `_pipe` because the only two fields `render` reads are
# assigned by hand below; `_opt` has thirty-one and needs them all.
_defaults_parser = ArgumentParser()
_pipe_group = PipelineParams(_defaults_parser)
_opt_group = OptimizationParams(_defaults_parser)
_defaults = _defaults_parser.parse_args([])

_pipe = _pipe_group.extract(_defaults)
# Must stay False. The rasterizer's debug path is not a no-op: it deep-copies
# every argument to CPU on each forward *and* backward so it can dump a snapshot
# on exception, and its backward branch still unpacks five gradients from a
# kernel that returns four, so it raises before the real error it exists to
# catch. TS+ itself defaults this to False and only flips it at --debug_from.
# Harmless while nothing called backward; fatal once optimisation does.
_pipe.debug = False
_pipe.convert_SHs_python = False
# learning rates, lambda_dssim, ...; the few values that matter per-run are
# overridden from the command line below.
_opt = _opt_group.extract(_defaults)
_BG = torch.tensor([0., 0., 0.], device="cuda")  # black background

background = torch.tensor([0., 0., 0.], device="cuda")  # black bg

def _log(msg):
    """Print only when there is something to print (timing may be disabled)."""
    if msg:
        print(msg)


ALPHA_SEEN = 0.5  # novelty: below this accumulated alpha => unseen => seed
# Softness. Global for every triangle -- _sigma is a scalar all the way into
# CUDA. TS+ trains from 1.0 downwards; the mapper has been running at 0.001,
# which is a very hard edge by comparison. Set with --sigma.
FIXED_SIGMA = 0.001
# Initial per-vertex opacity in (0,1). Stored as a logit internally; the old
# hard-coded vertex_weight of 20.0 was that logit, i.e. opacity ~1.0.
INIT_OPACITY = 0.99
# Softness at the end of the anneal. TS+ trains down to this.
SIGMA_FINAL = 0.0001
AGE_MAX_PATCHES = 200


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
                 gt_depth=None, verify_mesh=False,
                 occlusion_rel=OCCLUSION_REL,
                 occlusion_min_area=OCCLUSION_MIN_AREA,
                 min_region_frac=MIN_REGION_FRAC,
                 sigma=FIXED_SIGMA, sigma_final=SIGMA_FINAL, sigma_anneal_s=10.0,
                 opacity=INIT_OPACITY, iters_per_second=0.0,
                 record_fps=30.0, opt_window=8, weight_sparsity=False,
                 age_max_patches=AGE_MAX_PATCHES, eval_holdout=0,
                 lambda_normal=0.0, out_dir="keyframe_debug/",
                 save_npz=True, save_panels=True):
        self.out_dir = out_dir
        self.save_npz = save_npz
        self.save_panels = save_panels
        self.device = device
        self.mesh = GlobalMesh(device=device)
        self.debug = debug                  # seam tile + age view; both debug-only
        self.use_occlusion = use_occlusion
        self.occlusion_rel = occlusion_rel
        self.occlusion_min_area = occlusion_min_area
        self.min_region_frac = min_region_frac
        self.sigma0 = sigma                 # initial softness
        self.sigma_final = sigma if sigma_final is None else sigma_final
        self.sigma_anneal_s = sigma_anneal_s   # tail seconds to anneal over
        self.opacity = opacity              # initial transparency
        self.iters_per_second = iters_per_second
        self.record_fps = record_fps
        self.opt_window = opt_window
        self.weight_sparsity = weight_sparsity
        self.age_max_patches = age_max_patches
        self.eval_holdout = eval_holdout   # keep every Nth keyframe out
        self.lambda_normal = lambda_normal
        self._eval_views = []              # kept on CPU; see evaluate()
        self._views = []                    # cameras with real pixels
        self._overview = None               # fixed video camera
        self._video_mode = "rgb"
        self._video_fps = 30
        self._video_dir = None
        self._video_n = 0
        self._t0 = None
        self._last_t = None
        self._t0_wall = time.perf_counter()
        self._tail_t = None   # seconds into the tail; None while seeding
        self.gt_depth = gt_depth            # GtDepthSource or None
        self.verify_mesh = verify_mesh
        self._n_seen = 0
        self._model = None       # THE model: persistent, optimised in place
        self._step = 0           # global optimisation step, drives the LR schedule
        self._age_cache = {}     # mesh version -> age SH coefficients

    @property
    def patches(self):
        return self.mesh.patches

    def is_empty(self):
        return self.mesh.is_empty()

    def _build_model(self, colors_mode="rgb"):
        """The live model -- one object, one optimiser, for the whole run.

        Rebuilding it per keyframe would discard the optimiser with it, so
        patches are appended into it instead (see `_grow_model`). The age view
        renders this same model with the colours swapped; see `_render_age`.
        """
        return self._model

    def _age_features(self):
        """Age colours as SH DC coefficients, shaped like `_features_dc`."""
        from utils.sh_utils import RGB2SH
        if self._age_cache.get("version") != self.mesh.version:
            pids = self.mesh.vertex_patch.astype(np.int64)
            uniq = np.unique(pids)
            lut = torch.stack([age_to_color(int(p), self.age_max_patches,
                                            device=self.device) for p in uniq])
            cols = lut[torch.from_numpy(np.searchsorted(uniq, pids)).to(self.device)]
            self._age_cache = {"version": self.mesh.version,
                               "dc": RGB2SH(cols)[:, None, :]}
        return self._age_cache["dc"]

    def _render_age(self, cam):
        """Render the live model recoloured by patch age.

        Only the colours are swapped. Rebuilding a separate model from the mesh
        instead gave the age view its own *geometry* -- stale, because the mesh
        is only synced at the start of the next keyframe -- and its own
        vertex_weight, reset to the initial --opacity rather than the optimised
        value. At --opacity 0.28 that rendered every triangle at 28% opacity,
        so the age tile came out soft with the background showing through while
        the rgb tile beside it looked fine.
        """
        tm = self._model
        if tm is None:
            return None
        dc = self._age_features()
        if dc.shape[0] != tm._features_dc.shape[0]:
            return None                       # mesh and model briefly disagree
        saved = tm._features_dc
        try:
            tm._features_dc = dc              # attribute swap; Adam holds its own ref
            with tic("render", cuda=True):
                return render(cam, tm, _pipe, _BG)
        finally:
            tm._features_dc = saved

    def _grow_model(self, v_before):
        """Fold everything welded since `v_before` into the live model."""
        with tic("grow_model"):
            if self._model is None:
                self._model = self.mesh.to_triangle_model(
                    colors_mode="rgb", sigma=self.sigma0, opacity=self.opacity)
                self._model.training_setup(
                    _opt, _opt.feature_lr, _opt.weight_lr,
                    _opt.lr_triangles_points_init)
            else:
                self.mesh.append_into_model(self._model, v_before,
                                            opacity=self.opacity)

    def _model_boundary_loops(self):
        """Global boundary rings of the welded mesh.

        Was the offset-concatenation of each patch's own rings, which is wrong
        once anything welds: a welded seam edge has two faces from two different
        patches, so it is interior to the map, yet each patch on its own still
        reports it as boundary. Projecting those would make the seam boolean
        subtract regions that are no longer frontier.
        """
        return self.mesh.boundary_loops()

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
        if self._model is not None:
            # optimisation moves the model's vertices; pull them back so the
            # weld ids and the edge-split lerps refer to where geometry
            # actually is, not where it was when the patch was seeded
            self.mesh.sync_from(self._model)
        seed_version = self.mesh.version     # ids are captured against this
        if not self.mesh.is_empty():
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
                                           min_region_frac=self.min_region_frac,
                                           occlusion_rel=self.occlusion_rel,
                                           occlusion_min_area=self.occlusion_min_area,
                                           debug=self.debug)
        if patch is None:
            print(f"kf {record.index}: no patch seeded, skipping")
            _log(frame_end(record.index, os.path.join(self.out_dir, "timings.csv")))
            self._advance(self._elapsed(record.index), record.index)
            return

        if self.save_npz:
            save_patch_npz(patch,
                           os.path.join(self.out_dir, f"kf{kf.index:04d}.npz"),
                           ids)

        patch.age = record.index
        b_before = self.mesh.boundary_edge_count()
        v_before = len(self.mesh.vertices)
        count("n_patch_boundary_edges",
              len(boundary_halfedges(patch.faces.detach().cpu().numpy())))
        rec = self.mesh.weld(patch, ids, kf_index=record.index,
                             seed_version=seed_version,
                             splits=getattr(patch, "edge_splits", None))
        # the new geometry joins the model the optimiser is already working on,
        # rather than triggering a rebuild that would discard the optimiser
        self._grow_model(v_before)
        if self.verify_mesh:
            problems = self.mesh.verify(full=True)
            if problems:
                print("[mesh] " + "; ".join(problems))

        # Every keyframe becomes an evaluation view. A held-out one is still
        # seeded and welded -- its geometry belongs in the map -- but is never
        # an optimisation target, so it measures generalisation rather than fit.
        rgb_chw = kf.rgb.permute(2, 0, 1)
        heldout = (self.eval_holdout > 0
                   and len(self.mesh.patches) % self.eval_holdout == 0
                   and len(self.mesh.patches) > 1)
        cam = make_view(kf, kf.rgb.shape[1], kf.rgb.shape[0], image=rgb_chw)
        if self.lambda_normal > 0:
            # TS+ takes normals from the dataset; keyframes have none, but they
            # do have depth, and the renderer's own depth_to_normal turns one
            # into the other. This is a smoothness prior rather than independent
            # supervision -- the normals come from the same depth the geometry
            # was seeded from -- but it does tie the rendered surface normal to
            # the observed surface rather than leaving it unconstrained.
            with torch.no_grad():
                cam.normal_map = depth_to_normal(
                    cam, kf.depth[None]).permute(2, 0, 1).contiguous()
        self._eval_views.append({
            "name": f"kf{record.index:04d}", "camera": cam,
            "rgb": rgb_chw.detach().cpu(), "depth": kf.depth.detach().cpu(),
            "metres_per_unit": getattr(self.gt_depth, "last_frame_scale", None),
            "heldout": heldout})
        if not heldout:
            self._views.append(cam)
            if len(self._views) > self.opt_window:
                self._views = self._views[-self.opt_window:]

        opt_loss = self._advance(self._elapsed(record.index), record.index)

        if self.save_panels:
            self.dump_views(kf, self.out_dir, image)
        n_weld = int((ids >= 0).sum()) if ids is not None else 0
        # each welded seam edge closes one boundary edge on each side, so the
        # boundary grows by less than the patch's own outline
        b_grew = self.mesh.boundary_edge_count() - b_before
        print(f"kf {record.index}: {rec.n_faces} tris, {len(self.mesh.patches)} patches, "
              f"{n_weld} weldable verts, +{rec.n_vertices} new verts, "
              f"boundary +{b_grew} | {self.mesh.stats()}"
              + (f" | opt loss {opt_loss:.4f}" if opt_loss is not None else ""))
        _log(frame_end(record.index, os.path.join(self.out_dir, "timings.csv")))

    def optimize(self, n_iters, kf_index):
        """TS+ photometric refinement over a window of recent keyframes.

        Lifted from `train.py`'s inner loop, minus the parts that have no
        meaning here: no SH degree ramp (degree 0), and no normal loss, because
        keyframes carry no normal map. Sigma and the opacity floor anneal on
        *record time*, not on a global iteration count, since the map is built
        incrementally and there is no fixed schedule length to anneal against.

        Geometry is optimised on the TriangleModel and pushed back into the mesh
        afterwards. The Adam state does not survive that round trip -- the model
        is rebuilt from the mesh on the next keyframe -- so momentum restarts
        each keyframe. Acceptable while the windows are short; the fix is for
        the model to persist and be appended to, which needs
        `cat_tensors_to_optimizer` on the weld path.
        """
        if n_iters <= 0 or not self._views:
            return None
        from utils.loss_utils import l1_loss, ssim

        tm = self._model
        if tm is None:
            return None
        views = self._views[-self.opt_window:]

        losses = []
        with tic("optimize", cuda=True):
            for _ in range(n_iters):
                # a GLOBAL step count: the schedule, and the 1000-step freeze on
                # vertex positions, run across the whole session. Feeding a
                # per-call counter kept the vertex learning rate pinned at zero,
                # so geometry never moved at all.
                self._step += 1
                tm.update_learning_rate(self._step)
                tm.set_sigma(self._sigma_at(kf_index))
                cam = views[np.random.randint(len(views))]

                pkg = render(cam, tm, _pipe, _BG)
                image = pkg["render"]
                gt = cam.original_image

                pixel = l1_loss(image, gt)
                loss = ((1.0 - _opt.lambda_dssim) * pixel
                        + _opt.lambda_dssim * (1.0 - ssim(image, gt)))
                if self.lambda_normal > 0 and getattr(cam, "normal_map", None) is not None:
                    n_err = (1.0 - (pkg["rend_normal"] * cam.normal_map).sum(0))
                    loss = loss + self.lambda_normal * n_err.mean()
                if self.weight_sparsity:
                    loss = loss + (tm.get_vertex_weight[tm._triangle_indices].mean()
                                   * _opt.lambda_weight)

                loss.backward()
                with torch.no_grad():
                    # running maxima the prune/densify heuristics read
                    sz = pkg["scaling"].detach()
                    tm.image_size = torch.maximum(tm.image_size, sz)
                    imp = pkg["max_blending"].detach()
                    tm.importance_score = torch.maximum(tm.importance_score, imp)
                    tm.step_optimizer()
                    tm.optimizer.zero_grad(set_to_none=True)
                losses.append(loss.item())

        count("n_opt_iters", n_iters)
        count("opt_step", self._step)
        return float(np.mean(losses)) if losses else None

    def start_video(self, kf, colors_mode="rgb", back=1.5, fov_scale=2.0, fps=30,
                    size=(1280, 720)):
        """Fix the overview camera on the first keyframe and begin recording."""
        W, H = size
        self._overview = make_overview_view(kf, W, H, back=back,
                                            fov_scale=fov_scale)
        self._video_mode = colors_mode
        self._video_fps = fps
        self._video_dir = os.path.join(self.out_dir, "video")
        os.makedirs(self._video_dir, exist_ok=True)
        for f in os.listdir(self._video_dir):          # start clean
            if f.endswith(".png"):
                os.remove(os.path.join(self._video_dir, f))
        self._video_n = 0

    def record_frame(self):
        """Render the map from the fixed overview camera into the next frame."""
        if self._overview is None or self._model is None:
            return          # nothing welded yet; the clock still runs
        if self._video_mode == "age":
            pkg = self._render_age(self._overview)
            if pkg is None:
                return
        else:
            with tic("video_render", cuda=True):
                pkg = render(self._overview, self._model, _pipe, _BG)
        torchvision.utils.save_image(
            pkg["render"].clamp(0, 1),
            os.path.join(self._video_dir, f"f{self._video_n:06d}.png"))
        self._video_n += 1

    def run_tail(self, seconds):
        """Keep optimising after the keyframes run out, recording as it goes.

        The map stops growing here -- there is nothing left to seed -- so this
        is purely refinement, and it is what shows whether the optimiser is
        still improving the surface once the geometry has settled.
        """
        if seconds <= 0 or self._model is None:
            return
        n_frames = max(int(round(seconds * self._video_fps)), 1)
        per_frame = max(int(round(self.iters_per_second / self._video_fps)), 0)
        anneal = (f", sigma {self.sigma0:g} -> {self.sigma_final:g} over "
                  f"{self.sigma_anneal_s:g}s" if self.sigma_anneal_s > 0 else "")
        print(f"[tail] {seconds:g}s, {n_frames} frames, {per_frame} iterations "
              f"each{anneal}")
        if self.sigma_anneal_s > seconds:
            print(f"[tail] WARNING: the anneal spans {self.sigma_anneal_s:g}s but "
                  f"the tail is only {seconds:g}s, so sigma will stop at "
                  f"{self._lerp_sigma(seconds / self.sigma_anneal_s):g} rather "
                  f"than reaching {self.sigma_final:g}. Raise --tail_seconds.")

        last = self.mesh.patches[-1].kf_index if self.mesh.patches else 0
        for i in range(n_frames):
            self._tail_t = i / max(self._video_fps, 1)   # drives _sigma_at
            if per_frame:
                self.optimize(per_frame, last)
            elif self._model is not None:
                self._model.set_sigma(self._sigma_at(last))   # anneal anyway
            self.record_frame()
        self._tail_t = seconds

    def _lerp_sigma(self, a):
        a = min(max(a, 0.0), 1.0)
        return self.sigma0 + (self.sigma_final - self.sigma0) * a


    def _elapsed(self, kf_index):
        """Record seconds since the previous keyframe."""
        t = self._record_seconds(kf_index)
        dt = (t - self._last_t) if self._last_t is not None \
            else 1.0 / max(self.record_fps, 1e-6)   # give the first frame a slice
        self._last_t = t
        return max(dt, 0.0)

    def _advance(self, dt, kf_index):
        """Carry the map forward over `dt` seconds of record time.

        The interval's iteration budget is split across the video frames that
        span it, rather than run as one burst at the keyframe. Refinement is
        continuous, and because the frame count is dt * video_fps the video
        plays back at record speed -- a keyframe after a long pause gets both
        more iterations and more frames, in proportion.
        """
        total = int(round(dt * self.iters_per_second))
        frames = max(int(round(dt * self._video_fps)), 1) if self._overview else 1
        losses = []
        for i in range(frames):
            n = total // frames + (1 if i < total % frames else 0)
            if n:
                got = self.optimize(n, kf_index)
                if got is not None:
                    losses.append(got)
            self.record_frame()
        return float(np.mean(losses)) if losses else None

    def evaluate(self, use_lpips=False, json_path=None):
        """Score the finished map on every keyframe it saw."""
        if self._model is None or not self._eval_views:
            return None
        mpu = getattr(self.gt_depth, "scale", None) if self.gt_depth else None
        if self.gt_depth is not None and self.gt_depth.align == "per_frame":
            mpu = None      # the factor varies per keyframe; one number would lie
        with tic("evaluate", cuda=True):
            rows = ev.evaluate(self._model, self._eval_views,
                               lambda cam, m: render(cam, m, _pipe, _BG),
                               use_lpips=use_lpips, device=self.device)
        agg = ev.aggregate(rows, metres_per_unit=mpu)
        if json_path:
            ev.save(rows, agg, json_path)
            print(f"[eval] per-view metrics written to {json_path}")
        return ev.format_report(agg, metres_per_unit=mpu)

    def summary(self):
        """End-of-run accounting: where the time went and what came out."""
        stage, sums, _ = totals()
        wall = time.perf_counter() - self._t0_wall
        g = lambda k: stage.get(k, 0.0)

        # `seed_patch` wraps the whole CDT, and the entries below it are nested
        # inside it -- listing them as siblings would double count
        cdt = g("seed_patch")
        cdt_parts = [("region contour", g("region_loops")),
                     ("project map outline", g("project_loops")),
                     ("seam boolean", g("seam_boolean")),
                     ("PSLG + densify", g("pslg")),
                     ("masks", g("masks")),
                     ("interior seeds", g("seeds")),
                     ("triangulate", g("triangulate")),
                     ("edge splits", g("edge_splits")),
                     ("back-project", g("backproject")),
                     ("debug tile", g("debug_render"))]
        opt = g("optimize")
        integ = g("weld") + g("grow_model") + g("boundary_rings")
        rend = g("render") + g("video_render") + g("to_triangle_model")
        acct = cdt + opt + integ + rend

        m, kfs = self.mesh, self._n_seen
        n_patch = len(m.patches)
        rec_s = (self._last_t or 0.0) + (self._tail_t or 0.0)
        own = sums.get("n_patch_boundary_edges", 0)

        L = ["", "=" * 68,
             f"  {kfs} keyframes -> {n_patch} patches"
             f"{f' ({kfs - n_patch} skipped)' if kfs > n_patch else ''}",
             "=" * 68,
             f"  wall clock            {wall:8.1f} s",
             f"    CDT seeding         {cdt:8.1f} s  {100*cdt/max(wall,1e-9):5.1f}%"
             f"   {cdt/max(n_patch,1):6.3f} s/patch"]
        for name, v in sorted(cdt_parts, key=lambda kv: -kv[1]):
            if v > 0.005:
                L.append(f"        {name:<20}{v:8.1f} s  {100*v/max(cdt,1e-9):5.1f}% of CDT")
        L += [f"    optimisation        {opt:8.1f} s  {100*opt/max(wall,1e-9):5.1f}%",
              f"    integration         {integ:8.1f} s  {100*integ/max(wall,1e-9):5.1f}%"
              f"   (weld, model grow, boundary rings)",
              f"    rendering           {rend:8.1f} s  {100*rend/max(wall,1e-9):5.1f}%",
              f"    unaccounted         {wall-acct:8.1f} s  "
              f"{100*(wall-acct)/max(wall,1e-9):5.1f}%   (I/O, depth decode, encode)",
              ""]

        rate = (f", {self._step/opt:.0f}/s wall" if opt > 0.01 else "")
        L += [f"  optimisation    {self._step} iterations{rate}, "
              f"{self._step/max(n_patch,1):.0f} per patch",
              f"                  sigma {self.sigma0:g} -> {self._sigma_at(0):g}, "
              f"window {self.opt_window} keyframes, geometry "
              f"{'unfrozen' if self._step >= 1000 else f'STILL FROZEN ({1000-self._step} steps short)'}",
              f"  map             {len(m.vertices)} vertices, {len(m.faces)} faces, "
              f"{m.boundary_edge_count()} boundary edges"]
        if own:
            L.append(f"                  seam closure {100*(1-m.boundary_edge_count()/own):.0f}% "
                     f"({own} edges if patches had stayed separate)")
        L += [f"  welding         {sums.get('n_weld_reused',0)} vertices reused, "
              f"{sums.get('n_weld_new',0)} added, "
              f"{m.n_splits_applied} edge splits ({m.n_splits_skipped} skipped)",
              f"                  {m.n_faces_rejected} faces refused "
              f"(degenerate, duplicate or non-manifold)"]
        tm = self._model
        if tm is not None:
            def nbytes(*ts):
                return sum(t.numel() * t.element_size() for t in ts
                           if torch.is_tensor(t))
            params = nbytes(tm.vertices, tm.vertex_weight,
                            tm._features_dc, tm._features_rest)
            buffers = nbytes(tm._triangle_indices, tm.image_size,
                             tm.importance_score, getattr(tm, "face_patch", None))
            adam = sum(nbytes(st.get("exp_avg"), st.get("exp_avg_sq"))
                       for st in tm.optimizer.state.values()) \
                if tm.optimizer is not None else 0
            L.append(f"  model           {tm._triangle_indices.shape[0]} triangles, "
                     f"{(params + buffers) / 2**20:.1f} MiB "
                     f"({params / 2**20:.1f} params + {buffers / 2**20:.1f} buffers)"
                     + (f", +{adam / 2**20:.1f} MiB Adam state" if adam else ""))
        if torch.cuda.is_available():
            L.append(f"  gpu             peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB "
                     f"allocated, {torch.cuda.max_memory_reserved() / 2**20:.0f} MiB reserved")

        if rec_s > 0:
            L.append(f"  throughput      {rec_s:.1f} s of record in {wall:.1f} s wall "
                     f"= {rec_s/max(wall,1e-9):.2f}x real time, "
                     f"{wall/max(kfs,1):.2f} s/keyframe")
        if self._video_n:
            L.append(f"  video           {self._video_n} frames at {self._video_fps} fps "
                     f"= {self._video_n/max(self._video_fps,1):.1f} s")
        L.append("=" * 68)
        return "\n".join(L)

    def _sigma_at(self, kf_index):
        """Softness: held at --sigma while keyframes arrive, annealed after.

        TS+ anneals sigma across its whole schedule, but that schedule has a
        known length. Here the map is still growing while keyframes come in, and
        every new patch wants the soft edges that let the optimiser move
        geometry. So softness is held until the keyframes run out, then taken
        down to --sigma_final over --sigma_anneal_seconds of the tail, which is
        what sharpens the surface at the end.
        """
        if self._tail_t is None or self.sigma_anneal_s <= 0:
            return self.sigma0
        a = min(max(self._tail_t / self.sigma_anneal_s, 0.0), 1.0)
        return self.sigma0 + (self.sigma_final - self.sigma0) * a

    def _record_seconds(self, kf_index):
        """Elapsed record time at this keyframe, in seconds.

        Uses the dataset's own timestamps when the ground-truth depth loader
        parsed a TUM rgb.txt, since those are exact and unevenly spaced;
        otherwise falls back on --record_fps against the frame index.
        """
        gt = self.gt_depth
        assoc = getattr(gt, "_assoc", None) if gt is not None else None
        if assoc is not None and 0 <= kf_index < len(assoc["rgb_ts"]):
            ts = assoc["rgb_ts"]
            if self._t0 is None:
                self._t0 = float(ts[0])
            return float(ts[kf_index]) - self._t0
        return kf_index / max(self.record_fps, 1e-6)

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
        pkg = self._render_age(make_view(kf, kf.rgb.shape[1], kf.rgb.shape[0]))
        if pkg is not None:
            torchvision.utils.save_image(pkg["render"], path)

    def dump_views(self, kf, out_dir, patch_debug=None):
        os.makedirs(out_dir, exist_ok=True)
        idx = kf.index

        pkg_rgb = self._render(kf, colors_mode="rgb")
        gt = kf.rgb.permute(2, 0, 1)                       # (3,H,W)

        # 2x2: ground truth | render / age | seam. Accumulated alpha is dropped;
        # it duplicated what the render and the seam tile already show.
        tiles = [gt, pkg_rgb["render"].clamp(0, 1)]
        if self.debug:                                     # age view is debug-only
            age = self._render_age(make_view(kf, kf.rgb.shape[1], kf.rgb.shape[0]))
            if age is not None:
                tiles.append(age["render"].clamp(0, 1))
        if patch_debug is not None:
            tiles.append(patch_debug.to(gt.device))
        while len(tiles) < 4:
            tiles.append(torch.zeros_like(gt))
        panel = torch.cat([torch.cat(tiles[0:2], dim=2),
                           torch.cat(tiles[2:4], dim=2)], dim=1)
        torchvision.utils.save_image(
            panel, os.path.join(out_dir, f"kf{idx:04d}_panel.png"))


def source_from_dir(d):
    return slam_interface.iter_records(d)


def source_from_socket(port):
    return slam_interface.ipc.RecordReceiver(port=port)


AGE_MAX_PATCHES = 200


def age_to_color(ordinal, max_patches=AGE_MAX_PATCHES, device="cuda"):
    """Map a patch's ordinal (0,1,2,...) to a colour along the turbo ramp.

    Takes the patch ordinal, not the keyframe index. Keyframe indices are frame
    numbers into the source sequence -- freiburg1_room reaches 536 over 99
    keyframes -- so normalising them against 50 saturated the ramp after about a
    dozen patches and everything afterwards came out the same colour.
    """
    t = float(ordinal) / max(1, max_patches)
    t = min(max(t, 0.0), 1.0)
    rgba = cm.get_cmap("turbo")(t)  # returns (r,g,b,a) in [0,1]
    return torch.tensor(rgba[:3], dtype=torch.float32, device=device)


def make_overview_view(kf, out_w, out_h, back=1.5, fov_scale=2.0):
    """A fixed wide-angle camera sitting behind the given keyframe.

    `back` is measured in multiples of that keyframe's median depth rather than
    in metres, because the SLAM is monocular and its units are arbitrary -- a
    fixed distance would be in the scene for one sequence and outside the
    building for another. `fov_scale` divides the focal length, so 2.0 is
    roughly twice the field of view.

    The output resolution is independent of the keyframes'. The focal length is
    scaled by the resolution ratio first, so raising the resolution adds pixels
    instead of silently widening the shot -- only `fov_scale` changes framing.
    """
    src_h, src_w = kf.depth.shape
    z = kf.depth[kf.depth > 0]
    med = float(z.median()) if z.numel() else 1.0

    c2w = kf.c2w.detach().cpu().numpy().copy()
    # camera looks down +Z in this convention, so -Z is behind it
    offset = np.array([0.0, 0.0, -back * med, 1.0])
    c2w[:3, 3] = (c2w @ offset)[:3]

    w2c = np.linalg.inv(c2w)
    R = w2c[:3, :3].T
    T = w2c[:3, 3]
    fx = kf.K[0, 0].item() * (out_w / src_w) / fov_scale
    fy = kf.K[1, 1].item() * (out_h / src_h) / fov_scale
    return Camera(colmap_id=0, R=R, T=T,
                  FoVx=2 * math.atan(out_w / (2 * fx)),
                  FoVy=2 * math.atan(out_h / (2 * fy)),
                  image=torch.zeros(3, out_h, out_w), gt_alpha_mask=None,
                  image_name="overview", uid=-1)


def encode_video(frame_dir, out_path, fps):
    """Encode the PNGs in `frame_dir` into a video, if a backend is available."""
    frames = sorted(f for f in os.listdir(frame_dir) if f.endswith(".png"))
    if not frames:
        print("[video] no frames to encode")
        return None
    try:
        import imageio.v2 as imageio
        with imageio.get_writer(out_path, fps=fps) as w:
            for f in frames:
                w.append_data(imageio.imread(os.path.join(frame_dir, f)))
        print(f"[video] wrote {out_path} ({len(frames)} frames at {fps} fps)")
        return out_path
    except Exception as e:
        import shutil
        import subprocess
        if shutil.which("ffmpeg"):
            cmd = ["ffmpeg", "-y", "-framerate", str(fps),
                   "-i", os.path.join(frame_dir, "f%06d.png"),
                   "-pix_fmt", "yuv420p", "-vf",
                   "pad=ceil(iw/2)*2:ceil(ih/2)*2", out_path]
            subprocess.run(cmd, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"[video] wrote {out_path} ({len(frames)} frames at {fps} fps)")
            return out_path
        print(f"[video] {len(frames)} frames in {frame_dir}; no encoder available "
              f"({e}). Encode with:\n"
              f"  ffmpeg -framerate {fps} -i {frame_dir}/f%06d.png "
              f"-pix_fmt yuv420p {out_path}")
        return None


def make_view(kf, W, H, image=None):
    w2c = np.linalg.inv(kf.c2w.cpu().numpy())
    R = w2c[:3, :3].T  # Camera expects R transposed (3DGS convention) — verify in cameras.py
    T = w2c[:3, 3]
    fx, fy = kf.K[0, 0].item(), kf.K[1, 1].item()
    FoVx = 2 * math.atan(W / (2 * fx))
    FoVy = 2 * math.atan(H / (2 * fy))
    # `image` becomes Camera.original_image, which is the photometric target the
    # optimiser trains against. Rendering alone does not need it, so it stays
    # optional and defaults to the old dummy.
    if image is None:
        image = torch.zeros(3, H, W)
    return Camera(colmap_id=0, R=R, T=T, FoVx=FoVx, FoVy=FoVy,
                  image=image,
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
    p.add_argument("--out_dir", default="keyframe_debug/",
                   help="where panels, patch npz files, video frames and the "
                        "timing CSV go. Give each run of a sweep its own, or "
                        "they overwrite each other")
    p.add_argument("--no_panels", action="store_true",
                   help="skip the per-keyframe debug panel PNGs. With --fast "
                        "this leaves only the metrics, which is what an "
                        "evaluation run actually needs")
    p.add_argument("--no_patch_npz", action="store_true",
                   help="skip the per-keyframe .npz dumps. Nothing in-tree "
                        "reads them -- they exist for the polyscope viewer -- "
                        "and they dominate the output size of a long run")
    p.add_argument("--evaluate", action="store_true",
                   help="score the finished map on every keyframe and print a report")
    p.add_argument("--eval_holdout", type=int, default=0,
                   help="keep every Nth keyframe out of the optimisation targets "
                        "so it can be scored as a held-out view. Its geometry is "
                        "still seeded and welded; only the photometric loss skips "
                        "it. 0 means evaluate on trained-on views only")
    p.add_argument("--eval_lpips", action="store_true",
                   help="also compute LPIPS (slow, downloads VGG weights)")
    p.add_argument("--eval_json", default=None,
                   help="default: <out_dir>/eval.json")
    p.add_argument("--age_max_patches", type=int, default=AGE_MAX_PATCHES,
                   help="patches the age colour ramp spans before it "
                        "saturates. Set it near the number of keyframes you "
                        f"expect or the run stays one hue; default {AGE_MAX_PATCHES}")
    p.add_argument("--video", action="store_true",
                   help="record the map from a fixed wide-angle camera behind "
                        "the first keyframe and encode it to a video")
    p.add_argument("--video_mode", default="rgb", choices=("rgb", "age"),
                   help="colour the video by texture or by patch age")
    p.add_argument("--video_back", type=float, default=1.5,
                   help="how far behind the first keyframe to put the camera, "
                        "in multiples of that frame's median depth")
    p.add_argument("--video_fov_scale", type=float, default=2.0,
                   help="divides the focal length; 2.0 is roughly twice the "
                        "field of view")
    p.add_argument("--video_width", type=int, default=1920)
    p.add_argument("--video_height", type=int, default=1080)
    p.add_argument("--video_fps", type=int, default=30,
                   help="frame rate of the output video")
    p.add_argument("--tail_seconds", type=float, default=10.0,
                   help="keep optimising after the keyframes run out, recording "
                        "if --video. This is also when sigma anneals, so it should\n"
                        "be at least --sigma_anneal_seconds")
    p.add_argument("--video_out", default=None,
                   help="default: <out_dir>/map.mp4")
    p.add_argument("--sigma", type=float, default=FIXED_SIGMA,
                   help=f"initial softness for every triangle; default {FIXED_SIGMA} "
                        "(TS+ trains from 1.0)")
    p.add_argument("--sigma_final", type=float, default=SIGMA_FINAL,
                   help=f"softness at the end of the anneal; default {SIGMA_FINAL}")
    p.add_argument("--sigma_anneal_seconds", type=float, default=10.0,
                   help="seconds of the TAIL over which sigma goes from --sigma "
                        "to --sigma_final. Softness is held while keyframes are "
                        "still arriving, since new patches need soft edges for "
                        "the optimiser to move them; 0 disables. Needs "
                        "--tail_seconds at least this long to complete")
    p.add_argument("--opacity", type=float, default=INIT_OPACITY,
                   help=f"initial per-vertex opacity in (0,1); default {INIT_OPACITY}")
    p.add_argument("--iters_per_second", type=float, default=0.0,
                   help="optimisation iterations per second of RECORD time; "
                        "0 disables optimisation")
    p.add_argument("--record_fps", type=float, default=30.0,
                   help="frame rate used to turn frame indices into record "
                        "seconds when the dataset carries no timestamps")
    p.add_argument("--opt_window", type=int, default=8,
                   help="how many recent keyframes optimisation trains against")
    p.add_argument("--lambda_normal", type=float, default=0.0,
                   help="weight on a normal-consistency term. The target normals "
                        "are derived from the keyframe's own depth, since "
                        "keyframes carry no normal map; TS+ uses 0.001 with "
                        "dataset normals")
    p.add_argument("--weight_sparsity", action="store_true",
                   help="add the TS+ vertex-weight sparsity term to the loss")
    p.add_argument("--min_region_frac", type=float, default=MIN_REGION_FRAC,
                   help="discard isolated seed regions covering less than this "
                        "fraction of the frame from the camera. Note a genuine "
                        "new region is only a fraction of a percent once the map "
                        f"covers most of the view; default {MIN_REGION_FRAC}")
    p.add_argument("--occlusion_tol", type=float, default=OCCLUSION_REL,
                   help="how much further the map must be than the measured "
                        "depth, as a FRACTION of that depth, before its area is "
                        "re-opened for seeding. Raise to reduce doubling; "
                        f"default {OCCLUSION_REL}")
    p.add_argument("--occlusion_min_area", type=int, default=OCCLUSION_MIN_AREA,
                   help="smallest re-opened region worth believing, in pixels; "
                        f"default {OCCLUSION_MIN_AREA}")
    p.add_argument("--verify_mesh", action="store_true",
                   help="check mesh invariants after every weld (slow)")
    p.add_argument("--no_gt_depth", action="store_true",
                   help="use the record's SLAM depth even if <records_dir>_depth exists")
    p.add_argument("--gt_depth_dir", default=None,
                   help="override the ground-truth depth folder")
    p.add_argument("--gt_depth_scale", type=float, default=None,
                   help="PNG units per metre (default: 5000 TUM / 6553.5 Replica)")
    p.add_argument("--gt_depth_align", default="per_frame",
                   choices=("per_frame", "global", "none"),
                   help="rescale ground-truth depth into the pose frame. The "
                        "SLAM is monocular, so its poses carry an arbitrary "
                        "scale that also drifts along the trajectory; "
                        "'none' will not match the camera baselines at all and "
                        "'global' will drift away from them.")
    p.add_argument("--gt_depth_map", default=None,
                   help="force the index->file mapping, e.g. 'assoc', "
                        "'stem:depth{i:06d}', 'index', 'ordinal' "
                        "(default: auto-detect)")
    p.add_argument("--gt_depth_probe", type=int, default=8,
                   help="keyframes used to calibrate the mapping and scale")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.eval_json is None:
        args.eval_json = os.path.join(args.out_dir, "eval.json")
    if args.video_out is None:
        args.video_out = os.path.join(args.out_dir, "map.mp4")

    gt = None
    if not args.no_gt_depth and args.records_dir:
        gt_dir = args.gt_depth_dir or depth_dir_for(args.records_dir)
        if gt_dir is None:
            print(f"[gt-depth] no {args.records_dir}_depth folder; "
                  f"using the record's SLAM depth")
        else:
            # Spread the probe across the whole sequence. Sampling only the
            # first N cannot tell apart mappings that agree early and diverge
            # later: on freiburg1_room the rgb and depth streams stay in step
            # until rgb row 240, then differ for 1014 of 1362 rows, and the
            # records run to index 536.
            paths = slam_interface.list_record_paths(args.records_dir)
            n = min(args.gt_depth_probe, len(paths))
            pick = np.linspace(0, len(paths) - 1, n).round().astype(int)
            probe = [slam_interface.read(paths[i]) for i in sorted(set(pick))]
            gt = GtDepthSource(gt_dir, png_scale=args.gt_depth_scale,
                               align=args.gt_depth_align,
                               force_strategy=args.gt_depth_map).calibrate(probe)
            print(gt.summary())

    mapper = TriangleMapper(debug=not args.fast,
                            use_occlusion=not args.no_occlusion,
                            gt_depth=gt, verify_mesh=args.verify_mesh,
                            occlusion_rel=args.occlusion_tol,
                            occlusion_min_area=args.occlusion_min_area,
                            min_region_frac=args.min_region_frac,
                            sigma=args.sigma, sigma_final=args.sigma_final,
                            sigma_anneal_s=args.sigma_anneal_seconds,
                            opacity=args.opacity,
                            iters_per_second=args.iters_per_second,
                            record_fps=args.record_fps,
                            opt_window=args.opt_window,
                            weight_sparsity=args.weight_sparsity,
                            age_max_patches=args.age_max_patches,
                            eval_holdout=args.eval_holdout,
                            lambda_normal=args.lambda_normal,
                            out_dir=args.out_dir,
                            save_npz=not args.no_patch_npz,
                            save_panels=not args.no_panels)
    source = source_from_socket(args.port) if args.port else source_from_dir(args.records_dir)

    for i, record in enumerate(source):
        if args.max_kf is not None and i >= args.max_kf:
            break
        if args.video and i == 0:
            # fix the camera on the first keyframe, before anything is seeded,
            # so the whole map appears within a stationary frame
            mapper.start_video(Keyframe(record, mapper.device),
                               colors_mode=args.video_mode,
                               back=args.video_back,
                               fov_scale=args.video_fov_scale,
                               fps=args.video_fps,
                               size=(args.video_width, args.video_height))
        mapper.integrate(record)

    mapper.run_tail(args.tail_seconds)
    if args.video:
        encode_video(mapper._video_dir, args.video_out, args.video_fps)

    print(mapper.summary())
    if args.evaluate:
        _log(mapper.evaluate(use_lpips=args.eval_lpips,
                             json_path=args.eval_json))
    _log(report())
