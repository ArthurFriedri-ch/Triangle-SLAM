"""Evaluation for the incrementally built map.

Two axes, because a mapper can be good at one and bad at the other:

  photometric   render each view from its own pose and compare with the image
                the camera actually saw -- PSNR, SSIM, optionally LPIPS
  geometric     compare the rendered depth with the keyframe's depth, over the
                pixels where the map actually has surface. Reported alongside
                *coverage*, because depth error over a map that covers a tenth
                of the frame is not comparable with one that covers all of it.

Held-out views are the ones that matter. Every keyframe the optimiser trained
against is, by construction, a view the model has been fitted to; `--eval_holdout`
in the harness keeps a fraction of them out of the optimisation targets so there
is something honest to measure. Both sets are reported separately and neither is
labelled the headline number, because which one you want depends on whether you
are asking "did it fit" or "did it generalise".

Depths are in the record's own units -- the SLAM is monocular, so those are not
metres. Pass `metres_per_unit` (GtDepthSource.scale) to also get them in metres.
"""
import json
import os

import numpy as np
import torch

from utils.image_utils import psnr
from utils.loss_utils import ssim


def _depth_stats(rendered, target, covered, taus, mpu=None):
    """Depth agreement over the pixels the map actually covers."""
    valid = covered & (target > 0) & (rendered > 0) & torch.isfinite(rendered)
    n = int(valid.sum())
    if n == 0:
        return None
    err = (rendered[valid] - target[valid]).abs()
    rel = err / target[valid].clamp_min(1e-6)
    out = {
        "n_px": n,
        "l1": float(err.mean()),
        "rmse": float((err ** 2).mean().sqrt()),
        "median": float(err.median()),
        "rel": float(rel.mean()),
    }
    for t in taus:                      # fraction within a relative threshold
        out[f"delta_{t:g}"] = float((rel < t).float().mean())
    if mpu:
        # centimetres, the unit MonoGS and Point-SLAM report Depth L1 in. The
        # conversion is per view because monocular scale drifts along the
        # trajectory, so one global factor would be wrong at both ends.
        out["l1_cm"] = out["l1"] * mpu * 100.0
        out["rmse_cm"] = out["rmse"] * mpu * 100.0
        out["median_cm"] = out["median"] * mpu * 100.0
    return out


def evaluate(model, views, render_fn, alpha_thresh=0.5,
             taus=(0.01, 0.05, 0.10), use_lpips=False, device="cuda"):
    """Score `model` on `views`.

    `views` is a list of dicts with keys: name, camera, rgb (3,H,W in 0..1),
    depth (H,W) or None, heldout (bool). Tensors may live on the CPU; they are
    moved per view so a long sequence does not have to fit on the GPU at once.
    `render_fn(camera, model)` returns the render package.
    """
    lpips_fn = None
    if use_lpips:
        # lpipsPyTorch.lpips() constructs and loads a VGG on every call, which
        # over a whole sequence dominates the evaluation. Build it once.
        from lpipsPyTorch.modules.lpips import LPIPS
        _net = LPIPS("vgg", "0.1").to(device).eval()
        lpips_fn = lambda a, b: _net(a[None], b[None])

    rows = []
    for v in views:
        with torch.no_grad():
            pkg = render_fn(v["camera"], model)
            img = pkg["render"].clamp(0, 1)
            gt = v["rgb"].to(device)
            alpha = pkg["rend_alpha"].squeeze(0)
            covered = alpha > alpha_thresh

            row = {
                "name": v["name"],
                "heldout": bool(v["heldout"]),
                "psnr": float(psnr(img, gt).mean()),
                "ssim": float(ssim(img, gt)),
                "coverage": float(covered.float().mean()),
            }
            # Full-frame PSNR is dominated by how much of the frame is covered:
            # an uncovered pixel renders the background against a real image, so
            # 77% coverage caps PSNR near 13 dB however good the surface is. The
            # masked figures score only the pixels the map claims, which is what
            # separates "how much did it map" from "how well".
            m = covered.expand_as(img)
            if int(covered.sum()) > 0:
                mse = float(((img - gt)[m] ** 2).mean())
                row["psnr_masked"] = float("inf") if mse <= 0 else \
                    float(10.0 * np.log10(1.0 / mse))
                # composite the render over the ground truth outside the mask, so
                # SSIM and LPIPS see no synthetic edge at the coverage boundary
                blend = torch.where(m, img, gt)
                row["ssim_masked"] = float(ssim(blend, gt))
                if lpips_fn is not None:
                    row["lpips_masked"] = float(lpips_fn(blend, gt))
            if lpips_fn is not None:
                row["lpips"] = float(lpips_fn(img, gt))
            if v.get("depth") is not None:
                d = _depth_stats(pkg["surf_depth"].squeeze(0),
                                 v["depth"].to(device), covered, taus,
                                 mpu=v.get("metres_per_unit"))
                if d:
                    row["depth"] = d
        rows.append(row)
    return rows


def aggregate(rows, metres_per_unit=None):
    """Mean over views, split into held-out and trained-on."""
    def mean(vals):
        return float(np.mean(vals)) if vals else None

    out = {}
    for label, sel in (("heldout", [r for r in rows if r["heldout"]]),
                       ("train", [r for r in rows if not r["heldout"]]),
                       ("all", rows)):
        if not sel:
            continue
        g = {
            "n_views": len(sel),
            "psnr": mean([r["psnr"] for r in sel]),
            "ssim": mean([r["ssim"] for r in sel]),
            "coverage": mean([r["coverage"] for r in sel]),
        }
        for k in ("psnr_masked", "ssim_masked", "lpips", "lpips_masked"):
            vals = [r[k] for r in sel if k in r and np.isfinite(r[k])]
            if vals:
                g[k] = mean(vals)
        dep = [r["depth"] for r in sel if "depth" in r]
        if dep:
            g["depth"] = {k: mean([d[k] for d in dep])
                          for k in dep[0] if k != "n_px"}
            g["depth"]["n_px"] = int(np.sum([d["n_px"] for d in dep]))
            if metres_per_unit and "l1_cm" not in g["depth"]:
                for k in ("l1", "rmse", "median"):      # single global factor
                    g["depth"][k + "_cm"] = g["depth"][k] * metres_per_unit * 100.0
        out[label] = g
    return out


def format_report(agg, metres_per_unit=None):
    L = ["", "=" * 68, "  evaluation", "=" * 68]
    if not agg:
        return "\n".join(L + ["  no views to evaluate", "=" * 68])

    hdr = f"  {'':<10}{'views':>6}{'PSNR':>9}{'SSIM':>8}{'LPIPS':>8}{'coverage':>10}"
    L.append(hdr)
    for label in ("heldout", "train", "all"):
        g = agg.get(label)
        if not g:
            continue
        lp = f"{g['lpips']:8.4f}" if "lpips" in g else f"{'-':>8}"
        L.append(f"  {label:<10}{g['n_views']:6d}{g['psnr']:9.2f}{g['ssim']:8.4f}"
                 f"{lp}{100*g['coverage']:9.1f}%")

    dep = next((agg[k]["depth"] for k in ("heldout", "train", "all")
                if k in agg and "depth" in agg[k]), None)
    if dep:
        L += ["", "  depth, over covered pixels only "
                  "(record units; the SLAM is monocular)"]
        hdr2 = f"  {'':<10}{'L1':>10}{'RMSE':>10}{'median':>10}{'rel':>8}"
        taus = sorted(k for k in dep if k.startswith("delta_"))
        hdr2 += "".join(f"{t.replace('delta_', '<'):>8}" for t in taus)
        L.append(hdr2)
        for label in ("heldout", "train", "all"):
            g = agg.get(label)
            if not g or "depth" not in g:
                continue
            d = g["depth"]
            row = (f"  {label:<10}{d['l1']:10.4f}{d['rmse']:10.4f}"
                   f"{d['median']:10.4f}{d['rel']:8.3f}")
            row += "".join(f"{100*d[t]:7.1f}%" for t in taus)
            L.append(row)
        if any("l1_cm" in agg[k].get("depth", {}) for k in agg):
            L += ["", "  Depth L1 [cm] -- the unit MonoGS and Point-SLAM report"]
            for label in ("heldout", "train", "all"):
                d = agg.get(label, {}).get("depth", {})
                if "l1_cm" in d:
                    L.append(f"  {label:<10}{d['l1_cm']:10.2f}{d['rmse_cm']:10.2f}"
                             f"{d['median_cm']:10.2f}   (L1 / RMSE / median)")
            L.append("  NOTE: measured over covered pixels against the keyframe's own "
                     "depth.\n        Those works evaluate against a ground-truth mesh "
                     "over the full\n        frame, so treat this as indicative, not "
                     "a like-for-like number.")
        L.append(f"  measured over {dep['n_px']:,} pixels")
    L.append("=" * 68)
    return "\n".join(L)


def save(rows, agg, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"per_view": rows, "aggregate": agg}, fh, indent=2)
    return path
