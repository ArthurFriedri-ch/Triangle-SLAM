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


def _depth_stats(rendered, target, covered, taus):
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
        from lpipsPyTorch import lpips as _lpips
        lpips_fn = _lpips

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
            if lpips_fn is not None:
                row["lpips"] = float(lpips_fn(img, gt, net_type="vgg"))
            if v.get("depth") is not None:
                d = _depth_stats(pkg["surf_depth"].squeeze(0),
                                 v["depth"].to(device), covered, taus)
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
        lp = [r["lpips"] for r in sel if "lpips" in r]
        if lp:
            g["lpips"] = mean(lp)
        dep = [r["depth"] for r in sel if "depth" in r]
        if dep:
            g["depth"] = {k: mean([d[k] for d in dep])
                          for k in dep[0] if k != "n_px"}
            g["depth"]["n_px"] = int(np.sum([d["n_px"] for d in dep]))
            if metres_per_unit:
                for k in ("l1", "rmse", "median"):
                    g["depth"][k + "_m"] = g["depth"][k] * metres_per_unit
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
        if metres_per_unit:
            a = agg.get("all", {}).get("depth", {})
            if "l1_m" in a:
                L.append(f"  in metres ({metres_per_unit:.3f} m/unit): "
                         f"L1 {a['l1_m']:.4f} m, RMSE {a['rmse_m']:.4f} m")
        L.append(f"  measured over {dep['n_px']:,} pixels")
    L.append("=" * 68)
    return "\n".join(L)


def save(rows, agg, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"per_view": rows, "aggregate": agg}, fh, indent=2)
    return path
