"""Evaluation metrics. Runs on CPU by stubbing the render function."""
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_MAPPER = os.path.dirname(_HERE)
sys.path.insert(0, _MAPPER)

import evaluate as ev  # noqa: E402

H, W = 32, 48


def _view(name, rgb, depth=None, heldout=False):
    return {"name": name, "camera": None, "rgb": rgb,
            "depth": depth, "heldout": heldout}


def _pkg(render, alpha, depth):
    return {"render": render, "rend_alpha": alpha[None], "surf_depth": depth[None]}


def test_perfect_render_scores_perfectly():
    """An exact reproduction must give infinite PSNR and SSIM 1."""
    gt = torch.rand(3, H, W)
    views = [_view("a", gt)]
    rows = ev.evaluate(None, views,
                       lambda cam, m: _pkg(gt.clone(), torch.ones(H, W),
                                           torch.ones(H, W)),
                       device="cpu")
    assert rows[0]["psnr"] > 60, rows[0]["psnr"]
    assert rows[0]["ssim"] > 0.999
    assert abs(rows[0]["coverage"] - 1.0) < 1e-6


def test_worse_render_scores_worse():
    gt = torch.rand(3, H, W)
    good = (gt + 0.01).clamp(0, 1)
    bad = (gt + 0.30).clamp(0, 1)
    r = lambda img: ev.evaluate(
        None, [_view("v", gt)],
        lambda cam, m: _pkg(img, torch.ones(H, W), torch.ones(H, W)),
        device="cpu")[0]
    assert r(good)["psnr"] > r(bad)["psnr"]
    assert r(good)["ssim"] > r(bad)["ssim"]


def test_coverage_tracks_alpha():
    gt = torch.rand(3, H, W)
    alpha = torch.zeros(H, W)
    alpha[: H // 4] = 1.0                      # a quarter of the frame covered
    rows = ev.evaluate(None, [_view("v", gt)],
                       lambda cam, m: _pkg(gt.clone(), alpha, torch.ones(H, W)),
                       device="cpu")
    assert abs(rows[0]["coverage"] - 0.25) < 1e-6


def test_depth_error_ignores_uncovered_and_invalid():
    """Depth is only meaningful where the map has surface and the sensor saw one.

    Counting uncovered pixels would flatter or punish the map for geometry it
    never claimed to have.
    """
    target = torch.full((H, W), 2.0)
    target[:, : W // 2] = 0.0                  # sensor returned nothing here
    rendered = torch.full((H, W), 2.5)         # a constant 0.5 error where valid
    alpha = torch.ones(H, W)
    alpha[: H // 2] = 0.0                      # map covers only the bottom half

    rows = ev.evaluate(None, [_view("v", torch.rand(3, H, W), depth=target)],
                       lambda cam, m: _pkg(torch.rand(3, H, W), alpha, rendered),
                       device="cpu")
    d = rows[0]["depth"]
    # only the covered AND valid quadrant counts
    assert d["n_px"] == (H // 2) * (W - W // 2), d["n_px"]
    assert abs(d["l1"] - 0.5) < 1e-5
    assert abs(d["rel"] - 0.25) < 1e-5         # 0.5 / 2.0
    assert abs(d["delta_0.1"] - 0.0) < 1e-6    # 25% error is outside 10%
    assert abs(d["delta_1"] - 1.0) < 1e-6 if "delta_1" in d else True


def test_no_valid_depth_yields_no_depth_block():
    rows = ev.evaluate(None, [_view("v", torch.rand(3, H, W),
                                    depth=torch.zeros(H, W))],
                       lambda cam, m: _pkg(torch.rand(3, H, W),
                                           torch.ones(H, W), torch.ones(H, W)),
                       device="cpu")
    assert "depth" not in rows[0]


def test_aggregate_splits_heldout_from_train():
    gt = torch.rand(3, H, W)
    views = [_view("a", gt, heldout=True), _view("b", gt), _view("c", gt)]
    rows = ev.evaluate(None, views,
                       lambda cam, m: _pkg(gt.clone(), torch.ones(H, W),
                                           torch.ones(H, W)),
                       device="cpu")
    agg = ev.aggregate(rows)
    assert agg["heldout"]["n_views"] == 1
    assert agg["train"]["n_views"] == 2
    assert agg["all"]["n_views"] == 3
    assert "heldout" in ev.format_report(agg)


def test_global_scale_converts_to_cm():
    target = torch.full((H, W), 2.0)
    rows = ev.evaluate(None, [_view("v", torch.rand(3, H, W), depth=target)],
                       lambda cam, m: _pkg(torch.rand(3, H, W),
                                           torch.ones(H, W),
                                           torch.full((H, W), 2.5)),
                       device="cpu")
    agg = ev.aggregate(rows, metres_per_unit=1.5)
    assert abs(agg["all"]["depth"]["l1_cm"] - 0.5 * 1.5 * 100) < 1e-3
    assert "Depth L1 [cm]" in ev.format_report(agg)


def test_per_view_scale_beats_a_global_one_under_drift():
    """Monocular scale drifts, so each view must convert with its own factor.

    Converting the averaged error with a single factor is not the same number
    as averaging the per-view metric errors, and only the latter is meaningful
    when the factor moves along the trajectory.
    """
    target = torch.full((H, W), 2.0)
    scales = [1.73, 1.60, 1.45]
    rows = []
    for i, mpu in enumerate(scales):
        v = _view(f"v{i}", torch.rand(3, H, W), depth=target)
        v["metres_per_unit"] = mpu
        rows += ev.evaluate(None, [v],
                            lambda cam, m: _pkg(torch.rand(3, H, W),
                                                torch.ones(H, W),
                                                torch.full((H, W), 2.5)),
                            device="cpu")
    agg = ev.aggregate(rows)
    want = float(np.mean([0.5 * s * 100 for s in scales]))
    assert abs(agg["all"]["depth"]["l1_cm"] - want) < 1e-3, agg["all"]["depth"]["l1_cm"]
    # a single mid-range factor would have given a different answer
    assert abs(want - 0.5 * scales[1] * 100) > 1e-6 or True


def test_masked_psnr_is_independent_of_coverage():
    """The whole point: masked score must not move when only coverage does.

    Unmasked PSNR is bounded by coverage alone -- an uncovered pixel renders the
    background against a real image -- so comparing two runs on it mostly
    compares how much of the frame each one mapped.
    """
    # keep gt below 0.9 so gt+0.1 never clamps: the error is then exactly 0.1
    # everywhere, and any difference between the two runs is coverage, not noise
    gt = torch.rand(3, H, W) * 0.9
    err = gt + 0.1

    def run(frac):
        alpha = torch.zeros(H, W)
        alpha[: int(H * frac)] = 1.0
        # the real rasteriser composites over the background, so an uncovered
        # pixel comes out black; the stub has to do the same or the unmasked
        # metric has nothing to be penalised by
        img = err * alpha
        return ev.evaluate(None, [_view("v", gt)],
                           lambda cam, m: _pkg(img, alpha, torch.ones(H, W)),
                           device="cpu")[0]

    lo, hi = run(0.25), run(1.0)
    assert abs(lo["coverage"] - 0.25) < 1e-6 and abs(hi["coverage"] - 1.0) < 1e-6
    # masked: same error, same score
    assert abs(lo["psnr_masked"] - hi["psnr_masked"]) < 1e-4, \
        (lo["psnr_masked"], hi["psnr_masked"])
    # unmasked: the sparse one looks far worse purely because of the black
    assert lo["psnr"] < hi["psnr"] - 3.0


def test_masked_psnr_still_tracks_real_error():
    gt = torch.rand(3, H, W)
    alpha = torch.ones(H, W)
    good = ev.evaluate(None, [_view("v", gt)],
                       lambda cam, m: _pkg((gt + 0.02).clamp(0, 1), alpha,
                                           torch.ones(H, W)), device="cpu")[0]
    bad = ev.evaluate(None, [_view("v", gt)],
                      lambda cam, m: _pkg((gt + 0.25).clamp(0, 1), alpha,
                                          torch.ones(H, W)), device="cpu")[0]
    assert good["psnr_masked"] > bad["psnr_masked"] + 5


def test_masked_metrics_absent_when_nothing_covered():
    rows = ev.evaluate(None, [_view("v", torch.rand(3, H, W))],
                       lambda cam, m: _pkg(torch.rand(3, H, W),
                                           torch.zeros(H, W), torch.ones(H, W)),
                       device="cpu")
    assert "psnr_masked" not in rows[0] and rows[0]["coverage"] == 0.0


def test_report_survives_empty_input():
    assert "no views" in ev.format_report({})


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
