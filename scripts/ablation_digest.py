"""Collapse a sweep's per-run eval.json files into one comparison table.

Reads the JSON rather than scraping the logs, so the numbers are exactly the
ones evaluation computed. Falls back to noting a run that produced nothing,
since a crashed run should be visible rather than silently absent.
"""
import json
import os
import re
import sys


def load(run_dir):
    p = os.path.join(run_dir, "eval.json")
    if not os.path.isfile(p):
        return None
    with open(p) as fh:
        return json.load(fh).get("aggregate", {})


def scrape(log):
    """A few things the summary prints but evaluation does not record."""
    out = {}
    if not os.path.isfile(log):
        return out
    txt = open(log, errors="replace").read()
    for key, pat in (("tris", r"model\s+(\d+) triangles"),
                     ("mib", r"triangles, ([\d.]+) MiB"),
                     ("iters", r"optimisation\s+(\d+) iterations"),
                     ("closure", r"seam closure (\d+)%"),
                     ("wall", r"wall clock\s+([\d.]+) s"),
                     ("gpu", r"peak (\d+) MiB")):
        m = re.search(pat, txt)
        if m:
            out[key] = m.group(1)
    return out


def main(root):
    runs = sorted(d for d in os.listdir(root)
                  if os.path.isdir(os.path.join(root, d)))
    print()
    print("=" * 100)
    print(f"  ablation summary -- {root}")
    print("=" * 100)
    hdr = (f"  {'run':<22}{'set':<9}{'PSNR':>7}{'SSIM':>8}{'LPIPS':>8}"
           f"{'cover':>7}{'DepthL1cm':>11}{'tris':>8}{'iters':>8}{'wall':>8}")
    print(hdr)
    print("-" * 100)
    for r in runs:
        agg = load(os.path.join(root, r))
        extra = scrape(os.path.join(root, r + ".log"))
        if not agg:
            print(f"  {r:<22}{'NO RESULT -- check the log':<40}")
            continue
        for label in ("heldout", "all"):
            g = agg.get(label)
            if not g:
                continue
            d = g.get("depth", {})
            lp = f"{g['lpips']:8.4f}" if g.get("lpips") is not None else f"{'-':>8}"
            cm = f"{d['l1_cm']:11.2f}" if "l1_cm" in d else f"{d.get('l1', 0):11.4f}"
            first = label == "heldout"
            print(f"  {r if first else '':<22}{label:<9}{g['psnr']:7.2f}"
                  f"{g['ssim']:8.4f}{lp}{100*g['coverage']:6.1f}%{cm}"
                  f"{extra.get('tris', '-') if first else '':>8}"
                  f"{extra.get('iters', '-') if first else '':>8}"
                  f"{(extra.get('wall', '-') + 's') if first else '':>8}")
    print("=" * 100)
    print("  Depth L1 is in cm only where a metric scale was available; otherwise")
    print("  it is in record units. See the per-run logs for the caveat in full.")
    print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
