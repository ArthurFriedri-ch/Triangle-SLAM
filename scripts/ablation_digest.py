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
    hdr = (f"  {'run':<22}{'set':<9}{'cover':>7}{'PSNR*':>8}{'SSIM*':>8}"
           f"{'LPIPS*':>8}{'PSNR':>8}{'Depth L1':>12}{'tris':>8}{'wall':>8}")
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
            # Depth L1 is only in cm when a metric scale existed for that run;
            # otherwise it is in the tracker's own units. Printing both under one
            # header made office0 look 300x better than freiburg when it was not.
            if "l1_cm" in d:
                dep = f"{d['l1_cm']:9.2f} cm"
            elif "l1" in d:
                dep = f"{d['l1']:10.4f} u"
            else:
                dep = f"{'-':>12}"
            f4 = lambda k: (f"{g[k]:8.4f}" if g.get(k) is not None else f"{'-':>8}")
            f2 = lambda k: (f"{g[k]:8.2f}" if g.get(k) is not None else f"{'-':>8}")
            first = label == "heldout"
            print(f"  {r if first else '':<22}{label:<9}{100*g['coverage']:6.1f}%"
                  f"{f2('psnr_masked')}{f4('ssim_masked')}{f4('lpips_masked')}"
                  f"{g['psnr']:8.2f}{dep}"
                  f"{extra.get('tris', '-') if first else '':>8}"
                  f"{(extra.get('wall', '-') + 's') if first else '':>8}")
    print("=" * 100)
    print("  * scored over covered pixels only; plain PSNR is bounded by coverage.")
    print("  Depth L1: 'cm' where a metric scale existed, 'u' = tracker units")
    print("  (no ground-truth depth for that run) -- the two are NOT comparable.")
    print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
