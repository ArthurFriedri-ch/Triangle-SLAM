"""Per-keyframe stage timings and size counters.

The point of the counters is the slope, not the absolute number: "it dies around
four keyframes" is a statement about growth, and only a per-keyframe size series
can tell you which term is growing.

Timings are off unless MAPPER_TIME=1. CUDA syncs are separately opt-in via
MAPPER_TIME_CUDA=1, because synchronising around every block distorts the thing
being measured.
"""
import os
import time
from collections import OrderedDict
from contextlib import contextmanager

ENABLED = os.environ.get("MAPPER_TIME", "0") == "1"
_SYNC = os.environ.get("MAPPER_TIME_CUDA", "0") == "1"

_times = OrderedDict()   # stage name -> seconds accumulated this frame
_counts = OrderedDict()  # counter name -> scalar for this frame
_totals = OrderedDict()  # stage name -> seconds accumulated over the whole run
_frames = 0


def _sync():
    if not _SYNC:
        return
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


@contextmanager
def tic(name, cuda=False):
    """Accumulate wall time for `name` within the current frame."""
    if not ENABLED:
        yield
        return
    if cuda:
        _sync()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if cuda:
            _sync()
        dt = time.perf_counter() - t0
        _times[name] = _times.get(name, 0.0) + dt
        _totals[name] = _totals.get(name, 0.0) + dt


def count(name, value):
    """Record a size for this frame (number of triangles, points, ids, ...)."""
    if ENABLED:
        _counts[name] = value


def frame_end(kf_index, csv_path=None):
    """Close out a keyframe. Appends a CSV row and returns a one-line summary."""
    global _frames
    if not ENABLED:
        return ""
    _frames += 1
    total = sum(_times.values())

    if csv_path:
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        new = not os.path.exists(csv_path)
        cols = ["kf", "total"] + list(_times) + list(_counts)
        row = [kf_index, round(total, 6)]
        row += [round(v, 6) for v in _times.values()]
        row += list(_counts.values())
        with open(csv_path, "a") as fh:
            if new:
                fh.write(",".join(cols) + "\n")
            fh.write(",".join(str(v) for v in row) + "\n")

    hot = sorted(_times.items(), key=lambda kv: -kv[1])[:5]
    summary = "  ".join(f"{k}={v:.2f}s" for k, v in hot)
    sizes = "  ".join(f"{k}={v}" for k, v in _counts.items())
    _times.clear()
    _counts.clear()
    return f"[time] kf {kf_index}  total={total:.2f}s | {summary} | {sizes}"


def report():
    """Aggregate table over the whole run."""
    if not ENABLED or not _totals:
        return ""
    width = max(len(k) for k in _totals)
    lines = [f"=== stage totals over {_frames} keyframes ==="]
    for k, v in sorted(_totals.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {k:<{width}}  {v:8.2f}s  {v / max(_frames, 1):7.3f}s/kf")
    lines.append(f"  {'TOTAL':<{width}}  {sum(_totals.values()):8.2f}s")
    return "\n".join(lines)
