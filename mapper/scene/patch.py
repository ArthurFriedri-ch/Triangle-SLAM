import cv2
import torch
import numpy as np
import triangle as tr   # pip install triangle
import torch.nn.functional as F
from collections import Counter



class Patch:
    """One keyframe's seeded geometry. Geometry only today."""
    def __init__(self, vertices, colors, faces, age):
        self.vertices = vertices    # (V,3) world xyz
        self.colors   = colors      # (V,3) 0..1
        self.faces    = faces       # (F,3) int index triplets
        self.age      = age         # keyframe index at seeding -> for age coloring

def seed_patch(kf, model, mode="cdt", **kw):
    if mode == "grid":
        return seed_patch_grid(kf, model, **kw)
    elif mode == "cdt":
        return seed_patch_cdt(kf, model, **kw)
    raise ValueError(mode)

def seed_patch_grid(kf, mask, step=64, max_depth_jump=0.1, max_cov=5000):
    """
    Triangulate a coarse regular grid over the region of `kf` selected by
    `mask` (novelty mask: valid depth AND not-yet-covered), back-project
    to world space using kf.depth + kf.K + kf.c2w, and sample per-vertex
    color from kf.rgb. Appends the resulting Patch to self.patches.

    Returns the Patch, or None if nothing survives filtering.
    """
    device = kf.depth.device
    H, W = kf.depth.shape

    if mask.dtype != torch.bool:
        mask = mask.bool()

    fx, fy = kf.K[0, 0], kf.K[1, 1]
    cx, cy = kf.K[0, 2], kf.K[1, 2]

    ys = torch.arange(0, H, step, device=device)
    xs = torch.arange(0, W, step, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")   # (rows, cols)
    rows, cols = gy.shape

    z = kf.depth[gy, gx]
    valid = mask[gy, gx] & (z > 0)

    if max_cov is not None:
        valid &= kf.cov[gy, gx] < max_cov

    # back-project to camera space, then to world space
    X = (gx - cx) * z / fx
    Y = (gy - cy) * z / fy
    cam_pts = torch.stack([X, Y, z, torch.ones_like(z)], dim=-1)  # (rows, cols, 4)
    world_pts = torch.einsum("ij,rcj->rci", kf.c2w, cam_pts)[..., :3]

    vertices = world_pts.reshape(-1, 3)
    colors = kf.rgb[gy, gx].reshape(-1, 3)

    idx = torch.arange(rows * cols, device=device).reshape(rows, cols)

    faces = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            if not (valid[r, c] and valid[r, c + 1] and valid[r + 1, c] and valid[r + 1, c + 1]):
                continue
            if max_depth_jump is not None:
                quad_z = torch.stack([z[r, c], z[r, c + 1], z[r + 1, c], z[r + 1, c + 1]])
                if (quad_z.max() - quad_z.min()) > max_depth_jump:
                    continue

            v00, v01 = idx[r, c], idx[r, c + 1]
            v10, v11 = idx[r + 1, c], idx[r + 1, c + 1]

            d1 = abs(z[r, c] - z[r + 1, c + 1])
            d2 = abs(z[r, c + 1] - z[r + 1, c])
            if d1 <= d2:
                faces.append((v00, v10, v11)); faces.append((v00, v11, v01))
            else:
                faces.append((v00, v10, v01)); faces.append((v01, v10, v11))

    if not faces:
        return None

    faces_t = torch.tensor(faces, dtype=torch.int64, device=device)

    # drop unreferenced grid vertices (masked-out / filtered by covariance or depth jump)
    used = torch.unique(faces_t)
    remap = torch.full((rows * cols,), -1, dtype=torch.int64, device=device)
    remap[used] = torch.arange(len(used), device=device)

    patch = Patch(
        vertices=vertices[used],
        faces=remap[faces_t],
        colors=colors[used],
        age=kf.index
    )
    return patch

def seed_patch_cdt(kf, model=None, max_corners=400, corner_quality=0.01,
                   nms_radius=4, k_harris=0.04, smooth_ksize=5,
                   min_dist=24, cov_max=1e5, debug=True):
    """
    Triangulate the region that is (inside the covariance contour) AND (not
    already covered by `model` as seen from this camera). The model's projected
    boundary enters as a hard constraint, so the new patch's edge matches the
    existing geometry vertex-for-vertex.

    Returns (patch, ids) or (patch, ids, tile). `ids[k] >= 0` means vertex k
    is model vertex ids[k] — that is the welding table.
    """
    device = kf.depth.device
    H, W = kf.depth.shape
    depth = kf.depth

    # --- 1. covariance loops (vector) ------------------------------------
    res = _cov_loops(kf.cov, cov_max=cov_max)
    if res is None:
        return (None, None, None) if debug else (None, None)
    loops, is_hole, _markers, region = res      # markers unused now

    # --- 2. model boundary, projected ------------------------------------
    if model is not None and len(model._triangle_indices):
        P_m, S_m, ids_m, tri_uv = project_boundary(model, kf)
    else:
        P_m = np.empty((0, 2))
        S_m = np.empty((0, 2), np.int64)
        ids_m = np.empty(0, np.int64)
        tri_uv = np.empty((0, 3, 2))

    # --- 3. merge into one segment soup ----------------------------------
    P, S, ids, off = [P_m], [S_m], [ids_m], len(P_m)
    for xy in loops:
        k = len(xy)
        idx = np.arange(off, off + k)
        P.append(xy)
        S.append(np.stack([idx, np.roll(idx, -1)], 1))
        ids.append(-np.ones(k, np.int64))
        off += k
    P = np.concatenate(P).astype(np.float64)
    S = np.concatenate(S).astype(np.int64)
    ids = np.concatenate(ids)

    P, S, ids, _pad = _pad_ids(P, S, ids)          # see note below
    P, S = split_crossings(P, S)
    P, S = split_at_vertices(P, S) # NEW: resolve T-junctions
    ids = np.concatenate([ids, -np.ones(len(P) - len(ids), np.int64)])

    P, S, ids = _densify(P, S, ids, max_len=min_dist * 1.5)
    P, S, ids = _dedupe(P, S, ids)
    if len(S) < 3:
        return (None, None, None) if debug else (None, None)

    # --- 4. interior seeds -----------------------------------------------
    step = max(int(min_dist), 1)
    gy, gx = np.mgrid[step // 2:H:step, step // 2:W:step]
    cand = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float64)

    ok = _inside_loops(cand, loops, is_hole) & ~_covered(cand, tri_uv)
    cand = cand[ok]
    cand = cand[_seg_clearance(cand, P, S) > min_dist]

    corners = _harris_seeds(depth, region, loops, is_hole, tri_uv, P, S,
                            max_corners, corner_quality, nms_radius,
                            k_harris, smooth_ksize, min_dist, device)

    seeds = [p for p in (corners, cand) if len(p)]
    seeds = np.concatenate(seeds, 0) if seeds else np.empty((0, 2))
    if len(seeds):
        seeds = seeds[_thin(seeds, min_dist * 0.6)]

    # --- 5. triangulate: constraints only, no hole markers ---------------
    nP = len(P)
    vertices = np.vstack([P, seeds]) if len(seeds) else P
    tri_in = {"vertices": np.ascontiguousarray(vertices, np.float64),
              "segments": np.ascontiguousarray(S, np.int32)}

    assert S.max() < len(vertices)
    assert len(np.unique(vertices.round(6), axis=0)) == len(vertices)

    # p = _validate_pslg(P, S)
    # print({k: (len(v) if hasattr(v, "__len__") else v) for k, v in p.items()})
    out = tr.triangulate(tri_in, "pq20a%dY" % (min_dist ** 2 * 2))
    V = out["vertices"]
    assert np.allclose(V[:nP], P), "Y inserted boundary points; ids misaligned"

    ids_full = np.concatenate([ids, -np.ones(len(V) - nP, np.int64)])

    # --- 6. classify: inside covariance, outside the model ---------------
    F_all = out["triangles"].astype(np.int64)
    cen = V[F_all].mean(1)
    keep = _inside_loops(cen, loops, is_hole) & ~_covered(cen, tri_uv)
    F_all = F_all[keep]
    if len(F_all) == 0:
        return (None, None, None) if debug else (None, None)

    # --- 7. compact ------------------------------------------------------
    used = np.unique(F_all)
    remap = -np.ones(len(V), np.int64)
    remap[used] = np.arange(len(used))
    faces_np = remap[F_all]
    V = V[used]
    ids_full = ids_full[used]

    # --- 8. back-project -------------------------------------------------
    verts2d = torch.from_numpy(V).float().to(device)
    u = verts2d[:, 0].round().long().clamp(0, W - 1)
    v = verts2d[:, 1].round().long().clamp(0, H - 1)
    z = depth[v, u]
    fx, fy, cx, cy = kf.K[0, 0], kf.K[1, 1], kf.K[0, 2], kf.K[1, 2]
    cam = torch.stack([(verts2d[:, 0] - cx) * z / fx,
                       (verts2d[:, 1] - cy) * z / fy,
                       z, torch.ones_like(z)], -1)
    world = torch.einsum("ij,nj->ni", kf.c2w, cam)[:, :3]

    if not (z > 0).all():
        print(f"[seed_patch] {(z <= 0).sum().item()} vertices with zero depth")

    patch = Patch(vertices=world,
                  faces=torch.from_numpy(faces_np).to(device),
                  colors=kf.rgb[v, u], age=kf.index)

    if not debug:
        return patch, ids_full
    tile = debug_render_weld(kf.cov > cov_max, loops, is_hole, tri_uv,
                            V, faces_np, ids_full, seeds)
    return patch, tile, ids_full

def _thin(pts, r):
    """Greedy: keep a point only if it's > r from all previously kept ones."""
    keep = np.zeros(len(pts), bool)
    kept = []
    for i, p in enumerate(pts):
        if not kept or np.min(np.linalg.norm(np.array(kept) - p, axis=1)) > r:
            keep[i] = True
            kept.append(p)
    return keep

def _rasterize(loops, is_hole, shape):
    """Region enclosed by the outer loop minus the hole loops."""
    reg = np.zeros(shape, np.uint8)
    for xy, h in zip(loops, is_hole):
        if not h:
            cv2.fillPoly(reg, [np.round(xy).astype(np.int32)], 1)
    for xy, h in zip(loops, is_hole):
        if h:
            cv2.fillPoly(reg, [np.round(xy).astype(np.int32)], 0)
    return reg


def _cov_loops(cov, cov_max=1e5, margin=2, smooth=5, close_r=3,
               n_outer=256, n_hole=32, max_iter=5):
    """
    Returns (loops, is_hole, markers, region) where `region` is guaranteed to
    contain no pixel with cov > cov_max. Loops are the ONLY constraints.
    """
    inv0 = (cov > cov_max).detach().cpu().numpy().astype(np.uint8)
    shape = inv0.shape
    pad = margin + smooth

    for _ in range(max_iter):
        inv = inv0
        if close_r:
            inv = cv2.morphologyEx(inv, cv2.MORPH_CLOSE,
                                   np.ones((2 * close_r + 1,) * 2, np.uint8))
        inv = cv2.dilate(inv, np.ones((2 * pad + 1,) * 2, np.uint8))

        n, lab, st, _ = cv2.connectedComponentsWithStats(1 - inv, 8)
        if n < 2:
            return None
        core = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)

        cnts, hier = cv2.findContours(core, cv2.RETR_CCOMP,
                                      cv2.CHAIN_APPROX_NONE)
        if hier is None:
            pad += smooth
            continue
        hier = hier[0]

        loops, is_hole, markers, ok = [], [], [], True
        for i, cnt in enumerate(cnts):
            hole = hier[i][3] != -1
            xy = _resample_smooth(cnt, n_hole if hole else n_outer, smooth)
            if xy is None:                 # loop too small to represent
                ok = hole is False         # dropping a hole is not allowed
                if not ok:
                    break
                continue
            loops.append(xy)
            is_hole.append(hole)
        if not ok or not loops or not any(not h for h in is_hole):
            pad += smooth
            continue

        region = _rasterize(loops, is_hole, shape)
        if (region & inv0).any():          # smoothing leaked over bad pixels
            pad += smooth
            continue

        for xy, h in zip(loops, is_hole):  # markers only after verification
            if not h:
                continue
            tmp = np.zeros(shape, np.uint8)
            cv2.fillPoly(tmp, [np.round(xy).astype(np.int32)], 1)
            dt = cv2.distanceTransform(tmp, cv2.DIST_L2, 3)
            y, x = np.unravel_index(np.argmax(dt), dt.shape)
            markers.append((float(x), float(y)))

        return loops, is_hole, (np.array(markers, np.float64)
                                if markers else None), region
    return None

def _resample_smooth(cnt, n_pts, smooth):
    cnt = cnt.reshape(-1, 2).astype(np.float64)
    if len(cnt) < 8:
        return None
    seg = np.linalg.norm(np.diff(cnt, axis=0, append=cnt[:1]), axis=1)
    d = np.r_[0.0, np.cumsum(seg)]
    t = np.linspace(0, d[-1], n_pts, endpoint=False)
    xy = np.stack([np.interp(t, d[:-1], cnt[:, 0]),
                   np.interp(t, d[:-1], cnt[:, 1])], 1)
    if smooth > 1:
        w = np.ones(smooth) / smooth
        p = np.r_[xy[-smooth:], xy, xy[:smooth]]
        xy = np.stack([np.convolve(p[:, i], w, "same")[smooth:-smooth]
                       for i in range(2)], 1)
    return xy

def boundary_edges(faces):
    f = np.asarray(faces)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], 0)
    e = np.sort(e, 1)
    cnt = Counter(map(tuple, e))
    return np.array([k for k, v in cnt.items() if v == 1], np.int64)

def project_boundary(model, kf, near=1e-3):
    """-> pts (P,2), segs (S,2) int, ids (P,) model vertex index or -1,
          uv_all (N,2), tri_uv for the coverage test."""
    w2c = torch.inverse(kf.c2w)
    cam = (w2c[:3, :3] @ model.vertices.T).T + w2c[:3, 3]
    cam = cam.detach().cpu().numpy().astype(np.float64)
    K = kf.K.detach().cpu().numpy()
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    def proj(c):
        return np.stack([c[:, 0] * fx / c[:, 2] + cx,
                         c[:, 1] * fy / c[:, 2] + cy], 1)

    be = boundary_edges(model._triangle_indices.cpu().numpy())
    za, zb = cam[be[:, 0], 2], cam[be[:, 1], 2]

    pts, ids, segs = [], [], []
    index = {}

    def add(xyz, vid):
        if vid >= 0 and vid in index:
            return index[vid]
        pts.append(proj(xyz[None])[0])
        ids.append(vid)
        if vid >= 0:
            index[vid] = len(pts) - 1
        return len(pts) - 1

    for (i, j), zi, zj in zip(be, za, zb):
        if zi <= near and zj <= near:
            continue
        if zi > near and zj > near:
            segs.append((add(cam[i], int(i)), add(cam[j], int(j))))
            continue
        # one endpoint behind: clip to z = near
        (k, zk), (l, zl) = ((i, zi), (j, zj)) if zi > near else ((j, zj), (i, zi))
        t = (near - zk) / (zl - zk)
        mid = cam[k] + t * (cam[l] - cam[k])
        segs.append((add(cam[k], int(k)), add(mid, -1)))

    # projected triangles, for coverage classification later
    infront = (cam[model._triangle_indices.cpu().numpy(), 2] > near).all(1)
    tri_uv = proj(cam)[model._triangle_indices.cpu().numpy()[infront]]

    return (np.array(pts), np.array(segs, np.int64),
            np.array(ids, np.int64), tri_uv)

def split_crossings(P, S, tol=1e-7):
    P, S = P.copy(), S.copy()
    A, B = P[S[:, 0]], P[S[:, 1]]
    cuts = [[] for _ in range(len(S))]
    new_pts = []

    for i in range(len(S)):
        a, b = A[i], B[i]
        r = b - a
        j = np.arange(i + 1, len(S))
        if not len(j):
            break
        c, s = A[j], B[j] - A[j]
        denom = r[0] * s[:, 1] - r[1] * s[:, 0]
        ok = np.abs(denom) > tol
        d = c - a
        t = np.where(ok, (d[:, 0] * s[:, 1] - d[:, 1] * s[:, 0]) / np.where(ok, denom, 1), -1)
        u = np.where(ok, (d[:, 0] * r[1] - d[:, 1] * r[0]) / np.where(ok, -denom, 1), -1)
        hit = ok & (t > tol) & (t < 1 - tol) & (u > tol) & (u < 1 - tol)
        for jj, tt, uu in zip(j[hit], t[hit], u[hit]):
            k = len(P) + len(new_pts)
            new_pts.append(a + tt * r)
            cuts[i].append((tt, k))
            cuts[jj].append((uu, k))

    if new_pts:
        P = np.vstack([P, np.array(new_pts)])

    out = []
    for i, cl in enumerate(cuts):
        if not cl:
            out.append(S[i])
            continue
        chain = [S[i, 0]] + [k for _, k in sorted(cl)] + [S[i, 1]]
        out += [(chain[m], chain[m + 1]) for m in range(len(chain) - 1)]
    return P, np.array(out, np.int64)

def _pip(pts, poly):
    """Vectorised even-odd point-in-polygon. pts (N,2), poly (M,2) -> (N,) bool"""
    y = pts[:, 1]
    x1, y1 = poly[:, 0], poly[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    dy = np.where(y2 != y1, y2 - y1, 1.0)
    straddle = (y1 > y[:, None]) != (y2 > y[:, None])
    xint = x1 + (y[:, None] - y1) * (x2 - x1) / dy
    return (straddle & (pts[:, 0:1] < xint)).sum(1) % 2 == 1


def _inside_loops(pts, loops, is_hole):
    """Inside the outer loops, outside every hole loop."""
    inside = np.zeros(len(pts), bool)
    for xy, h in zip(loops, is_hole):
        if not h:
            inside |= _pip(pts, xy)
    for xy, h in zip(loops, is_hole):
        if h:
            inside &= ~_pip(pts, xy)
    return inside


def _covered(pts, tri_uv, chunk=4096):
    """Point inside any projected model triangle (either winding)."""
    if len(tri_uv) == 0:
        return np.zeros(len(pts), bool)
    a, b, c = tri_uv[:, 0], tri_uv[:, 1], tri_uv[:, 2]
    e1, e2, e3 = b - a, c - b, a - c
    hit = np.zeros(len(pts), bool)
    for i in range(0, len(pts), chunk):
        p = pts[i:i + chunk][:, None, :]
        d1 = e1[:, 0] * (p[..., 1] - a[:, 1]) - e1[:, 1] * (p[..., 0] - a[:, 0])
        d2 = e2[:, 0] * (p[..., 1] - b[:, 1]) - e2[:, 1] * (p[..., 0] - b[:, 0])
        d3 = e3[:, 0] * (p[..., 1] - c[:, 1]) - e3[:, 1] * (p[..., 0] - c[:, 0])
        hit[i:i + chunk] = (((d1 >= 0) & (d2 >= 0) & (d3 >= 0)) |
                            ((d1 <= 0) & (d2 <= 0) & (d3 <= 0))).any(1)
    return hit


def _seg_clearance(pts, P, S, chunk=2048):
    """Min distance from each point to any segment. pts (N,2) -> (N,)"""
    A, B = P[S[:, 0]], P[S[:, 1]]
    AB = B - A
    L2 = (AB ** 2).sum(1)
    L2 = np.where(L2 > 1e-12, L2, 1.0)
    out = np.empty(len(pts))
    for i in range(0, len(pts), chunk):
        p = pts[i:i + chunk][:, None, :]
        t = (((p - A) * AB).sum(-1) / L2).clip(0.0, 1.0)
        proj = A + t[..., None] * AB
        out[i:i + chunk] = np.linalg.norm(p - proj, axis=-1).min(1)
    return out


def _densify(P, S, ids, max_len):
    """Split segments longer than max_len; inserted points get id -1."""
    P = list(P)
    ids = list(ids)
    out = []
    for i, j in S:
        a, b = np.asarray(P[i], float), np.asarray(P[j], float)
        n = int(np.linalg.norm(b - a) // max_len)
        if n < 1:
            out.append((i, j))
            continue
        chain = [i]
        for k in range(1, n + 1):
            P.append(a + (k / (n + 1)) * (b - a))
            ids.append(-1)
            chain.append(len(P) - 1)
        chain.append(j)
        out += [(chain[m], chain[m + 1]) for m in range(len(chain) - 1)]
    return np.array(P), np.array(out, np.int64), np.array(ids, np.int64)


def _dedupe(P, S, ids, grid=1e-4):
    """Collapse coincident vertices, preferring model-tagged ones."""
    order = np.argsort(-(ids >= 0).astype(np.int8), kind="stable")
    Pp, idsp = P[order], ids[order]
    back = np.empty(len(P), np.int64)
    back[order] = np.arange(len(P))

    q = np.round(Pp / grid).astype(np.int64)
    _, first, inv = np.unique(q, axis=0, return_index=True,
                              return_inverse=True)
    P2, ids2 = Pp[first], idsp[first]

    S2 = inv[back[S]]
    S2 = S2[S2[:, 0] != S2[:, 1]]
    S2 = np.unique(np.sort(S2, 1), axis=0)
    return P2, S2, ids2

def _pad_ids(P, S, ids):
    if len(ids) < len(P):
        ids = np.concatenate([ids, -np.ones(len(P) - len(ids), np.int64)])
    return P, S, ids, len(P)

def _harris_seeds(depth, region, loops, is_hole, tri_uv, P, S, max_corners,
                  corner_quality, nms_radius, k_harris, smooth_ksize,
                  min_dist, device):
    reg_t = torch.from_numpy(region).bool().to(device)
    d = depth.float().clone()
    d[~reg_t] = 0.0
    Ix, Iy = torch.zeros_like(d), torch.zeros_like(d)
    Ix[:, 1:-1] = (d[:, 2:] - d[:, :-2]) * 0.5
    Iy[1:-1, :] = (d[2:, :] - d[:-2, :]) * 0.5

    def bb(x, k):
        return F.avg_pool2d(x[None, None], k, stride=1, padding=k // 2)[0, 0]

    Sxx, Syy, Sxy = bb(Ix * Ix, smooth_ksize), bb(Iy * Iy, smooth_ksize), \
                    bb(Ix * Iy, smooth_ksize)
    R = (Sxx * Syy - Sxy * Sxy) - k_harris * (Sxx + Syy) ** 2
    R[~reg_t] = R.min()
    pooled = F.max_pool2d(R[None, None], 2 * nms_radius + 1,
                          stride=1, padding=nms_radius)[0, 0]
    peak = (R == pooled) & (R > corner_quality * R.max())
    ys, xs = torch.nonzero(peak, as_tuple=True)
    if ys.numel() > max_corners:
        top = torch.topk(R[ys, xs], max_corners).indices
        ys, xs = ys[top], xs[top]
    if ys.numel() == 0:
        return np.empty((0, 2))
    pts = torch.stack([xs.float(), ys.float()], 1).cpu().numpy().astype(np.float64)
    ok = _inside_loops(pts, loops, is_hole) & ~_covered(pts, tri_uv)
    pts = pts[ok]
    return pts[_seg_clearance(pts, P, S) > min_dist] if len(pts) else pts

def debug_render_weld(invalid, loops, is_hole, tri_uv, V, faces, ids, seeds):
    inv = invalid.detach().cpu().numpy().astype(np.uint8)
    H, W = inv.shape
    c = np.zeros((H, W, 3), np.uint8)
    c[inv > 0] = (70, 0, 60)

    for t in tri_uv:                                   # existing model, blue
        cv2.fillPoly(c, [np.round(t).astype(np.int32)], (25, 35, 70))
    for t in V[faces]:                                 # new mesh, grey edges
        cv2.polylines(c, [np.round(t).astype(np.int32)], True,
                      (150, 150, 150), 1, cv2.LINE_AA)
    for xy, h in zip(loops, is_hole):                  # covariance constraint
        cv2.polylines(c, [np.round(xy).astype(np.int32)], True,
                      (255, 165, 0) if h else (0, 255, 255), 2, cv2.LINE_AA)
    for p in V[ids >= 0]:                              # weldable, magenta
        cv2.circle(c, tuple(np.round(p).astype(int)), 3, (255, 0, 255), -1)
    for p in seeds:
        cv2.circle(c, tuple(np.round(p).astype(int)), 1, (0, 255, 0), -1)
    return torch.from_numpy(c).float().div(255.).permute(2, 0, 1)

def _validate_pslg(P, S, tol=1e-6):
    problems = {}

    # 1. zero-length / degenerate segments
    seg_len = np.linalg.norm(P[S[:, 0]] - P[S[:, 1]], axis=1)
    problems["zero_len"] = np.where(seg_len < tol)[0]

    # 2. duplicate segments (same endpoints, either order)
    ss = np.sort(S, axis=1)
    _, idx, cnt = np.unique(ss, axis=0, return_index=True, return_counts=True)
    problems["dup_seg"] = int((cnt > 1).sum())

    # 3. near-duplicate vertices at a Triangle-scale tolerance
    span = P.max(0) - P.min(0)
    eps = 1e-6 * np.linalg.norm(span)          # Triangle-like relative eps
    q = np.round(P / max(eps, 1e-9)).astype(np.int64)
    _, cnt = np.unique(q, axis=0, return_counts=True)
    problems["near_dup_verts"] = int((cnt > 1).sum())

    # 4. T-junctions: a vertex lying on the interior of a segment
    A, B = P[S[:, 0]], P[S[:, 1]]
    AB = B - A
    L2 = (AB ** 2).sum(1).clip(1e-12)
    tj = []
    for vi, p in enumerate(P):
        t = (((p - A) * AB).sum(1) / L2)
        proj = A + t[:, None] * AB
        d = np.linalg.norm(p - proj, axis=1)
        on = (d < 1e-4) & (t > 1e-4) & (t < 1 - 1e-4)
        on &= (S[:, 0] != vi) & (S[:, 1] != vi)   # not an endpoint of that seg
        if on.any():
            tj.append(vi)
    problems["t_junctions"] = np.array(tj)

    return problems

def split_at_vertices(P, S, on_tol=1e-4, t_tol=1e-6):
    """Split any segment that a vertex lands on the interior of (T-junctions)."""
    A, B = P[S[:, 0]], P[S[:, 1]]
    AB = B - A
    L2 = (AB ** 2).sum(1).clip(1e-12)
    cuts = [[] for _ in range(len(S))]
    for vi, p in enumerate(P):
        t = ((p - A) * AB).sum(1) / L2
        proj = A + t[:, None] * AB
        d = np.linalg.norm(p - proj, axis=1)
        on = (d < on_tol) & (t > t_tol) & (t < 1 - t_tol) & \
             (S[:, 0] != vi) & (S[:, 1] != vi)
        for si in np.where(on)[0]:
            cuts[si].append((t[si], vi))
    out = []
    for i, cl in enumerate(cuts):
        if not cl:
            out.append(tuple(S[i]))
            continue
        chain = [S[i, 0]] + [k for _, k in sorted(cl)] + [S[i, 1]]
        out += [(chain[m], chain[m + 1]) for m in range(len(chain) - 1)]
    return P, np.array(out, np.int64)