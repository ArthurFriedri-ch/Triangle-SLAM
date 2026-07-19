import torch

class Patch:
    """One keyframe's seeded geometry. Geometry only today."""
    def __init__(self, vertices, colors, faces, age):
        self.vertices = vertices    # (V,3) world xyz
        self.colors   = colors      # (V,3) 0..1
        self.faces    = faces       # (F,3) int index triplets
        self.age      = age         # keyframe index at seeding -> for age coloring

def seed_patch(kf, mask, step=4, max_depth_jump=None, max_cov=None):
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