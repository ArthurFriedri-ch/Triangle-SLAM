import argparse
import numpy as np
import polyscope as ps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", help="path to a patch .npz written by save_patch_npz")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    V = d["vertices"].astype(np.float64)
    F = d["faces"].astype(np.int64)

    ps.init()
    ps.set_up_dir("neg_y_up")   # tweak to match your camera convention
    m = ps.register_surface_mesh("patch", V, F, smooth_shade=False)
    m.set_transform(np.diag([50, 50, 50, 1]).astype(np.float64))

    if "colors" in d:
        m.add_color_quantity("rgb", np.clip(d["colors"], 0, 1), enabled=True)
    if "age_colors" in d:
        m.add_color_quantity("age", np.clip(d["age_colors"], 0, 1))

    # debug overlays (useful for the regular-grid artefacts)
    m.add_scalar_quantity("z (depth)", V[:, 2], cmap="viridis")

    valence = np.bincount(F.reshape(-1), minlength=V.shape[0]).astype(np.float64)
    m.add_scalar_quantity("valence", valence, cmap="coolwarm")

    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    m.add_scalar_quantity("tri area", area, defined_on="faces", cmap="reds")

    degen = ((F[:, 0] == F[:, 1]) | (F[:, 1] == F[:, 2]) |
             (F[:, 0] == F[:, 2])).astype(np.float64)
    m.add_scalar_quantity("degenerate", degen, defined_on="faces", cmap="reds")

    print(f"loaded {V.shape[0]} verts, {F.shape[0]} faces from {args.npz}")
    ps.show()   # interactive: scroll=zoom, drag=rotate, right-drag=pan


if __name__ == "__main__":
    main()