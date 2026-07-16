# scripts/view_records.py
import argparse, numpy as np, slam_interface
import matplotlib.pyplot as plt

def show(record):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(record.rgb); ax[0].set_title(f"RGB (kf {record.index})")
    d = record.depth.copy(); d[d <= 0] = np.nan
    im1 = ax[1].imshow(d, cmap="turbo", vmin=0.0, vmax=3.0)
    ax[1].set_title("depth (m)")
    fig.colorbar(im1, ax=ax[1], fraction=0.046)
    cov = record.depth_cov
    import matplotlib.colors as mcolors
    cpos = np.where(cov > 0, cov, np.nan)
    im2 = ax[2].imshow(cpos, cmap="magma", norm=mcolors.LogNorm(vmin=1e0, vmax=1e6)); ax[2].set_title("depth covariance")
    fig.colorbar(im2, ax=ax[2], fraction=0.046)
    print(f"kf {record.index}: depth {np.nanmin(d):.3f}..{np.nanmax(d):.3f} m | "
          f"cov {cov.min():.2e}..{cov.max():.2e} std {cov.std():.2e} | "
          f"pose_frame={record.pose_frame}")
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("records_dir")
    p.add_argument("--stride", type=int, default=1)
    args = p.parse_args()
    for i, r in enumerate(slam_interface.iter_records(args.records_dir)):
        if i % args.stride == 0:
            show(r)