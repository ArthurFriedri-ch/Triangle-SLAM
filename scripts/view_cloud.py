# scripts/view_cloud.py
import argparse, numpy as np, slam_interface

def backproject(record):
    H, W = record.depth.shape
    fx, fy = record.intrinsics[0, 0], record.intrinsics[1, 1]
    cx, cy = record.intrinsics[0, 2], record.intrinsics[1, 2]
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z = record.depth
    valid = z > 0
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pts_cam = np.stack([x, y, z], -1)[valid]              # camera frame
    rgb = record.rgb[valid] / 255.0
    # to world: c2w @ [X,1]. If pose_frame is w2c, invert first.
    T = record.pose if record.pose_frame == slam_interface.POSE_FRAME_C2W \
        else np.linalg.inv(record.pose)
    pts_h = np.concatenate([pts_cam, np.ones((len(pts_cam), 1))], -1)
    pts_world = (T @ pts_h.T).T[:, :3]
    return pts_world, rgb

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("records_dir"); p.add_argument("--max_kf", type=int, default=5)
    args = p.parse_args()
    import open3d as o3d   # laptop-side, CPU is fine
    allpts, allrgb = [], []
    for i, r in enumerate(slam_interface.iter_records(args.records_dir)):
        if i >= args.max_kf: break
        pts, rgb = backproject(r); allpts.append(pts); allrgb.append(rgb)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np.concatenate(allpts))
    pc.colors = o3d.utility.Vector3dVector(np.concatenate(allrgb))
    o3d.visualization.draw_geometries([pc])