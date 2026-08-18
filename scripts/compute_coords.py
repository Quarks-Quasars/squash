"""
Convert a tracks CSV (from track.py) into real-world court coordinates
using a calibrated homography, then compute distance run per player.

Run:
    python compute_coords.py --tracks tracked_output_tracks.csv --homography homography.json --out player_coords.csv
"""

import argparse
import json
import csv
import numpy as np
import cv2


def load_homography(path):
    with open(path) as f:
        data = json.load(f)
    return np.array(data["homography"], dtype=np.float64)


def foot_point(x1, y1, x2, y2):
    """Bottom-center of the bounding box - standard proxy for a player's
    feet/ground contact point. Note: this is only a reasonable ground-plane
    estimate if the box is tight; loose boxes will bias distance estimates."""
    return ((x1 + x2) / 2.0, y2)


def smooth(coords, window=5):
    """Simple moving average to reduce detection jitter before summing
    distances - raw per-frame noise otherwise inflates distance-run a lot."""
    if len(coords) < window:
        return coords
    arr = np.array(coords)
    kernel = np.ones(window) / window
    smoothed = np.zeros_like(arr)
    for dim in range(arr.shape[1]):
        smoothed[:, dim] = np.convolve(arr[:, dim], kernel, mode="same")
    return smoothed.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", default=r"C:\Users\light\datasets\runs\coordinates\track_tracks.csv")
    parser.add_argument("--homography", default=r"C:\Users\light\datasets\runs\coordinates\homography.json")
    parser.add_argument("--out", default="player_coords.csv")
    parser.add_argument("--fps", type=float, default=25.0)
    args = parser.parse_args()

    H = load_homography(args.homography)

    # frame, track_id, x1, y1, x2, y2 -> group by track_id, keep frame order
    by_track = {}
    with open(args.tracks) as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = int(row["track_id"])
            frame = int(row["frame"])
            x1, y1, x2, y2 = (float(row["x1"]), float(row["y1"]),
                               float(row["x2"]), float(row["y2"]))
            by_track.setdefault(tid, []).append((frame, x1, y1, x2, y2))

    out_rows = []
    distances = {}

    for tid, rows in by_track.items():
        rows.sort(key=lambda r: r[0])
        px_points = [foot_point(x1, y1, x2, y2) for _, x1, y1, x2, y2 in rows]

        pts = np.array(px_points, dtype=np.float32).reshape(-1, 1, 2)
        world_pts = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
        world_pts = smooth(world_pts.tolist(), window=5)

        total_dist = 0.0
        for i in range(len(world_pts)):
            frame = rows[i][0]
            wx, wy = world_pts[i]
            out_rows.append((frame, tid, wx, wy))
            if i > 0:
                px, py = world_pts[i - 1]
                total_dist += float(np.hypot(wx - px, wy - py))

        distances[tid] = total_dist

    out_rows.sort(key=lambda r: (r[0], r[1]))
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "track_id", "court_x_m", "court_y_m"])
        writer.writerows(out_rows)

    print(f"Wrote real-world coordinates to {args.out}\n")
    print("Distance run per track:")
    for tid, dist in sorted(distances.items()):
        n_frames = len(by_track[tid])
        duration_s = n_frames / args.fps
        print(f"  Player {tid}: {dist:.1f} m over {duration_s:.1f}s "
              f"({n_frames} frames) -> avg speed {dist / max(duration_s, 1e-6):.2f} m/s")


if __name__ == "__main__":
    main()
