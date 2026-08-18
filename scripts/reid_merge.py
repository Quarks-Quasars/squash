"""
Merge fragmented track IDs (caused by camera cuts breaking ByteTrack/DeepSORT
association) into exactly 2 persistent player IDs, using color-appearance
matching anchored from two seed track IDs you pick by eye.

How to pick seeds:
    Open your annotated tracked_output.mp4, find a clean frame (e.g. frame 10)
    where both players are clearly visible and not swapped, and note the two
    track_id numbers drawn on them.

Run:
    python reid_merge.py --video match_clip.mp4 --tracks tracked_output_tracks.csv \
        --player1-seed-id 56 --player2-seed-id 89 --out merged_tracks.csv
"""

import argparse
import csv
import cv2
import numpy as np


def build_histogram(crops):
    """Average HSV color histogram (hue+saturation only, value/brightness
    dropped) across a set of image crops. Dropping V makes this fairly
    robust to lighting changes between camera angles, which matters a lot
    here since different cuts often have different exposure/lighting."""
    hists = []
    for crop in crops:
        if crop.size == 0:
            continue
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        hists.append(hist)
    if not hists:
        return None
    return np.mean(hists, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=r"C:\Users\light\datasets\runs\coordinates\track.mp4")
    parser.add_argument("--tracks", default=r"C:\Users\light\datasets\runs\coordinates\track_tracks.csv")
    parser.add_argument("--out", default=r"C:\Users\light\datasets\runs\coordinates\merged_tracks.csv")
    parser.add_argument("--player1", type=int, required=True)
    parser.add_argument("--player2", type=int, required=True)
    parser.add_argument("--min-frames", type=int, default=5,
                         help="drop fragments shorter than this many frames as noise")
    parser.add_argument("--samples-per-track", type=int, default=6,
                         help="how many frames to sample per fragment for its color signature")
    args = parser.parse_args()

    # 1. group rows by original (fragmented) track_id
    by_track = {}
    with open(args.tracks) as f:
        for row in csv.DictReader(f):
            tid = int(row["track_id"])
            frame = int(row["frame"])
            box = (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]))
            by_track.setdefault(tid, []).append((frame, *box))

    for tid in by_track:
        by_track[tid].sort(key=lambda r: r[0])

    dropped = [tid for tid, rows in by_track.items() if len(rows) < args.min_frames]
    for tid in dropped:
        del by_track[tid]
    if dropped:
        print(f"Dropped {len(dropped)} noise fragments (< {args.min_frames} frames): {dropped}")

    if args.player1 not in by_track or args.player2 not in by_track:
        raise RuntimeError("Seed track ID not found (or got dropped as noise - lower --min-frames?)")

    # 2. figure out which (frame, track_id) crops we need, then read the video ONCE
    #    sequentially rather than seeking per-crop, which is slow/unreliable on
    #    compressed video
    needed = {}  # frame_idx -> list of (track_id, box)
    for tid, rows in by_track.items():
        n = len(rows)
        step = max(1, n // args.samples_per_track)
        sampled = rows[::step][: args.samples_per_track]
        for frame, x1, y1, x2, y2 in sampled:
            needed.setdefault(frame, []).append((tid, (x1, y1, x2, y2)))

    max_frame = max(needed.keys())
    crops_by_track = {tid: [] for tid in by_track}

    cap = cv2.VideoCapture(args.video)
    frame_idx = 0
    while frame_idx <= max_frame:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx in needed:
            h, w = frame.shape[:2]
            for tid, (x1, y1, x2, y2) in needed[frame_idx]:
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(w, int(x2)), min(h, int(y2))
                crop = frame[y1:y2, x1:x2]
                crops_by_track[tid].append(crop)
        frame_idx += 1
    cap.release()

    # 3. build a color signature per fragment
    hist_by_track = {}
    for tid, crops in crops_by_track.items():
        h = build_histogram(crops)
        if h is not None:
            hist_by_track[tid] = h

    seed1_hist = hist_by_track[args.player1]
    seed2_hist = hist_by_track[args.player2]

    # 4. classify every fragment against the two seed signatures
    assignment = {args.player1: 1, args.player2: 2}
    print("\nFragment -> player assignment (correlation to seed1 / seed2):")
    for tid, hist in hist_by_track.items():
        if tid in assignment:
            continue
        sim1 = cv2.compareHist(hist, seed1_hist, cv2.HISTCMP_CORREL)
        sim2 = cv2.compareHist(hist, seed2_hist, cv2.HISTCMP_CORREL)
        chosen = 1 if sim1 >= sim2 else 2
        assignment[tid] = chosen
        flag = "  <- low confidence, check this one" if abs(sim1 - sim2) < 0.1 else ""
        print(f"  track {tid:>4} -> player {chosen}   (sim1={sim1:.3f}, sim2={sim2:.3f}){flag}")

    # 5. write merged output - same row format as tracks.csv, but track_id is
    #    now one of exactly {1, 2}
    out_rows = []
    for tid, rows in by_track.items():
        pid = assignment.get(tid)
        if pid is None:
            continue
        for frame, x1, y1, x2, y2 in rows:
            out_rows.append((frame, pid, x1, y1, x2, y2))

    out_rows.sort(key=lambda r: (r[0], r[1]))
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "track_id", "x1", "y1", "x2", "y2"])
        writer.writerows(out_rows)

    print(f"\nWrote merged 2-player track file to {args.out}")


if __name__ == "__main__":
    main()
