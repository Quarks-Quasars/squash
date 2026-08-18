"""
Click known court reference points in a video frame to compute a
pixel -> real-world-court homography matrix.

Uses official WSF singles court dimensions (meters). Coordinate system:
    origin = back-left corner (where back wall meets left side wall)
    x = across the court width (0 -> 6.4)
    y = down the court length, back wall (y=0) -> front wall (y=9.75)

Run:
    python calibrate.py --video match_clip.mp4 --out homography.json

Click the points IN THIS ORDER when prompted (skip any not visible by
pressing 's', but you need at least 4 to compute a homography):
    1. Back-left corner
    2. Back-right corner
    3. Short line - left side wall intersection
    4. Short line - right side wall intersection
    5. T-point (half-court line meets short line)
    6. Front-left corner
    7. Front-right corner
"""

import argparse
import json
import cv2

# Real-world court coordinates in meters, in the same order as the click prompts
COURT_POINTS = {
    "Back-left corner":            (0.0, 0.0),
    "Back-right corner":           (6.4, 0.0),
    "Short line - left wall":      (0.0, 4.26),
    "Short line - right wall":     (6.4, 4.26),
    "T-point":                     (3.2, 4.26),
    "Front-left corner":           (0.0, 9.75),
    "Front-right corner":          (6.4, 9.75),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=r"C:\Users\light\datasets\media\videos\single_shot_squash.mp4")
    parser.add_argument("--out", default=r"C:\Users\light\datasets\runs\coordinates\homography.json")
    parser.add_argument("--frame", type=int, default=10, help="which frame to calibrate on")
    args = parser.parse_args()
 
    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Could not read frame from video")
 
    clicked_px = []
    clicked_world = []
    clicked_labels = []
    labels = list(COURT_POINTS.keys())
    idx = [0]
 
    display = frame.copy()
 
    def redraw():
        nonlocal display
        display = frame.copy()
        for (px, py), lbl in zip(clicked_px, clicked_labels):
            cv2.circle(display, (px, py), 5, (0, 255, 0), -1)
            cv2.putText(display, lbl, (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1)
        if idx[0] < len(labels):
            cv2.putText(display, f"Click: {labels[idx[0]]}  (press 's' to skip, 'q' to finish)",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("calibrate", display)
 
    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and idx[0] < len(labels):
            clicked_px.append((x, y))
            clicked_world.append(COURT_POINTS[labels[idx[0]]])
            clicked_labels.append(labels[idx[0]])
            idx[0] += 1
            redraw()
 
    cv2.namedWindow("calibrate")
    cv2.setMouseCallback("calibrate", on_click)
    redraw()
 
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord('s') and idx[0] < len(labels):
            idx[0] += 1
            redraw()
        elif key == ord('q') or idx[0] >= len(labels):
            break
 
    cv2.destroyAllWindows()
 
    if len(clicked_px) < 4:
        raise RuntimeError(f"Need at least 4 points, only got {len(clicked_px)}. Re-run and click more.")
 
    import numpy as np
    src = np.array(clicked_px, dtype=np.float32)
    dst = np.array(clicked_world, dtype=np.float32)
 
    H, mask = cv2.findHomography(src, dst, method=cv2.RANSAC if len(src) > 4 else 0)
 
    with open(args.out, "w") as f:
        json.dump({"homography": H.tolist(), "num_points": len(clicked_px)}, f, indent=2)
 
    print(f"Saved homography ({len(clicked_px)} points) to {args.out}")
 
 
if __name__ == "__main__":
    main()