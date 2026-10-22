"""
Track the squash ball across a video using your fine-tuned YOLO26n ball detector.
Supports two tracker backends via --tracker: bytetrack | kalman | both

Why not DeepSORT: DeepSORT's whole value-add over ByteTrack is its appearance
embedder (used to re-identify an object after it's lost, by what it *looks*
like). A squash ball has basically zero distinguishing appearance - it's one
small, textureless, motion-blurred circle - so you're paying DeepSORT's
embedder cost (the thing that made your player tracker slow) for a feature
that can't help you here. ByteTrack skips the embedder entirely and is
motion/IoU based, so it should already be much faster on the same GPU.

The second tracker (kalman) is a from-scratch single-object tracker built
specifically for how a squash ball actually moves: constant-velocity-between-
bounces, with sudden hard direction changes off walls/floor that a normal
Kalman filter (tuned for smooth motion, like ByteTrack's internal one) tends
to lag behind or lose entirely. It uses:
  - high process noise -> the filter trusts its own velocity prediction less,
    so it adapts fast right after a bounce instead of "coasting" in the old
    direction for several frames
  - a search gate that GROWS the longer the ball has been unseen (occlusion
    behind a player, or briefly leaving frame) instead of using appearance
    to reacquire it
  - explicit status per frame in the output (detected / predicted / lost) so
    your downstream distance-traveled calc never silently treats a coasted
    guess as a real observation

Setup:
    pip install -U ultralytics opencv-python --break-system-packages

Run a single tracker:
    python track_ball.py --video match_clip.mp4 --weights ball_best.pt --tracker bytetrack --device 0
    python track_ball.py --video match_clip.mp4 --weights ball_best.pt --tracker kalman --device 0

Run both and get a comparison report:
    python track_ball.py --video match_clip.mp4 --weights ball_best.pt --tracker both --device 0
"""

import argparse
import csv
import json
import time
import cv2
import numpy as np
import torch
from collections import deque, defaultdict
from ultralytics import YOLO


# --------------------------------------------------------------------------
# Static (seed) distractor blacklist - permanent exclusions you've manually
# confirmed, e.g. from a previous run's learned_blacklist.json
# --------------------------------------------------------------------------

def load_blacklist(path):
    if not path:
        return []
    with open(path) as f:
        return json.load(f)


def is_blacklisted(cx, cy, blacklist):
    for spot in blacklist:
        if np.hypot(cx - spot["x"], cy - spot["y"]) <= spot["radius"]:
            return True
    return False


# --------------------------------------------------------------------------
# Live, self-updating distractor filter (replaces the separate
# find_distractors.py pre-pass). Watches EVERY raw detection each frame -
# not just whichever one gets picked as "the ball" - in a sliding time
# window. A real ball moves constantly, so it can never accumulate enough
# hits in one tight spot to trip this. A static false positive (bell,
# clock, light) sits in the same tiny spot every frame and trips it fast.
#
# Because the window slides, this also self-corrects: if a distractor
# stops firing (lighting changes, camera angle shifts), its old hits age
# out and it stops being suppressed - a static blacklist.json can't do
# that on its own.
# --------------------------------------------------------------------------

class OnlineDistractorFilter:
    def __init__(self, bin_size=25, window=500, min_frequency=0.5,
                 min_hits=30, max_stddev=8.0):
        self.bin_size = bin_size
        self.window = window              # frames of history to consider
        self.min_frequency = min_frequency  # fraction of window a bin must hit to be flagged
        self.min_hits = min_hits          # don't judge a bin until it's been seen this many times
        self.max_stddev = max_stddev      # how "static" (tight) a bin's hits must be
        self.history = deque()            # (frame_idx, key, cx, cy), oldest first
        self.bin_stats = defaultdict(lambda: {"count": 0, "sx": 0.0, "sy": 0.0,
                                                "sxx": 0.0, "syy": 0.0})

    def _key(self, cx, cy):
        return (int(cx // self.bin_size), int(cy // self.bin_size))

    def _expire(self, frame_idx):
        while self.history and frame_idx - self.history[0][0] >= self.window:
            _, key, ox, oy = self.history.popleft()
            s = self.bin_stats[key]
            s["count"] -= 1
            s["sx"] -= ox
            s["sy"] -= oy
            s["sxx"] -= ox * ox
            s["syy"] -= oy * oy
            if s["count"] <= 0:
                del self.bin_stats[key]

    def observe(self, frame_idx, cx, cy):
        """Feed in every raw detection this frame, regardless of whether it
        gets used - this is what lets the filter learn what's real vs static."""
        key = self._key(cx, cy)
        self.history.append((frame_idx, key, cx, cy))
        s = self.bin_stats[key]
        s["count"] += 1
        s["sx"] += cx
        s["sy"] += cy
        s["sxx"] += cx * cx
        s["syy"] += cy * cy
        self._expire(frame_idx)

    def _bin_verdict(self, s):
        n = s["count"]
        if n < self.min_hits:
            return None
        freq = n / self.window
        mean_x, mean_y = s["sx"] / n, s["sy"] / n
        var_x = max(0.0, s["sxx"] / n - mean_x ** 2)
        var_y = max(0.0, s["syy"] / n - mean_y ** 2)
        std = float(np.hypot(var_x ** 0.5, var_y ** 0.5))
        if freq >= self.min_frequency and std <= self.max_stddev:
            return mean_x, mean_y, freq, std
        return None

    def is_distractor(self, cx, cy):
        s = self.bin_stats.get(self._key(cx, cy))
        return s is not None and self._bin_verdict(s) is not None

    def export_blacklist(self, extra_radius_mult=1.5):
        """Current set of live-flagged regions, in the same format as
        find_distractors.py's blacklist.json - save this at the end of a
        run and pass it back in via --blacklist next time as a seed, so
        warm-up isn't needed from zero every single run."""
        out = []
        for s in self.bin_stats.values():
            v = self._bin_verdict(s)
            if v:
                mean_x, mean_y, freq, std = v
                out.append({"x": mean_x, "y": mean_y,
                             "radius": self.bin_size * extra_radius_mult,
                             "frequency": freq, "position_stddev_px": std})
        return out


# --------------------------------------------------------------------------
# ByteTrack backend
# --------------------------------------------------------------------------

def write_bytetrack_config(track_high_thresh, track_low_thresh, new_track_thresh,
                            track_buffer, match_thresh, path="_bytetrack_ball.yaml"):
    """Tuned differently from your player config: your ball detector's recall
    (~0.77 on test) is meaningfully lower than precision (~0.93), so
    track_low_thresh is dropped further than default to let the tracker use
    weaker detections to KEEP an existing track alive, even if they weren't
    strong enough to START a new one (new_track_thresh stays higher to avoid
    spawning tracks off stray false positives)."""
    content = f"""tracker_type: bytetrack
track_high_thresh: {track_high_thresh}
track_low_thresh: {track_low_thresh}
new_track_thresh: {new_track_thresh}
track_buffer: {track_buffer}
match_thresh: {match_thresh}
fuse_score: True
"""
    with open(path, "w") as f:
        f.write(content)
    return path


def run_bytetrack(video_path, weights_path, out_path, conf_thresh, device,
                   ball_class_id=0, track_buffer=15, track_high_thresh=0.35,
                   track_low_thresh=0.05, new_track_thresh=0.4, match_thresh=0.8,
                   blacklist=None, imgsz=640, distractor_filter=None):
    blacklist = blacklist or []
    model = YOLO(weights_path)
    tracker_cfg = write_bytetrack_config(track_high_thresh, track_low_thresh,
                                          new_track_thresh, track_buffer, match_thresh)

    results_gen = model.track(
        source=video_path,
        conf=conf_thresh,
        classes=[ball_class_id],
        device=device,
        half=(device != "cpu"),
        tracker=tracker_cfg,
        persist=True,
        stream=True,
        verbose=False,
        save=False,
        imgsz=imgsz,
    )

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    track_log = []
    last_id_seen = None
    id_switches = 0

    for frame_idx, r in enumerate(results_gen):
        frame = r.orig_img
        status = "lost"
        cx = cy = None
        conf = None
        track_id = None

        if r.boxes.id is not None and len(r.boxes) > 0:
            # ball should be singular - if the tracker reports more than one
            # live track this frame (false positive spawned a second track),
            # keep only the highest-confidence one that ISN'T a known OR
            # freshly-learned static distractor
            candidates = []
            for i, (box, tid, c) in enumerate(zip(r.boxes.xyxy.tolist(),
                                                    r.boxes.id.int().tolist(),
                                                    r.boxes.conf.tolist())):
                x1, y1, x2, y2 = box
                ccx, ccy = (x1 + x2) / 2, (y1 + y2) / 2
                if distractor_filter is not None:
                    distractor_filter.observe(frame_idx, ccx, ccy)  # learn from EVERY detection
                is_static = is_blacklisted(ccx, ccy, blacklist) or (
                    distractor_filter is not None and distractor_filter.is_distractor(ccx, ccy))
                if not is_static:
                    candidates.append((box, tid, c, ccx, ccy))

            if candidates:
                box, track_id, conf, cx, cy = max(candidates, key=lambda t: t[2])
                status = "detected"

                if last_id_seen is not None and track_id != last_id_seen:
                    id_switches += 1
                last_id_seen = track_id

                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 255), 2)
                cv2.circle(frame, (int(cx), int(cy)), 4, (0, 200, 255), -1)
                cv2.putText(frame, f"ball {track_id} ({conf:.2f})", (int(x1), max(0, int(y1) - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)

        track_log.append((frame_idx, status, track_id, cx, cy, conf))
        writer.write(frame)

    writer.release()
    extra = {"id_switches": id_switches}
    if distractor_filter is not None:
        extra["live_flagged_distractor_regions"] = len(distractor_filter.export_blacklist())
    return track_log, extra


# --------------------------------------------------------------------------
# Custom bounce-aware Kalman backend
# --------------------------------------------------------------------------

class BounceAwareBallTracker:
    """Single-object constant-velocity Kalman filter, deliberately loose
    (high process noise) so it adapts quickly to the hard velocity reversals
    a squash ball does on every wall/floor bounce, instead of smoothing
    through them like a tracker tuned for gradually-moving objects would.

    No appearance/ReID step at all - with one object of interest, "which
    detection is the ball" is answered by proximity to the physics
    prediction and confidence, not by what it looks like.
    """

    def __init__(self, process_noise=50.0, measurement_noise=4.0,
                 base_gate=40.0, gate_growth=25.0, max_coast_frames=10,
                 reacquire_after=20):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                              [0, 1, 0, 1],
                                              [0, 0, 1, 0],
                                              [0, 0, 0, 1]], np.float32)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                               [0, 1, 0, 0]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise

        self.base_gate = base_gate
        self.gate_growth = gate_growth
        self.max_coast_frames = max_coast_frames
        self.reacquire_after = reacquire_after

        self.initialized = False
        self.frames_lost = 0

    def _current_gate(self):
        return self.base_gate + self.gate_growth * self.frames_lost

    def _predicted_point(self):
        pred = self.kf.predict()
        return float(pred[0][0]), float(pred[1][0])

    def update(self, detections):
        """detections: list of (cx, cy, conf, box). Returns (status, cx, cy, conf, box)."""
        if not self.initialized:
            if detections:
                # bootstrap on the single highest-confidence detection
                cx, cy, conf, box = max(detections, key=lambda d: d[2])
                self.kf.statePre = np.array([[cx], [cy], [0], [0]], np.float32)
                self.kf.statePost = np.array([[cx], [cy], [0], [0]], np.float32)
                self.initialized = True
                self.frames_lost = 0
                return "detected", cx, cy, conf, box
            return "lost", None, None, None, None

        pred_x, pred_y = self._predicted_point()

        # find the closest detection to the physics prediction, within gate
        candidate = None
        best_dist = None
        gate = self._current_gate()
        for cx, cy, conf, box in detections:
            dist = np.hypot(cx - pred_x, cy - pred_y)
            if dist <= gate and (best_dist is None or dist < best_dist):
                candidate = (cx, cy, conf, box)
                best_dist = dist

        if candidate is not None:
            cx, cy, conf, box = candidate
            meas = np.array([[cx], [cy]], np.float32)
            self.kf.correct(meas)
            self.frames_lost = 0
            return "detected", cx, cy, conf, box

        # nothing matched the physics-consistent gate this frame
        self.frames_lost += 1

        if self.frames_lost > self.reacquire_after and detections:
            # been lost a while - stop trusting the coasted prediction and
            # snap onto the best available detection anywhere in frame
            # (handles: ball re-entered frame somewhere the old trajectory
            # doesn't explain, e.g. after leaving frame during a let/stoppage)
            cx, cy, conf, box = max(detections, key=lambda d: d[2])
            self.kf.statePre = np.array([[cx], [cy], [0], [0]], np.float32)
            self.kf.statePost = np.array([[cx], [cy], [0], [0]], np.float32)
            self.frames_lost = 0
            return "detected", cx, cy, conf, box

        if self.frames_lost <= self.max_coast_frames:
            # short gap (occlusion behind a player, one bad frame) - coast
            # on the physics prediction, but mark it clearly as NOT observed
            return "predicted", pred_x, pred_y, None, None

        return "lost", None, None, None, None


def run_kalman(video_path, weights_path, out_path, conf_thresh, device,
                ball_class_id=0, process_noise=50.0, measurement_noise=4.0,
                base_gate=40.0, gate_growth=25.0, max_coast_frames=10,
                reacquire_after=20, blacklist=None, imgsz=640, distractor_filter=None):
    blacklist = blacklist or []
    model = YOLO(weights_path)
    tracker = BounceAwareBallTracker(process_noise, measurement_noise, base_gate,
                                      gate_growth, max_coast_frames, reacquire_after)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    track_log = []
    frame_idx = 0
    reinit_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(frame, conf=conf_thresh, classes=[ball_class_id],
                                 device=device, half=(device != "cpu"), verbose=False,
                                 imgsz=imgsz)[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if distractor_filter is not None:
                distractor_filter.observe(frame_idx, cx, cy)  # learn from EVERY detection
            is_static = is_blacklisted(cx, cy, blacklist) or (
                distractor_filter is not None and distractor_filter.is_distractor(cx, cy))
            if not is_static:
                detections.append((cx, cy, conf, (x1, y1, x2, y2)))

        was_lost = tracker.frames_lost > tracker.reacquire_after
        status, cx, cy, conf, box = tracker.update(detections)
        if was_lost and status == "detected":
            reinit_count += 1

        if status == "detected":
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(frame, (int(cx), int(cy)), 4, (0, 255, 0), -1)
            cv2.putText(frame, f"ball ({conf:.2f})", (int(x1), max(0, int(y1) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        elif status == "predicted":
            cv2.circle(frame, (int(cx), int(cy)), 6, (0, 165, 255), 2)  # hollow orange = coasted guess
            cv2.putText(frame, "predicted", (int(cx) + 8, int(cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

        track_log.append((frame_idx, status, None, cx, cy, conf))
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    extra = {"reinit_count": reinit_count}
    if distractor_filter is not None:
        extra["live_flagged_distractor_regions"] = len(distractor_filter.export_blacklist())
    return track_log, extra


# --------------------------------------------------------------------------
# Shared output + comparison
# --------------------------------------------------------------------------

def write_csv(track_log, path):
    with open(path, "w", newline="") as f:
        w_csv = csv.writer(f)
        w_csv.writerow(["frame", "status", "track_id", "x", "y", "conf"])
        w_csv.writerows(track_log)


def summarize(track_log, extra_stats, label):
    total = len(track_log)
    detected = sum(1 for row in track_log if row[1] == "detected")
    predicted = sum(1 for row in track_log if row[1] == "predicted")
    lost = sum(1 for row in track_log if row[1] == "lost")

    # gap analysis: runs of consecutive non-"detected" frames
    gaps, cur_gap = [], 0
    for row in track_log:
        if row[1] != "detected":
            cur_gap += 1
        else:
            if cur_gap > 0:
                gaps.append(cur_gap)
            cur_gap = 0
    if cur_gap > 0:
        gaps.append(cur_gap)

    # path length over DETECTED points only, to avoid coasted/interpolated
    # points inflating or smoothing the distance number
    pts = [(row[3], row[4]) for row in track_log if row[1] == "detected"]
    path_len = sum(np.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
                    for i in range(1, len(pts))) if len(pts) > 1 else 0.0

    print(f"\n--- {label} ---")
    print(f"frames: {total}  detected: {detected} ({100*detected/total:.1f}%)  "
          f"predicted(coasted): {predicted}  lost: {lost}")
    if gaps:
        print(f"gaps in detection: {len(gaps)}, avg length {np.mean(gaps):.1f} frames, "
              f"longest {max(gaps)} frames")
    else:
        print("gaps in detection: 0")
    print(f"path length (detected points only, px): {path_len:.0f}")
    for k, v in extra_stats.items():
        print(f"{k}: {v}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=r"C:\Users\light\datasets\media\videos\short_clip.mp4")
    parser.add_argument("--weights", default=r"C:\Users\light\Downloads\different weights\runs\detect\runs\fine_tune\weights\best.pt")
    parser.add_argument("--out", default=r"C:\Users\light\datasets\Ball Detection\ball_trackv1.mp4")
    parser.add_argument("--conf", type=float, default=0.2,
                         help="detector confidence threshold - kept low since recall is the "
                              "weaker of your two detection metrics; let the tracker's motion "
                              "logic filter noise rather than starving it of detections")
    parser.add_argument("--imgsz", type=int, default=1280,
                         help="inference resolution. If detection is failing near-totally on real "
                              "match footage (full 1080p+), try 1280 or 1536 - a tiny ball can shrink "
                              "to sub-pixel size once YOLO letterboxes a wide-FOV frame down to 640")
    parser.add_argument("--tracker", choices=["bytetrack", "kalman", "both"], default="both")
    parser.add_argument("--device", default="0")
    parser.add_argument("--ball-class-id", type=int, default=0,
                         help="class index for 'ball' in your data.yaml - check it, don't assume 0")

    # kalman-specific knobs
    parser.add_argument("--process-noise", type=float, default=50.0)
    parser.add_argument("--measurement-noise", type=float, default=4.0)
    parser.add_argument("--base-gate", type=float, default=40.0)
    parser.add_argument("--gate-growth", type=float, default=25.0)
    parser.add_argument("--max-coast-frames", type=int, default=10)
    parser.add_argument("--reacquire-after", type=int, default=20)

    # bytetrack-specific knobs
    parser.add_argument("--track-buffer", type=int, default=15)

    parser.add_argument("--blacklist",
                         help="path to a blacklist.json (e.g. a previous run's "
                              "*_learned_blacklist.json) - permanent seed exclusions, always "
                              "applied in addition to the live auto-filter below")

    parser.add_argument("--auto-distractor-filter", dest="auto_distractor_filter",
                         action="store_true", default=True,
                         help="live filter that watches every raw detection and auto-suppresses "
                              "any spot that fires too often without moving (default: on)")
    parser.add_argument("--no-auto-distractor-filter", dest="auto_distractor_filter",
                         action="store_false")
    parser.add_argument("--distractor-window", type=int, default=500,
                         help="frames of rolling history the live filter considers (default ~20s at 25fps)")
    parser.add_argument("--distractor-min-freq", type=float, default=0.5,
                         help="fraction of the window a spot must fire in to get flagged as static")
    parser.add_argument("--distractor-min-hits", type=int, default=30,
                         help="don't judge a spot until it's fired at least this many times")
    parser.add_argument("--distractor-max-stddev", type=float, default=8.0,
                         help="how tightly clustered (px) a spot's hits must be to count as 'static' "
                              "rather than just an area the real ball passes through often")

    args = parser.parse_args()
    blacklist = load_blacklist(args.blacklist)
    if blacklist:
        print(f"Loaded {len(blacklist)} blacklisted region(s) from {args.blacklist}")

    device = args.device
    if device != "cpu" and not torch.cuda.is_available():
        print("CUDA not available - falling back to CPU (this will be slow)")
        device = "cpu"
    else:
        print(f"Running on: {'GPU ' + torch.cuda.get_device_name(0) if device != 'cpu' else 'CPU'}")

    runs = []
    if args.tracker in ("bytetrack", "both"):
        runs.append("bytetrack")
    if args.tracker in ("kalman", "both"):
        runs.append("kalman")

    for name in runs:
        out_video = f"{args.out}_{name}.mp4"
        out_csv = f"{args.out}_{name}_tracks.csv"
        t0 = time.time()

        distractor_filter = OnlineDistractorFilter(
            window=args.distractor_window,
            min_frequency=args.distractor_min_freq,
            min_hits=args.distractor_min_hits,
            max_stddev=args.distractor_max_stddev,
        ) if args.auto_distractor_filter else None

        if name == "bytetrack":
            log, extra = run_bytetrack(args.video, args.weights, out_video, args.conf, device,
                                        ball_class_id=args.ball_class_id,
                                        track_buffer=args.track_buffer,
                                        blacklist=blacklist, imgsz=args.imgsz,
                                        distractor_filter=distractor_filter)
        else:
            log, extra = run_kalman(args.video, args.weights, out_video, args.conf, device,
                                     ball_class_id=args.ball_class_id,
                                     process_noise=args.process_noise,
                                     measurement_noise=args.measurement_noise,
                                     base_gate=args.base_gate,
                                     gate_growth=args.gate_growth,
                                     max_coast_frames=args.max_coast_frames,
                                     reacquire_after=args.reacquire_after,
                                     blacklist=blacklist, imgsz=args.imgsz,
                                     distractor_filter=distractor_filter)

        elapsed = time.time() - t0
        write_csv(log, out_csv)
        print(f"\n[{name}] done in {elapsed:.1f}s -> {out_video}, {out_csv}")
        summarize(log, extra, name)

        if distractor_filter is not None:
            learned = distractor_filter.export_blacklist()
            if learned:
                learned_path = f"{args.out}_{name}_learned_blacklist.json"
                with open(learned_path, "w") as fp:
                    json.dump(blacklist + learned, fp, indent=2)  # merge seed + newly learned
                print(f"[{name}] auto-flagged {len(learned)} static distractor region(s) live "
                      f"-> saved to {learned_path}")
                for spot in learned:
                    print(f"    at ({spot['x']:.0f},{spot['y']:.0f}) - "
                          f"freq {100*spot['frequency']:.0f}% of window, "
                          f"stddev {spot['position_stddev_px']:.1f}px")
                print(f"    reuse next run with --blacklist {learned_path} to skip re-warming-up")

    if len(runs) == 2:
        print("\n=== how to read the comparison ===")
        print("- higher 'detected %' = fewer frames the model+tracker missed the ball outright")
        print("- fewer/shorter 'gaps' = tracker recovers faster after occlusion/misses")
        print("- bytetrack 'id_switches' > 0 means it silently relabeled the ball as a new")
        print("  object mid-rally (usually right after a sharp bounce) - a real failure mode")
        print("  for a single-object case where there should only ever be one id")
        print("- kalman 'reinit_count' = how often it gave up coasting and snapped onto a")
        print("  fresh detection - some is normal (ball leaving/re-entering frame), a lot")
        print("  suggests base_gate/gate_growth need loosening")
        print("- path length differs because 'predicted' (coasted) frames are excluded from")
        print("  both - don't treat a longer path as automatically 'more accurate', watch the")
        print("  output video for whichever one visually tracks bounces without lagging")


if __name__ == "__main__":
    main()
