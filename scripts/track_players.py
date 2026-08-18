"""
Track squash players across a video using your fine-tuned YOLO26n detector.
Supports two tracker backends via --tracker: deepsort | bytetrack

Setup:
    pip install -U ultralytics deep-sort-realtime opencv-python --break-system-packages

Run on GPU (default):
    python track.py --video match_clip.mp4 --weights best.pt --tracker bytetrack --device 0

Run on CPU (fallback / debugging):
    python track.py --video match_clip.mp4 --weights best.pt --device cpu
"""

import argparse
import csv
import time
from collections import deque
import cv2
import numpy as np
import torch
from ultralytics import YOLO


def write_tracker_config(track_buffer: int, path="_bytetrack_custom.yaml"):
    """Ultralytics' default bytetrack.yaml keeps a lost track alive for only
    ~30 frames before giving up and letting a re-appearance become a brand
    new ID. Bumping track_buffer gives occlusions (players crossing/blocking
    each other) more patience to be recovered as the SAME id instead of a
    fresh one - directly addresses ID churn during occlusion."""
    content = f"""tracker_type: bytetrack
track_high_thresh: 0.5
track_low_thresh: 0.1
new_track_thresh: 0.6
track_buffer: {track_buffer}
match_thresh: 0.8
fuse_score: True
"""
    with open(path, "w") as f:
        f.write(content)
    return path


def crop_hist(frame, box):
    """Hue+saturation histogram of a box crop - used as a lightweight
    appearance signature for re-identifying a track after ByteTrack loses
    and respawns it (across a cut, or after a brief occlusion)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def box_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def spatial_proximity(boxA, boxB, frame_diag):
    """1.0 if same spot, decaying toward 0.0 as centers get farther apart,
    normalized by the frame's diagonal so it's resolution-independent."""
    cA, cB = box_center(boxA), box_center(boxB)
    dist = np.hypot(cA[0] - cB[0], cA[1] - cB[1])
    return max(0.0, 1.0 - dist / frame_diag)


def calc_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def run_bytetrack(video_path, weights_path, out_path, conf_thresh, device,
                   reid=True, max_players=2, reid_window=30, track_buffer=90,
                   crossing_iou_thresh=0.05, spatial_weight=1.5):
    """Uses Ultralytics' built-in tracker, GPU-native (no separate CPU-bound
    embedder). When reid=True, also collapses ByteTrack's raw (and
    fragmenting-across-cuts) local IDs down to a small set of persistent
    player IDs using a rolling appearance signature per player.

    Critically: within a single frame, each persistent player id can be
    claimed by at most one detection. This is enforced explicitly rather
    than trusting appearance similarity alone, which is exactly what broke
    before - two simultaneously-visible players both getting labeled the
    same id when their appearance briefly looked similar (motion blur,
    lighting near a wall, etc). Since we always know there are at most
    max_players people on screen, this constraint alone rules out that
    failure mode regardless of how noisy the appearance signal gets.
    """
    model = YOLO(weights_path)
    track_log = []

    persistent_of_local = {}          # local_id -> persistent player id
    signatures = {}                    # persistent id -> deque of recent hists
    in_crossing = False               # are pid 1 & 2 currently overlapping?
    frozen_sig = {}                   # snapshot of each pid's appearance right before a crossing started
    last_box = {}                     # pid -> (frame_idx, box) - last confirmed position, for spatial continuity

    tracker_cfg = write_tracker_config(track_buffer)

    results_gen = model.track(
        source=video_path,
        conf=conf_thresh,
        classes=[0],
        device=device,
        half=(device != "cpu"),
        tracker=tracker_cfg,
        persist=True,
        stream=True,
        verbose=False,
        save=False,
    )

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    frame_diag = float(np.hypot(w, h))

    for frame_idx, r in enumerate(results_gen):
        frame = r.orig_img
        detections = []
        if r.boxes.id is not None:
            detections = list(zip(r.boxes.xyxy.tolist(), r.boxes.id.int().tolist()))

        frame_assignments = {}  # local_id -> pid, for THIS frame only

        # --- crossing-aware identity guard ---
        # Works on the mapping BEFORE this frame's new/lost ids are resolved,
        # since it's only meaningful for the 2 tracks that are continuously
        # being followed (this catches ByteTrack silently swapping which
        # local id is which during a close pass - a DIFFERENT failure mode
        # from track loss, and invisible to the new/lost-id logic below).
        local_box_now = {lid: box for box, lid in detections}
        existing_pid_local = {persistent_of_local[lid]: lid
                               for lid in local_box_now if lid in persistent_of_local}

        overlapping_now = False
        if 1 in existing_pid_local and 2 in existing_pid_local:
            iou = calc_iou(local_box_now[existing_pid_local[1]], local_box_now[existing_pid_local[2]])
            overlapping_now = iou > crossing_iou_thresh

        if overlapping_now and not in_crossing:
            # entering a crossing - snapshot appearance now, before it gets
            # ambiguous, so we have a clean "before" to check against after
            frozen_sig = {
                pid: (np.mean(list(signatures[pid]), axis=0).astype(np.float32)
                      if signatures.get(pid) else None)
                for pid in (1, 2)
            }
            in_crossing = True

        elif not overlapping_now and in_crossing:
            # just separated - verify the two boxes still match who we think
            # they are, and correct the mapping if ByteTrack swapped them
            local_1, local_2 = existing_pid_local.get(1), existing_pid_local.get(2)
            if local_1 and local_2 and frozen_sig.get(1) is not None and frozen_sig.get(2) is not None:
                hist_1 = crop_hist(frame, local_box_now[local_1])
                hist_2 = crop_hist(frame, local_box_now[local_2])
                if hist_1 is not None and hist_2 is not None:
                    correct_score = (cv2.compareHist(hist_1, frozen_sig[1], cv2.HISTCMP_CORREL)
                                      + cv2.compareHist(hist_2, frozen_sig[2], cv2.HISTCMP_CORREL))
                    swapped_score = (cv2.compareHist(hist_1, frozen_sig[2], cv2.HISTCMP_CORREL)
                                      + cv2.compareHist(hist_2, frozen_sig[1], cv2.HISTCMP_CORREL))
                    if swapped_score > correct_score:
                        persistent_of_local[local_1] = 2
                        persistent_of_local[local_2] = 1
                        print(f"  [reid] CROSSING CORRECTION at frame {frame_idx}: "
                              f"swapped local tracks {local_1}/{local_2} back to the right players "
                              f"(match {swapped_score:.2f} vs {correct_score:.2f})")
            in_crossing = False
            frozen_sig = {}
        # --- end crossing guard ---

        # --- continuous positional consistency check ---
        # Runs every frame, on BOTH currently-mapped ids (not just new/
        # ambiguous ones) - this is what the crossing guard above misses:
        # ByteTrack can silently keep feeding the wrong player's box into a
        # local id that never actually changed number, with no "new id"
        # event to hang a check off of. Bottom-left corner, L1 distance,
        # smaller total distance wins - if swapping the labels would make
        # both players' frame-to-frame motion look more continuous than
        # keeping them as-is, that's a strong signal a swap just happened.
        if reid and 1 in existing_pid_local and 2 in existing_pid_local:
            local_1, local_2 = existing_pid_local[1], existing_pid_local[2]
            box_1, box_2 = local_box_now[local_1], local_box_now[local_2]
            if 1 in last_box and 2 in last_box:
                bl = lambda b: (b[0], b[3])  # bottom-left corner: (x1, y2)
                l1 = lambda p, q: abs(p[0] - q[0]) + abs(p[1] - q[1])
                last_bl_1, last_bl_2 = bl(last_box[1][1]), bl(last_box[2][1])
                cur_bl_1, cur_bl_2 = bl(box_1), bl(box_2)

                current_cost = l1(cur_bl_1, last_bl_1) + l1(cur_bl_2, last_bl_2)
                swapped_cost = l1(cur_bl_1, last_bl_2) + l1(cur_bl_2, last_bl_1)

                if swapped_cost < current_cost:
                    persistent_of_local[local_1], persistent_of_local[local_2] = 2, 1
                    print(f"  [reid] POSITION SWAP at frame {frame_idx}: tracks {local_1}/{local_2} "
                          f"were closer to the other player's last spot "
                          f"(cost {swapped_cost:.1f} vs {current_cost:.1f})")
                    # keep local_box_now/existing_pid_local consistent for pass1 below
                    existing_pid_local[1], existing_pid_local[2] = local_2, local_1
        # --- end positional consistency check ---

        if reid:
            claimed_this_frame = set()
            new_ones = []  # (local_id, box, hist)

            # pass 1: anything already mapped keeps its id and claims its slot
            for box, local_id in detections:
                if local_id in persistent_of_local:
                    pid = persistent_of_local[local_id]
                    frame_assignments[local_id] = pid
                    claimed_this_frame.add(pid)
                    if not overlapping_now:   # don't pollute signatures mid-crossing
                        hist = crop_hist(frame, box)
                        if hist is not None:
                            signatures[pid].append(hist)
                else:
                    new_ones.append((local_id, box, crop_hist(frame, box)))

            # pass 2: resolve brand-new local ids against whatever persistent
            # slots are NOT already claimed by someone else on screen right now
            free_pids = [p for p in signatures if p not in claimed_this_frame]
            room_for_new = max_players - len(signatures)
            remaining = [t for t in new_ones if t[2] is not None]

            # forced case: exactly one unaccounted-for slot and one new
            # detection - no ambiguity, skip appearance matching entirely
            if len(remaining) == 1 and len(free_pids) == 1 and room_for_new == 0:
                local_id, box, hist = remaining[0]
                pid = free_pids[0]
                signatures[pid].append(hist)
                persistent_of_local[local_id] = pid
                frame_assignments[local_id] = pid
                print(f"  [reid] local track {local_id} -> player {pid} "
                      f"(forced: only one player unaccounted for this frame)")
                remaining = []

            # hard positional lock: ONE ambiguous box, BOTH slots technically
            # open (e.g. the other player is occluded behind this one). This
            # is exactly the "who's in front" case - appearance is unreliable
            # here since the crop may be blended/partial, but position is
            # strong: whichever player was last seen closest to this spot is
            # almost certainly who this still is. Deliberately no staleness
            # cutoff - always trust position over appearance in this specific
            # single-box-ambiguous case, per your call.
            if len(remaining) == 1 and len(free_pids) == 2:
                local_id, box, hist = remaining[0]
                candidates = [pid for pid in free_pids if pid in last_box]
                if candidates:
                    pid = max(candidates, key=lambda p: spatial_proximity(box, last_box[p][1], frame_diag))
                    signatures[pid].append(hist)
                    persistent_of_local[local_id] = pid
                    frame_assignments[local_id] = pid
                    prox = spatial_proximity(box, last_box[pid][1], frame_diag)
                    print(f"  [reid] local track {local_id} -> player {pid} "
                          f"(positional lock: closest to last known spot, prox={prox:.3f})")
                    remaining = []
                # else: no position history at all for either slot - fall through to appearance below

            # general case: greedy best-score-first matching, combining
            # appearance with spatial continuity (how close is this box to
            # where that player was last confirmed?) - spatial dominates
            # when it's fresh (occlusion just ended), and fades out if the
            # last known position is stale (e.g. after a real cut, where
            # "last position" isn't meaningful anymore)
            while remaining and free_pids:
                best = None
                for local_id, box, hist in remaining:
                    for pid in free_pids:
                        avg_hist = np.mean(list(signatures[pid]), axis=0).astype(np.float32)
                        app_score = cv2.compareHist(hist, avg_hist, cv2.HISTCMP_CORREL)

                        spatial_score = 0.0
                        if pid in last_box:
                            last_frame, last_b = last_box[pid]
                            if frame_idx - last_frame <= track_buffer:
                                spatial_score = spatial_proximity(box, last_b, frame_diag)

                        score = app_score + spatial_weight * spatial_score
                        if best is None or score > best[-1]:
                            best = (local_id, box, hist, pid, app_score, spatial_score, score)
                if best is None:
                    break
                local_id, box, hist, pid, app_score, spatial_score, score = best
                signatures[pid].append(hist)
                persistent_of_local[local_id] = pid
                frame_assignments[local_id] = pid
                flag = "  <- low confidence" if app_score < 0.75 and spatial_score < 0.5 else ""
                print(f"  [reid] local track {local_id} -> player {pid} "
                      f"(appearance={app_score:.3f}, position={spatial_score:.3f}){flag}")
                remaining = [t for t in remaining if t[0] != local_id]
                free_pids.remove(pid)

            # leftover: open brand-new player slots if we haven't seen
            # max_players people yet, otherwise drop as likely noise
            for local_id, box, hist in remaining:
                if room_for_new > 0:
                    pid = len(signatures) + 1
                    signatures[pid] = deque(maxlen=reid_window)
                    signatures[pid].append(hist)
                    persistent_of_local[local_id] = pid
                    frame_assignments[local_id] = pid
                    room_for_new -= 1
                    print(f"  [reid] local track {local_id} -> NEW player {pid}")
                else:
                    print(f"  [reid] local track {local_id} -> dropped (no free player slot)")
        else:
            for box, local_id in detections:
                frame_assignments[local_id] = local_id

        for box, local_id in detections:
            if local_id not in frame_assignments:
                continue
            pid = frame_assignments[local_id]
            last_box[pid] = (frame_idx, box)   # keep position fresh for spatial continuity matching
            x1, y1, x2, y2 = map(int, box)
            track_log.append((frame_idx, pid, x1, y1, x2, y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"player {pid}", (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        writer.write(frame)

    writer.release()
    return track_log


def run_deepsort(video_path, weights_path, out_path, conf_thresh, device):
    from deep_sort_realtime.deepsort_tracker import DeepSort

    model = YOLO(weights_path)
    tracker = DeepSort(
        max_age=45,
        n_init=3,
        nms_max_overlap=1.0,
        embedder="mobilenet",
        half=(device != "cpu"),
        bgr=True,
        embedder_gpu=(device != "cpu"),   # <- this was likely silently running on CPU before
    )

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_idx = 0
    track_log = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(frame, conf=conf_thresh, classes=[0], device=device,
                                 half=(device != "cpu"), verbose=False)[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "player"))

        tracks = tracker.update_tracks(detections, frame=frame)
        for t in tracks:
            if not t.is_confirmed():
                continue
            track_id = t.track_id
            x1, y1, x2, y2 = map(int, t.to_ltrb())
            track_log.append((frame_idx, track_id, x1, y1, x2, y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"player {track_id}", (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    return track_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=r"C:\Users\light\datasets\media\videos\Single_camera_squash.mov")
    parser.add_argument("--weights", default=r"C:\Users\light\datasets\runs\detect\runs\squash_player_det\yolo26n_finetune\weights\best.pt")
    parser.add_argument("--out", default=r"C:\Users\light\datasets\runs\coordinates\track_v2.mp4")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--tracker", choices=["deepsort", "bytetrack"], default="bytetrack")
    parser.add_argument("--device", default="0",
                         help='"0" for first GPU, "0,1" for multi-GPU, "cpu" to force CPU')
    parser.add_argument("--reid", action="store_true", default=True,
                         help="collapse fragmented track IDs into persistent player IDs (bytetrack only)")
    parser.add_argument("--no-reid", dest="reid", action="store_false",
                         help="disable persistent re-id, keep raw tracker IDs")
    parser.add_argument("--max-players", type=int, default=2)
    parser.add_argument("--reid-window", type=int, default=30,
                         help="how many recent appearance samples per player to keep for matching")
    parser.add_argument("--track-buffer", type=int, default=90,
                         help="frames a lost ByteTrack track survives before giving up - raise this "
                              "to make occlusions less likely to spawn a new id (default 90 ~= 3-4s)")
    parser.add_argument("--crossing-iou-thresh", type=float, default=1,
                         help="box overlap above this counts as a 'crossing' - triggers identity "
                              "re-verification once the players separate again")
    parser.add_argument("--spatial-weight", type=float, default=2,
                         help="how much to trust 'this box is where that player last was' vs pure "
                              "appearance when resolving an ambiguous/new track - higher favors position")
    args = parser.parse_args()

    device = args.device
    if device != "cpu" and not torch.cuda.is_available():
        print("CUDA not available - falling back to CPU (this will be slow)")
        device = "cpu"
    else:
        print(f"Running on: {'GPU ' + torch.cuda.get_device_name(0) if device != 'cpu' else 'CPU'}")

    t0 = time.time()
    if args.tracker == "bytetrack":
        track_log = run_bytetrack(args.video, args.weights, args.out, args.conf, device,
                                   reid=args.reid, max_players=args.max_players,
                                   reid_window=args.reid_window, track_buffer=args.track_buffer,
                                   crossing_iou_thresh=args.crossing_iou_thresh,
                                   spatial_weight=args.spatial_weight)
    else:
        if args.reid:
            print("Note: --reid is only implemented for --tracker bytetrack right now, ignoring for deepsort")
        track_log = run_deepsort(args.video, args.weights, args.out, args.conf, device)
    elapsed = time.time() - t0

    csv_path = args.out.rsplit(".", 1)[0] + "_tracks.csv"
    with open(csv_path, "w", newline="") as f:
        w_csv = csv.writer(f)
        w_csv.writerow(["frame", "track_id", "x1", "y1", "x2", "y2"])
        w_csv.writerows(track_log)

    print(f"Done in {elapsed:.1f}s using {args.tracker} on {device}")
    print(f"Annotated video: {args.out}")
    print(f"Track log: {csv_path}")


if __name__ == "__main__":
    main()