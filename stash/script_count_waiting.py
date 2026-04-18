"""
Video Vehicle Queue Counter

This script processes a traffic video using YOLO detection/tracking,
calculates vehicle speeds, and reports how many vehicles are "waiting" (speed < 2 m/s)
in each direction (north, south, east, west) every second.

No SUMO dependency — this is a standalone analysis tool.

Performance tips (use --help to see all options):
    --imgsz 640        Lower inference resolution (biggest speedup, default: 1920)
    --vid-stride 3     Only process every Nth frame (e.g. 3 = skip 2 of every 3)
    --model MODEL      Use a faster model (e.g. model/yolo11l-visdrone.pt)
    --half             Enable FP16 half-precision inference (GPU only)
    --no-display       Skip video rendering entirely

Usage:
    # Full quality (slow)
    python script_count_waiting.py --video data/tphcm/tphcm-2p.MOV

    # Fast real-time (~5-10 FPS)
    python script_count_waiting.py --video data/tphcm/tphcm-2p.MOV --imgsz 640 --vid-stride 3 --half

    # Maximum speed (headless)
    python script_count_waiting.py --video data/tphcm/tphcm-2p.MOV --imgsz 640 --vid-stride 3 --half --no-display
"""

import cv2
import os
import csv
import math
import json
import time
import argparse
from datetime import datetime
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

# Reuse helper functions from the existing script
from stash.script_stream import (
    draw_polygonal_region,
    detect_region,
    load_regions_from_json,
)

# ============================================================
# Constants
# ============================================================

# YOLO class IDs we want to track
TRACKED_CLASS_IDS = {2, 3, 5, 8}  # car, motorcycle, bus, truck(trailer)

# Map YOLO class indices to readable names
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Pixel-to-meter conversion factor (calibrated for this camera setup)
METERS_PER_PIXEL = 50 / 1420

# Speed threshold: vehicles below this are considered "waiting" (m/s)
WAITING_SPEED_THRESHOLD = 2.0  # m/s


# ============================================================
# Speed Calculator
# ============================================================

class SpeedCalculator:
    """
    Tracks per-vehicle positions across frames and computes instantaneous speed.
    Speed is smoothed using exponential moving average to reduce jitter.
    """

    def __init__(self, fps, smoothing=0.3):
        self.fps = fps
        self.smoothing = smoothing
        self.tracks = {}

    def update(self, object_id, cx, cy, frame_count, region):
        """
        Update vehicle position and return (speed_ms, entry_region).
        Coordinates (cx, cy) should be in the ORIGINAL video pixel space.
        """
        if object_id not in self.tracks:
            self.tracks[object_id] = {
                "last_frame": frame_count,
                "last_cx": cx,
                "last_cy": cy,
                "speed_ms": 0.0,
                "entry_region": region,
            }
            return 0.0, region

        track = self.tracks[object_id]
        frame_diff = frame_count - track["last_frame"]

        if frame_diff > 0:
            distance_px = math.hypot(cx - track["last_cx"], cy - track["last_cy"])
            distance_m = distance_px * METERS_PER_PIXEL
            time_sec = frame_diff / self.fps
            raw_speed = distance_m / time_sec

            track["speed_ms"] = (
                self.smoothing * raw_speed + (1 - self.smoothing) * track["speed_ms"]
            )

        track["last_frame"] = frame_count
        track["last_cx"] = cx
        track["last_cy"] = cy

        return track["speed_ms"], track["entry_region"]


# ============================================================
# Main Processing Loop
# ============================================================

def process_video(video_path, display=True, output_dir="output",
                  speed_threshold=WAITING_SPEED_THRESHOLD,
                  imgsz=1920, vid_stride=1, model_path="model/yolo11x.pt",
                  half=False, display_scale=0.5):
    """
    Process the video and print per-second waiting vehicle counts per direction.
    Results are saved to a CSV file line-by-line (survives sudden termination).
    """

    # --- Load YOLO model ---
    model = YOLO(model_path)
    model.verbose = False

    # --- Open video ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Error opening video file: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    effective_fps = fps / vid_stride

    print(f"Video: {width}x{height} @ {fps:.2f} FPS")
    print(f"Inference resolution: {imgsz}px | vid_stride: {vid_stride} | half: {half}")
    print(f"Effective rate: {effective_fps:.1f} frames/sec processed")
    print(f"Model: {model_path}")
    print(f"Waiting threshold: speed < {speed_threshold} m/s")
    print(f"{'='*60}")

    # --- Load regions ---
    regions = load_regions_from_json("regions.json")
    if regions is None:
        print("ERROR: Could not load regions.json")
        return

    # --- Setup CSV output ---
    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"{video_name}_{timestamp}.csv"
    csv_path = os.path.join(output_dir, csv_filename)

    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["second", "north_waiting", "south_waiting", "east_waiting", "west_waiting", "total_waiting",
                         "north_total", "south_total", "east_total", "west_total"])
    csv_file.flush()
    print(f"CSV output: {csv_path}")

    # --- Initialize ---
    speed_calc = SpeedCalculator(fps)
    frame_count = 0
    last_reported_second = -1

    current_second_waiting = defaultdict(set)
    current_second_total = defaultdict(set)

    # FPS measurement
    fps_timer = time.time()
    fps_frame_count = 0

    # --- Print header ---
    print(f"\n{'Time':>6s} | {'North':>6s} | {'South':>6s} | {'East':>6s} | {'West':>6s} | {'Total':>6s} | {'FPS':>5s}")
    print(f"{'-'*6} | {'-'*6} | {'-'*6} | {'-'*6} | {'-'*6} | {'-'*6} | {'-'*5}")

    # --- Process video with YOLO ---
    track_results = model.track(
        source=video_path,
        imgsz=imgsz,
        conf=0.4,
        show=False,
        stream=True,
        verbose=False,
        persist=True,
        tracker="botsort.yaml",
        vid_stride=vid_stride,
        half=half,
    )

    actual_fps = 0.0

    try:
        for results in track_results:
            frame_count += 1
            fps_frame_count += 1

            # Physical time in the video (accounting for skipped frames)
            current_time = (frame_count * vid_stride) / fps
            current_second = int(current_time)

            # Measure actual processing FPS
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                actual_fps = fps_frame_count / elapsed
                fps_frame_count = 0
                fps_timer = time.time()

            # --- Report at each new second boundary ---
            if current_second > last_reported_second and last_reported_second >= 0:
                _report_second(
                    last_reported_second,
                    current_second_waiting,
                    current_second_total,
                    csv_writer,
                    csv_file,
                    actual_fps,
                )
                current_second_waiting = defaultdict(set)
                current_second_total = defaultdict(set)

            last_reported_second = current_second

            # Get frame for display
            frame = results.orig_img.copy() if display else None
            if display:
                draw_polygonal_region(frame, regions)

            # --- Process each detected vehicle ---
            for box in results.boxes:
                if box.id is None:
                    continue

                object_id = int(box.id[0])
                cls = int(box.cls[0])

                if cls not in TRACKED_CLASS_IDS:
                    continue

                label = model.names[cls]

                # Bounding box center (in original video pixel space)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                # Detect which region this vehicle is in
                region = detect_region(cx, cy, regions)

                # Calculate speed (use physical frame count for correct time delta)
                speed_ms, entry_region = speed_calc.update(
                    object_id, cx, cy, frame_count * vid_stride, region
                )
                speed_kmh = speed_ms * 3.6

                # Track per-direction counts
                if region is not None:
                    current_second_total[region].add(object_id)
                    if speed_ms < speed_threshold:
                        current_second_waiting[region].add(object_id)

                # --- Draw on frame ---
                if display:
                    is_waiting = speed_ms < speed_threshold
                    color = (0, 0, 255) if is_waiting else (0, 255, 0)
                    status = "WAIT" if is_waiting else ""

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        f"ID:{object_id} {label} {speed_kmh:.1f}km/h {status}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2,
                    )

            # --- HUD overlay ---
            if display:
                waiting_summary = {
                    d: len(current_second_waiting[d])
                    for d in ["north", "south", "east", "west"]
                }
                total_waiting = sum(waiting_summary.values())

                cv2.putText(
                    frame,
                    f"Time: {current_time:.1f}s | FPS: {actual_fps:.1f} | "
                    f"Waiting: N={waiting_summary['north']} S={waiting_summary['south']} "
                    f"E={waiting_summary['east']} W={waiting_summary['west']} "
                    f"Total={total_waiting}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2,
                )

                # Resize display window for faster rendering
                if display_scale != 1.0:
                    disp_w = int(frame.shape[1] * display_scale)
                    disp_h = int(frame.shape[0] * display_scale)
                    frame = cv2.resize(frame, (disp_w, disp_h))

                cv2.imshow("Vehicle Queue Counter", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\nUser pressed 'q' — stopping.")
                    break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        if last_reported_second >= 0:
            _report_second(
                last_reported_second,
                current_second_waiting,
                current_second_total,
                csv_writer,
                csv_file,
                actual_fps,
            )

        csv_file.close()

        print(f"\n{'='*60}")
        print(f"Total seconds processed: {last_reported_second + 1}")
        print(f"Total frames processed:  {frame_count}")
        print(f"Total vehicles tracked:  {len(speed_calc.tracks)}")
        print(f"Results saved to: {csv_path}")

        cap.release()
        if display:
            cv2.destroyAllWindows()


def _report_second(second, waiting_by_dir, total_by_dir, csv_writer, csv_file, actual_fps):
    """Print and append one row to CSV (flushed immediately)."""
    directions = ["north", "south", "east", "west"]
    counts = {d: len(waiting_by_dir[d]) for d in directions}
    totals = {d: len(total_by_dir[d]) for d in directions}
    total_waiting = sum(counts.values())

    time_str = f"{second:>5d}s"
    print(
        f"{time_str} | {counts['north']:>6d} | {counts['south']:>6d} | "
        f"{counts['east']:>6d} | {counts['west']:>6d} | {total_waiting:>6d} | {actual_fps:>5.1f}"
    )

    csv_writer.writerow([
        second,
        counts["north"], counts["south"], counts["east"], counts["west"],
        total_waiting,
        totals["north"], totals["south"], totals["east"], totals["west"],
    ])
    csv_file.flush()


# ============================================================
# Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Count waiting vehicles per direction from traffic video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Performance presets:
  Balanced:   --imgsz 960  --vid-stride 2
  Fast:       --imgsz 640  --vid-stride 3 --half
  Max speed:  --imgsz 640  --vid-stride 5 --half --no-display
        """,
    )
    parser.add_argument(
        "--video", type=str, required=True,
        help="Path to the input video file",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Run without showing the video window (headless mode)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="output",
        help="Folder to save CSV results (default: output/)",
    )
    parser.add_argument(
        "--threshold", type=float, default=WAITING_SPEED_THRESHOLD,
        help=f"Speed threshold in m/s for 'waiting' (default: {WAITING_SPEED_THRESHOLD})",
    )

    # --- Performance options ---
    perf = parser.add_argument_group("performance options")
    perf.add_argument(
        "--imgsz", type=int, default=1920,
        help="YOLO inference resolution in pixels (default: 1920). "
             "Lower = faster. Try 960, 640, or 480.",
    )
    perf.add_argument(
        "--vid-stride", type=int, default=1,
        help="Process every Nth frame (default: 1 = all frames). "
             "E.g. 3 = process 1 of every 3 frames. Tracking still works.",
    )
    perf.add_argument(
        "--model", type=str, default="model/yolo11x.pt",
        help="Path to YOLO model weights (default: model/yolo11x.pt). "
             "Smaller models are faster, e.g. model/yolo11l-visdrone.pt",
    )
    perf.add_argument(
        "--half", action="store_true",
        help="Enable FP16 half-precision inference (GPU only, ~30%% faster)",
    )
    perf.add_argument(
        "--display-scale", type=float, default=0.5,
        help="Scale factor for display window (default: 0.5 = half size). "
             "Lower = faster rendering. Set to 1.0 for full size.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"ERROR: Video file not found: {args.video}")
        return

    process_video(
        video_path=args.video,
        display=not args.no_display,
        output_dir=args.output_dir,
        speed_threshold=args.threshold,
        imgsz=args.imgsz,
        vid_stride=args.vid_stride,
        model_path=args.model,
        half=args.half,
        display_scale=args.display_scale,
    )


if __name__ == "__main__":
    main()