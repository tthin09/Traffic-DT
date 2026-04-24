"""
Real-Time Video → SUMO Digital Twin via TraCI

This script processes a traffic video frame-by-frame using YOLO detection/tracking,
and mirrors each detected vehicle's actual position into a running SUMO simulation
using traci.vehicle.moveToXY(). This creates a true digital twin where SUMO reflects
the real-world traffic state observed in the video.

Another application can connect to SUMO to read the live traffic state.

Usage:
    python script_realtime.py --video data/tphcm/tphcm-2p.MOV [--gui] [--sumo-port 8813]
"""

import cv2
import os
import sys
import math
import argparse
import time as _time
import threading
import queue
import xml.etree.ElementTree as ET
from ultralytics import YOLO

# --- SUMO/TraCI imports ---
if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    print("WARNING: SUMO_HOME environment variable is not set.")
    print("Please set it to your SUMO installation directory.")

import traci
import traci.constants as tc

# Reuse helper functions from the existing script
from stash.script_stream import (
    generate_nod_file,
    generate_type_file,
    draw_polygonal_region,
    detect_region,
    load_regions_from_json,
)

# ============================================================
# Constants
# ============================================================

# Map YOLO class indices to vehicle class names
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# YOLO class IDs we want to track
TRACKED_CLASS_IDS = {2, 3, 5, 8}

# Pre-defined vehicle type dimensions for SUMO
# Default acceleration = 1.5 m/s², max velocity = 10 m/s for all types
VTYPE_DIMENSIONS = {
    "motorcycle": {
        "length": "2.2", "width": "0.8", "minGap": "0.5",
        "minGapLat": "0.3", "maxSpeedLat": "1.0", "latAlignment": "center",
        "accel": "1.5", "decel": "6.0", "sigma": "0.5",
        "guiShape": "motorcycle",
    },
    "car": {
        "length": "4.5", "width": "1.8", "minGap": "2.0",
        "minGapLat": "0.6", "maxSpeedLat": "0.5", "latAlignment": "center",
        "accel": "1.5", "decel": "5.0", "sigma": "0.5",
        "guiShape": "passenger",
    },
    "bus": {
        "length": "12.0", "width": "2.5", "minGap": "2.5",
        "minGapLat": "0.8", "maxSpeedLat": "0.3", "latAlignment": "center",
        "accel": "1.5", "decel": "4.0", "sigma": "0.5",
        "guiShape": "bus",
    },
    "truck": {
        "length": "8.0", "width": "2.3", "minGap": "2.5",
        "minGapLat": "0.7", "maxSpeedLat": "0.4", "latAlignment": "center",
        "accel": "1.5", "decel": "4.5", "sigma": "0.5",
        "guiShape": "truck",
    },
}

# All possible routes through the intersection
ROUTE_DEFINITIONS = {
    "route_north_to_south": "north_to_center center_to_south",
    "route_south_to_north": "south_to_center center_to_north",
    "route_east_to_west":   "east_to_center center_to_west",
    "route_west_to_east":   "west_to_center center_to_east",
    "route_north_to_east":  "north_to_center center_to_east",
    "route_north_to_west":  "north_to_center center_to_west",
    "route_east_to_south":  "east_to_center center_to_south",
    "route_east_to_north":  "east_to_center center_to_north",
    "route_south_to_west":  "south_to_center center_to_west",
    "route_south_to_east":  "south_to_center center_to_east",
    "route_west_to_south":  "west_to_center center_to_south",
    "route_west_to_north":  "west_to_center center_to_north",
}

OPPOSITE_DIRECTION = {
    "north": "south", "south": "north",
    "east": "west",   "west": "east",
}

# Number of frames a vehicle can be missing before being removed
LOST_VEHICLE_THRESHOLD = 30  # ~1 second at 30fps

# Minimum cumulative pixel displacement before a vehicle is injected into SUMO.
# Filters out parked vehicles that YOLO detects but never actually move.
MIN_MOVEMENT_PX = 20.0

# Default max speed in m/s
DEFAULT_MAX_SPEED_MS = 10.0





# ============================================================
# Network Setup Functions
# ============================================================

def generate_edg_file_uniform(output_file):
    """
    Generate edge file with uniform 2-lane roads for ALL directions.
    (The original network had south/west with only 1 lane.)
    """
    root = ET.Element("edges")

    directions = [
        ("n1", "center", "north_to_center"),
        ("center", "n1", "center_to_north"),
        ("n2", "center", "east_to_center"),
        ("center", "n2", "center_to_east"),
        ("n3", "center", "south_to_center"),
        ("center", "n3", "center_to_south"),
        ("n4", "center", "west_to_center"),
        ("center", "n4", "center_to_west"),
    ]

    for from_node, to_node, edge_id in directions:
        ET.SubElement(root, "edge",
                      **{"from": from_node, "to": to_node,
                         "id": edge_id, "type": "2L45", "numLanes": "2"})

    tree = ET.ElementTree(root)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)
    print(f"Generated uniform 2-lane edge file: {output_file}")


def generate_empty_route_file(output_file):
    """
    Generate a route file with all route + vType definitions but NO vehicles.
    Vehicles will be added dynamically via TraCI.
    """
    root = ET.Element("routes")

    # Pre-define vehicle types
    for cls_name, dims in VTYPE_DIMENSIONS.items():
        attrs = {"id": f"vType_{cls_name}", "maxSpeed": str(DEFAULT_MAX_SPEED_MS)}
        attrs.update(dims)
        ET.SubElement(root, "vType", **attrs)

    # Pre-define all routes
    for route_id, edges in ROUTE_DEFINITIONS.items():
        ET.SubElement(root, "route", id=route_id, edges=edges)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(output_file, encoding="utf-8", xml_declaration=True)
    print(f"Generated empty route file: {output_file}")


def generate_realtime_config(output_file, step_length=0.033):
    """
    Generate a SUMO config for real-time use with sublane model.
    """
    root = ET.Element("configuration")

    input_el = ET.SubElement(root, "input")
    ET.SubElement(input_el, "net-file", value="simple_nw_se.net.xml")
    ET.SubElement(input_el, "route-files", value="route.rou.xml")

    time_el = ET.SubElement(root, "time")
    ET.SubElement(time_el, "begin", value="0")
    ET.SubElement(time_el, "step-length", value=f"{step_length:.4f}")

    processing_el = ET.SubElement(root, "processing")
    ET.SubElement(processing_el, "lateral-resolution", value="0.8")
    ET.SubElement(processing_el, "collision.action", value="warn")
    ET.SubElement(processing_el, "collision.mingap-factor", value="0")

    report_el = ET.SubElement(root, "report")
    ET.SubElement(report_el, "verbose", value="true")
    ET.SubElement(report_el, "no-step-log", value="true")

    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(output_file, encoding="utf-8", xml_declaration=True)
    print(f"Generated real-time config: {output_file}")


def setup_sumo_network(output_dir):
    """Generate all SUMO network files and run netconvert."""
    os.makedirs(output_dir, exist_ok=True)

    nod_path = os.path.join(output_dir, "nod.xml")
    edg_path = os.path.join(output_dir, "edg.xml")
    type_path = os.path.join(output_dir, "type.xml")
    route_path = os.path.join(output_dir, "route.rou.xml")
    net_path = os.path.join(output_dir, "simple_nw_se.net.xml")

    generate_nod_file(nod_path)
    generate_edg_file_uniform(edg_path)  # All directions: 2 lanes
    generate_type_file(type_path)
    generate_empty_route_file(route_path)

    netconvert_cmd = (
        f'netconvert --node-files "{nod_path}" '
        f'--edge-files "{edg_path}" '
        f'--type-files "{type_path}" '
        f'-o "{net_path}"'
    )
    print(f"Running: {netconvert_cmd}")
    ret = os.system(netconvert_cmd)
    if ret != 0:
        raise RuntimeError(f"netconvert failed with return code {ret}")

    print("SUMO network files generated successfully!")


class VehicleManager:
    """
    Manages the lifecycle of vehicles in the SUMO simulation.

    Spawn-and-release strategy:
      - When a new vehicle is detected and has moved enough pixels (not parked),
        spawn it at the start of its entry edge with a default straight-through route.
      - Immediately release it to SUMO's autonomous driving (accel=1.5, maxSpeed=10).
      - Do NOT control the vehicle's position after spawn.

    Turn detection:
      - Each frame, check the vehicle's current region from the video (via regions.json).
      - If the current region differs from the entry region, the vehicle has turned.
      - Change the vehicle's route in SUMO to match the observed turn direction.
    """

    def __init__(self):
        self.active_vehicles = {}    # object_id -> {sumo_id, class, entry, last_seen_frame, current_route_exit}
        self.pending_vehicles = {}   # object_id -> {last_cx, last_cy, cumulative_dist, class, entry, last_frame}
        self.total_added = 0
        self.total_removed = 0
        self.total_filtered = 0      # Parked vehicles that were never injected
        self.total_rerouted = 0      # Vehicles whose route was changed due to turn detection

    def update_vehicle(self, object_id, px_cx, px_cy, vehicle_class,
                       entry_region, current_region, frame_count):
        """
        Update a vehicle. If new, check movement filter then spawn.
        If already active, check for turn detection and reroute if needed.

        :param object_id: YOLO tracker ID
        :param px_cx: Pixel center X
        :param px_cy: Pixel center Y
        :param vehicle_class: 'car', 'motorcycle', 'bus', 'truck'
        :param entry_region: The region where the vehicle first appeared
        :param current_region: The region the vehicle is currently in
        :param frame_count: Current frame number
        """

        # ---- Movement Distance Filter for new vehicles ----
        if object_id not in self.active_vehicles:
            if object_id not in self.pending_vehicles:
                # First time seeing this vehicle — start tracking displacement
                self.pending_vehicles[object_id] = {
                    "last_cx": px_cx,
                    "last_cy": px_cy,
                    "cumulative_dist": 0.0,
                    "class": vehicle_class,
                    "entry": entry_region,
                    "last_frame": frame_count,
                }
                return False  # Not added to SUMO yet

            # Accumulate displacement
            pending = self.pending_vehicles[object_id]
            dx = px_cx - pending["last_cx"]
            dy = px_cy - pending["last_cy"]
            pending["cumulative_dist"] += math.hypot(dx, dy)
            pending["last_cx"] = px_cx
            pending["last_cy"] = px_cy
            pending["last_frame"] = frame_count

            # Update entry if it was None before and now we have a region
            if pending["entry"] is None and entry_region is not None:
                pending["entry"] = entry_region

            if pending["cumulative_dist"] < MIN_MOVEMENT_PX:
                return False  # Still hasn't moved enough — probably parked

            # Vehicle has moved enough! Promote to active with spawn-and-release.
            entry_region = pending["entry"]
            del self.pending_vehicles[object_id]

            if entry_region is None:
                return False  # Can't spawn without knowing entry direction

            sumo_id = f"veh_{object_id}"
            if not self._spawn_vehicle(sumo_id, vehicle_class, entry_region):
                return False

            # Default route exit is straight through (opposite direction)
            default_exit = OPPOSITE_DIRECTION.get(entry_region, "south")
            self.active_vehicles[object_id] = {
                "sumo_id": sumo_id,
                "class": vehicle_class,
                "entry": entry_region,
                "last_seen_frame": frame_count,
                "current_route_exit": default_exit,
            }
            self.total_added += 1
            return True

        # ---- Known active vehicle: check for turn detection ----
        info = self.active_vehicles[object_id]
        info["last_seen_frame"] = frame_count

        if current_region is not None and current_region != info["entry"]:
            # Vehicle is now in a different region than where it entered.
            # This means it turned! Update route if not already set.
            if info["current_route_exit"] != current_region:
                self._reroute_vehicle(info, current_region)

        return True

    def _spawn_vehicle(self, sumo_id, vehicle_class, entry_region):
        """
        Spawn a vehicle at the start of its entry edge with a default
        straight-through route, then release it to SUMO's autonomous control.

        The vehicle will drive itself with accel=1.5 m/s², maxSpeed=10 m/s.
        """
        vtype_id = f"vType_{vehicle_class}"

        if entry_region in OPPOSITE_DIRECTION:
            exit_region = OPPOSITE_DIRECTION[entry_region]
        else:
            entry_region = "north"
            exit_region = "south"

        route_id = f"route_{entry_region}_to_{exit_region}"

        # Lane preferences: motorcycle=0 (inner), everything else=1 (outer)
        preferred_lane = 0 if vehicle_class == "motorcycle" else 1

        try:
            traci.vehicle.add(
                vehID=sumo_id,
                routeID=route_id,
                typeID=vtype_id,
                depart="now",
                departLane=str(preferred_lane),
                departSpeed="max",
                departPos="0.1",
            )
            # Let SUMO control the vehicle autonomously — no manual position control
            # Default speedMode=31 and laneChangeMode=1621 are the SUMO defaults
            # which handle car-following, right-of-way, etc.
            traci.vehicle.setSpeedMode(sumo_id, 31)
            traci.vehicle.setLaneChangeMode(sumo_id, 1621)
            traci.vehicle.setSpeed(sumo_id, -1)  # -1 = SUMO controls speed

            print(f"  [+] Spawned {sumo_id} ({vehicle_class}) on "
                  f"{entry_region}_to_center, route={route_id}")
            return True
        except traci.exceptions.TraCIException as e:
            print(f"  [!] Failed to add {sumo_id}: {e}")
            return False

    def _reroute_vehicle(self, info, new_exit_region):
        """
        Change a vehicle's route when a turn is detected in the video.

        The vehicle entered from info['entry'] and is now observed in
        new_exit_region, so we change its SUMO route accordingly.
        """
        sumo_id = info["sumo_id"]
        entry = info["entry"]
        new_route_id = f"route_{entry}_to_{new_exit_region}"

        if new_route_id not in ROUTE_DEFINITIONS:
            return  # Invalid route combination

        try:
            # Build the edge list for the new route
            new_edges = ROUTE_DEFINITIONS[new_route_id].split()
            traci.vehicle.setRoute(sumo_id, new_edges)

            old_exit = info["current_route_exit"]
            info["current_route_exit"] = new_exit_region
            self.total_rerouted += 1

            print(f"  [↪] Rerouted {sumo_id}: "
                  f"{entry}→{old_exit} => {entry}→{new_exit_region}")
        except traci.exceptions.TraCIException as e:
            print(f"  [!] Failed to reroute {sumo_id}: {e}")

    def cleanup_pending_vehicles(self, current_frame):
        """
        Remove pending vehicles that haven't been seen for a while.
        These are parked cars that never moved enough to be injected.
        """
        to_remove = []
        for object_id, info in self.pending_vehicles.items():
            if current_frame - info["last_frame"] > LOST_VEHICLE_THRESHOLD:
                to_remove.append(object_id)

        for object_id in to_remove:
            del self.pending_vehicles[object_id]
            self.total_filtered += 1

    def cleanup_lost_vehicles(self, current_frame):
        """
        Remove vehicles from tracking that haven't been detected for
        LOST_VEHICLE_THRESHOLD frames. The vehicle continues to exist
        in SUMO and drives itself out of the network autonomously.
        """
        to_cleanup = []

        for object_id, info in self.active_vehicles.items():
            frames_missing = current_frame - info["last_seen_frame"]

            if frames_missing > LOST_VEHICLE_THRESHOLD:
                sumo_id = info["sumo_id"]
                try:
                    # Verify vehicle still exists in simulation
                    traci.vehicle.getPosition(sumo_id)
                    # Vehicle is still driving — just stop tracking it
                    self.total_removed += 1
                    print(f"  [~] Released tracking of {sumo_id} "
                          f"(still driving autonomously in SUMO)")
                except traci.exceptions.TraCIException:
                    # Vehicle already left the network
                    self.total_removed += 1
                to_cleanup.append(object_id)

        for object_id in to_cleanup:
            del self.active_vehicles[object_id]


# ============================================================
# Threaded Pipeline Components
# ============================================================

class FrameReader:
    """Background thread that decodes video frames into a queue."""

    def __init__(self, video_path, max_queue_size=2):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"Error opening video file: {video_path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._queue = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self._frame_index = 0
        self._total_skipped = 0
        self._lock = threading.Lock()
        self._wall_start = _time.monotonic()
        self.max_skip_per_burst = 8
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _reader_loop(self):
        while not self._stop_event.is_set():
            with self._lock:
                wall_elapsed = _time.monotonic() - self._wall_start
                expected_frame = int(wall_elapsed * self.fps)
                frames_behind = expected_frame - self._frame_index
                if frames_behind > 1:
                    skip_n = min(frames_behind - 1, self.max_skip_per_burst)
                    for _ in range(skip_n):
                        if not self.cap.grab():
                            break
                        self._frame_index += 1
                        self._total_skipped += 1
                ret, frame = self.cap.read()
                if not ret:
                    self._queue.put(None)
                    return
                self._frame_index += 1
                idx = self._frame_index
            try:
                self._queue.put((idx, frame), timeout=1.0)
            except queue.Full:
                if self._stop_event.is_set():
                    return
                continue

    def read(self):
        try:
            return self._queue.get(timeout=2.0)
        except queue.Empty:
            return None

    @property
    def frame_index(self):
        with self._lock:
            return self._frame_index

    @property
    def total_skipped(self):
        with self._lock:
            return self._total_skipped

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=3.0)
        self.cap.release()


class TraCIWorker:
    """
    Background thread for all SUMO/TraCI communication.
    Main thread queues commands; this thread executes them serially,
    hiding ~5-10ms IPC latency behind the next YOLO inference.
    """

    def __init__(self):
        self._queue = queue.Queue(maxsize=64)
        self._stop_event = threading.Event()
        self._error = None
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                cmd = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if cmd is None:
                return
            try:
                cmd()
            except traci.exceptions.TraCIException as e:
                self._error = e

    def submit(self, fn):
        """Queue a callable to be executed on the TraCI thread."""
        self._queue.put(fn)

    def flush(self):
        """Wait until all queued commands are processed."""
        self._queue.join() if hasattr(self._queue, 'join') else None

    @property
    def error(self):
        return self._error

    def stop(self):
        self._stop_event.set()
        self._queue.put(None)
        self._thread.join(timeout=3.0)


def process_video_realtime(video_path, output_dir, use_gui=True, sumo_port=None):
    """
    3-thread real-time pipeline:
      Thread 1 (FrameReader):  CPU video decode → queue
      Thread 2 (Main):         YOLO inference (GPU) + detection processing
      Thread 3 (TraCIWorker):  SUMO IPC (spawn, reroute, simulationStep)

    Plus: YOLO runs every 2nd frame (INFER_STRIDE=2) to halve GPU load.
    On non-inference frames, only SUMO time is advanced.
    """

    # --- Load YOLO model ---
    # Prefer TensorRT engine if available (3-5× faster), fall back to .pt
    engine_path = "model/yolo11x.engine"
    pt_path = "model/yolo11x.pt"
    if os.path.exists(engine_path):
        model_name = engine_path
        print(f"Using TensorRT engine: {engine_path} (optimized for this GPU)")
    else:
        model_name = pt_path
        print(f"Using PyTorch model: {pt_path}")
        print("  TIP: Export to TensorRT for 3-5× speedup:")
        print(f'       python -c "from ultralytics import YOLO; YOLO(\'{pt_path}\').export(format=\'engine\', half=True, imgsz=960)"')
    model = YOLO(model_name)
    model.verbose = False

    # --- Start threaded frame reader ---
    reader = FrameReader(video_path, max_queue_size=2)
    fps = reader.fps
    width = reader.width
    height = reader.height
    step_length = 1.0 / fps

    print(f"Video: {width}x{height} @ {fps:.2f} FPS (step_length={step_length:.4f}s)")
    print(f"Pipeline: Threaded reader (queue=2) + main inference loop")

    # --- Setup SUMO network ---
    setup_sumo_network(output_dir)

    config_path = os.path.join(output_dir, "sumo_config.sumocfg")
    generate_realtime_config(config_path, step_length=step_length)

    # --- Start SUMO ---
    sumo_binary = "sumo-gui" if use_gui else "sumo"
    sumo_cmd = [sumo_binary, "-c", config_path, "--start"]

    if sumo_port:
        traci.start(sumo_cmd, port=sumo_port)
    else:
        traci.start(sumo_cmd)

    print(f"SUMO started ({'GUI' if use_gui else 'headless'}) with config: {config_path}")

    # --- Load regions ---
    regions = load_regions_from_json("regions.json")
    if regions is None:
        print("ERROR: Could not load regions.json")
        reader.stop()
        traci.close()
        return

    # --- Create vehicle manager (spawn-and-release + turn detection) ---
    vehicle_mgr = VehicleManager()

    # --- Start async TraCI worker ---
    traci_worker = TraCIWorker()

    # --- Tracking state ---
    track_data = {}  # object_id -> list of (frame, cx, cy, speed, label, entry, region)
    processed_count = 0      # frames actually run through YOLO
    infer_count = 0          # frames that actually ran YOLO

    # Only run YOLO every INFER_STRIDE frames — biggest perf win
    INFER_STRIDE = 2

    # Display every N processed frames
    DISPLAY_INTERVAL = 10

    # Cleanup SUMO stale vehicles every N frames
    CLEANUP_INTERVAL = 30

    # --- Process video with YOLO ---
    print("\n=== Starting real-time digital twin ===")
    print(f"Pipeline: reader thread + YOLO(stride={INFER_STRIDE}) + async TraCI")
    print("Press 'q' in the OpenCV window to stop.\n")

    wall_start = reader._wall_start
    t_infer_ms = 0.0

    try:
        while True:
            # --- Pull next pre-decoded frame from reader thread ---
            item = reader.read()
            if item is None:
                print("End of video.")
                break

            frame_count, frame = item
            processed_count += 1
            current_time = frame_count / fps

            # --- Run YOLO only on stride frames (every Nth) ---
            run_yolo = (processed_count % INFER_STRIDE == 0)

            if run_yolo:
                infer_count += 1
                t_infer_start = _time.monotonic()
                results_list = model.track(
                    source=frame,
                    imgsz=960,
                    conf=0.4,
                    half=True,
                    show=False,
                    stream=False,
                    verbose=False,
                    persist=True,
                    tracker="botsort.yaml",
                )
                results = results_list[0] if results_list else None
                t_infer_ms = (_time.monotonic() - t_infer_start) * 1000

                # --- Decide if we should render display this frame ---
                should_display = (infer_count % (DISPLAY_INTERVAL // INFER_STRIDE or 1) == 0)
                if should_display:
                    display_frame = frame.copy()
                    draw_polygonal_region(display_frame, regions)

                if results is not None:
                    for box in results.boxes:
                        if box.id is None:
                            continue
                        object_id = int(box.id[0])
                        cls = int(box.cls[0])
                        label = model.names[cls]
                        if cls not in TRACKED_CLASS_IDS:
                            continue

                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        region = detect_region(cx, cy, regions)

                        if object_id not in track_data:
                            speed = 0.0
                            entry_point = region
                        else:
                            last_frame, last_cx, last_cy, last_speed, *_ = track_data[object_id][-1]
                            frame_diff = frame_count - last_frame
                            if frame_diff > 0:
                                meters_per_pixel = 50 / 1420
                                distance_px = math.hypot(cx - last_cx, cy - last_cy)
                                distance_m = distance_px * meters_per_pixel
                                time_sec = frame_diff / fps
                                speed = (distance_m / time_sec) * 3.6
                            else:
                                speed = last_speed
                            entry_point = None

                        track_data.setdefault(object_id, []).append(
                            (frame_count, cx, cy, speed, label, entry_point, region)
                        )

                        vehicle_class = VEHICLE_CLASSES.get(cls, "car")
                        effective_entry = entry_point if entry_point else (
                            track_data[object_id][0][5] if track_data[object_id] else None
                        )

                        vehicle_mgr.update_vehicle(
                            object_id=object_id,
                            px_cx=cx,
                            px_cy=cy,
                            vehicle_class=vehicle_class,
                            entry_region=effective_entry,
                            current_region=region,
                            frame_count=frame_count,
                        )

                        if should_display:
                            in_sumo = "◉" if object_id in vehicle_mgr.active_vehicles else ""
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(
                                display_frame,
                                f"ID:{object_id} {label} {speed:.1f}km/h {in_sumo}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2,
                            )

                # --- Cleanup vehicles (batched) ---
                if infer_count % CLEANUP_INTERVAL == 0:
                    vehicle_mgr.cleanup_lost_vehicles(frame_count)
                    vehicle_mgr.cleanup_pending_vehicles(frame_count)
            else:
                should_display = False

            # --- Advance SUMO (async on TraCI thread) ---
            step_time = current_time
            traci_worker.submit(lambda t=step_time: traci.simulationStep(t))

            if traci_worker.error:
                print(f"TraCI error: {traci_worker.error}")
                break

            # --- HUD overlay & display (only on display frames) ---
            if should_display:
                total_skipped = reader.total_skipped
                active_count = len(vehicle_mgr.active_vehicles)
                effective_fps = 1000.0 / t_infer_ms if t_infer_ms > 0 else 0
                wall_elapsed = _time.monotonic() - wall_start
                avg_skip_per_sec = total_skipped / wall_elapsed if wall_elapsed > 0 else 0
                cv2.putText(
                    display_frame,
                    f"Frame: {frame_count} | Time: {current_time:.1f}s | "
                    f"YOLO: {t_infer_ms:.0f}ms ({effective_fps:.1f} FPS) | "
                    f"Skipped: {total_skipped} ({avg_skip_per_sec:.1f}/s)",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2,
                )
                cv2.putText(
                    display_frame,
                    f"Active: {active_count} | Added: {vehicle_mgr.total_added} | "
                    f"Rerouted: {vehicle_mgr.total_rerouted}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2,
                )
                cv2.imshow("Real-Time Traffic Digital Twin", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\nUser pressed 'q' — stopping.")
                    break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        total_skipped = reader.total_skipped
        reader.stop()
        traci_worker.stop()

        total_wall = _time.monotonic() - wall_start
        avg_skip = total_skipped / total_wall if total_wall > 0 else 0
        print(f"\n=== Session Summary ===")
        print(f"Frames read (video):  {reader.frame_index}")
        print(f"Frames processed:    {processed_count}")
        print(f"YOLO inferences:     {infer_count}")
        print(f"Frames skipped:      {total_skipped} (avg {avg_skip:.1f}/s)")
        print(f"Vehicles added:      {vehicle_mgr.total_added}")
        print(f"Vehicles released:   {vehicle_mgr.total_removed}")
        print(f"Vehicles rerouted:   {vehicle_mgr.total_rerouted}")
        print(f"Parked filtered:     {vehicle_mgr.total_filtered}")
        print(f"Still active:        {len(vehicle_mgr.active_vehicles)}")
        print(f"Still pending:       {len(vehicle_mgr.pending_vehicles)}")
        print(f"Total tracked:       {len(track_data)}")

        cv2.destroyAllWindows()
        try:
            traci.close()
            print("SUMO simulation closed.")
        except Exception:
            pass


# ============================================================
# Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Real-Time Video → SUMO Digital Twin via TraCI"
    )
    parser.add_argument(
        "--video", type=str, required=True,
        help="Path to the input video file",
    )
    parser.add_argument(
        "--gui", action="store_true", default=True,
        help="Launch SUMO with GUI (default: True)",
    )
    parser.add_argument(
        "--no-gui", action="store_true",
        help="Run SUMO headless (no GUI)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="sumo_files/realtime",
        help="Directory for generated SUMO files (default: sumo_files/realtime)",
    )
    parser.add_argument(
        "--sumo-port", type=int, default=None,
        help="TraCI port for SUMO (default: auto-assigned)",
    )

    args = parser.parse_args()
    use_gui = not args.no_gui

    if not os.path.exists(args.video):
        print(f"ERROR: Video file not found: {args.video}")
        sys.exit(1)

    process_video_realtime(
        video_path=args.video,
        output_dir=args.output_dir,
        use_gui=use_gui,
        sumo_port=args.sumo_port,
    )


if __name__ == "__main__":
    main()
