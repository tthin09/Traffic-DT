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
import json
import argparse
import numpy as np
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
VTYPE_DIMENSIONS = {
    "motorcycle": {
        "length": "2.2", "width": "0.8", "minGap": "0.5",
        "minGapLat": "0.3", "maxSpeedLat": "1.0", "latAlignment": "center",
        "accel": "2.5", "decel": "6.0", "sigma": "0.5",
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
        "accel": "0.8", "decel": "4.0", "sigma": "0.5",
        "guiShape": "bus",
    },
    "truck": {
        "length": "8.0", "width": "2.3", "minGap": "2.5",
        "minGapLat": "0.7", "maxSpeedLat": "0.4", "latAlignment": "center",
        "accel": "1.0", "decel": "4.5", "sigma": "0.5",
        "guiShape": "truck",
    },
}

DEFAULT_SPEED_KMH = 36.0  # 36 km/h = 10 m/s — must be >= departSpeed (10 m/s)

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

# Number of frames a vehicle can be missing before being released to auto-drive
LOST_VEHICLE_THRESHOLD = 30  # ~1 second at 30fps

# The visible road length in the video (meters)
# Vehicles will be mapped to only this range near the intersection
VISIBLE_ROAD_LENGTH = 10.0

# Total SUMO edge length (distance from outer node to center = 100m)
SUMO_EDGE_LENGTH = 100.0

# Minimum cumulative pixel displacement before a vehicle is injected into SUMO.
# Filters out parked vehicles that YOLO detects but never actually move.
MIN_MOVEMENT_PX = 20.0


# ============================================================
# Pixel → SUMO Edge-Based Position Mapping
# ============================================================

class EdgePositionMapper:
    """
    Maps pixel coordinates to SUMO edge + normalized position (0..1) by
    projecting the vehicle's pixel position onto the road's center line.

    Returns (edge_id, t) where t=0 is at the far entry, t=1 is at the center.
    Uses traci.vehicle.moveTo() which guarantees vehicles are ON the lane.
    """

    # Pixel reference lines for each road direction:
    #   "far"  = where vehicles enter the frame (far from intersection)
    #   "near" = where the road meets the intersection center
    # Estimated from regions.json polygon boundaries.
    PIXEL_AXES = {
        "north": {"far": (835, 0),      "near": (912, 348)},
        "east":  {"far": (1920, 468),   "near": (1105, 488)},
        "south": {"far": (1195, 1080),  "near": (770, 678)},
        "west":  {"far": (0, 520),      "near": (978, 538)},
    }

    def map_to_edge(self, px_cx, px_cy, region, entry_region):
        """
        Map a pixel position to a SUMO edge + normalized position.

        The video only covers ~10m of road near the intersection.
        We map that to the last 10m of the 100m SUMO edge (positions 90-100m
        for inbound, 0-10m for outbound).

        :param px_cx: Vehicle pixel center X
        :param px_cy: Vehicle pixel center Y
        :param region: Current detected region (north/south/east/west)
        :param entry_region: Region where this vehicle first appeared
        :returns: (edge_id, t) where t is 0..1 along the edge, or None
        """
        if region is None or region not in self.PIXEL_AXES:
            return None

        # Determine if vehicle is inbound (approaching center) or outbound
        if region == entry_region:
            edge_id = f"{region}_to_center"      # inbound
        else:
            edge_id = f"center_to_{region}"       # outbound

        # Project pixel position onto the road axis to get t_pixel ∈ [0, 1]
        # t_pixel=0 means at the "far" end (entry), t_pixel=1 means at "near" (center)
        axis = self.PIXEL_AXES[region]
        far = np.array(axis["far"], dtype=np.float64)
        near = np.array(axis["near"], dtype=np.float64)
        point = np.array([px_cx, px_cy], dtype=np.float64)

        axis_vec = near - far
        t_pixel = np.dot(point - far, axis_vec) / np.dot(axis_vec, axis_vec)
        t_pixel = np.clip(t_pixel, 0.0, 1.0)

        # Map t_pixel to only the last VISIBLE_ROAD_LENGTH meters of the edge.
        # For inbound edges (X_to_center): visible zone is [EDGE_LENGTH - VISIBLE, EDGE_LENGTH]
        #   t_pixel=0 (far in video) -> position = EDGE_LENGTH - VISIBLE_ROAD_LENGTH
        #   t_pixel=1 (near center)  -> position = EDGE_LENGTH
        # We express as normalized t over the full edge:
        visible_fraction = VISIBLE_ROAD_LENGTH / SUMO_EDGE_LENGTH  # 0.10
        start_fraction = 1.0 - visible_fraction  # 0.90

        if region == entry_region:
            # Inbound: map to last 10m (t = 0.90 .. 0.99)
            t = start_fraction + t_pixel * visible_fraction
            t = np.clip(t, start_fraction + 0.01, 0.99)
        else:
            # Outbound (center_to_X): map to first 10m (t = 0.01 .. 0.10)
            t = (1.0 - t_pixel) * visible_fraction
            t = np.clip(t, 0.01, visible_fraction - 0.01)

        return edge_id, float(t)


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
        attrs = {"id": f"vType_{cls_name}", "maxSpeed": str(DEFAULT_SPEED_KMH / 3.6)}
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


# ============================================================
# TraCI Vehicle Management
# ============================================================

class VehicleManager:
    """
    Manages the lifecycle of vehicles in the SUMO simulation.
    Tracks which vehicles are active, adds new ones, updates positions,
    and removes vehicles that are no longer detected.

    Uses a Movement Distance Filter: vehicles must accumulate at least
    MIN_MOVEMENT_PX pixels of displacement before being injected into SUMO.
    This filters out parked vehicles.
    """

    def __init__(self, edge_mapper):
        self.edge_mapper = edge_mapper
        self.active_vehicles = {}    # object_id -> {sumo_id, class, entry, last_seen_frame, ...}
        self.pending_vehicles = {}   # object_id -> {first_cx, first_cy, last_cx, last_cy,
                                     #               cumulative_dist, class, entry, last_frame}
        self.total_added = 0
        self.total_removed = 0
        self.total_filtered = 0      # Parked vehicles that were never injected

    def update_vehicle(self, object_id, px_cx, px_cy, speed_kmh, vehicle_class,
                       entry_region, current_region, frame_count):
        """
        Update a vehicle's position in SUMO. If the vehicle doesn't exist yet,
        check if it has moved enough to be considered a real moving vehicle.

        New vehicles are hard-spawned at a queue-aware position 10 m behind the
        intersection (see _add_vehicle). The pixel-based _move_vehicle is NOT
        called on the first spawn frame so the initial position is preserved.

        :param object_id: YOLO tracker ID
        :param px_cx: Pixel center X
        :param px_cy: Pixel center Y
        :param speed_kmh: Detected speed in km/h
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

            mapping = self.edge_mapper.map_to_edge(px_cx, px_cy, current_region, pending["entry"])
            if mapping is None:
                return False  # Wait until it maps to a valid lane

            # Vehicle has moved enough! Promote to active with hard-spawn.
            entry_region = pending["entry"]
            del self.pending_vehicles[object_id]

            _, t_position = mapping
            sumo_id = f"veh_{object_id}"
            if not self._add_vehicle(sumo_id, vehicle_class, entry_region, t_position):
                return False

            self.active_vehicles[object_id] = {
                "sumo_id": sumo_id,
                "class": vehicle_class,
                "entry": entry_region,
                "last_seen_frame": frame_count,
                "last_speed_kmh": speed_kmh,
                "released": False,
            }
            self.total_added += 1
            # Return here — do NOT call _move_vehicle on the first spawn frame
            # so the queue-aware spawn position is preserved.
            return True

        # ---- Known active vehicle: pixel-track its position ----
        sumo_id = f"veh_{object_id}"

        mapping = self.edge_mapper.map_to_edge(px_cx, px_cy, current_region, entry_region)
        if mapping is None:
            # Vehicle is not in any known region — keep last_seen updated
            self.active_vehicles[object_id]["last_seen_frame"] = frame_count
            return False

        edge_id, t_position = mapping
        self._move_vehicle(sumo_id, edge_id, t_position, speed_kmh, vehicle_class)
        self.active_vehicles[object_id]["last_seen_frame"] = frame_count
        self.active_vehicles[object_id]["last_speed_kmh"] = speed_kmh
        self.active_vehicles[object_id]["released"] = False
        return True

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

    def release_lost_vehicles(self, current_frame):
        """
        When a vehicle hasn't been detected for LOST_VEHICLE_THRESHOLD frames,
        release it to SUMO's autonomous driving at its last known speed.
        The vehicle will continue driving and exit the network naturally.
        """
        to_release = []
        to_cleanup = []

        for object_id, info in self.active_vehicles.items():
            frames_missing = current_frame - info["last_seen_frame"]

            if frames_missing > LOST_VEHICLE_THRESHOLD and not info.get("released", False):
                to_release.append(object_id)

        for object_id in to_release:
            info = self.active_vehicles[object_id]
            sumo_id = info["sumo_id"]
            last_speed = info.get("last_speed_kmh", DEFAULT_SPEED_KMH)

            try:
                # Re-enable SUMO's autonomous driving
                traci.vehicle.setSpeedMode(sumo_id, 31)  # Default speed mode
                traci.vehicle.setLaneChangeMode(sumo_id, 1621)  # Default lane change
                traci.vehicle.setSpeed(sumo_id, -1)  # -1 = let SUMO control speed
                traci.vehicle.setMaxSpeed(sumo_id, last_speed / 3.6)  # Set max speed

                info["released"] = True
                self.total_removed += 1
                print(f"  [~] Released {sumo_id} to auto-drive at {last_speed:.1f} km/h")

            except traci.exceptions.TraCIException:
                # Vehicle already left the network
                to_cleanup.append(object_id)

        # Also check if released vehicles have left the network entirely
        for object_id, info in list(self.active_vehicles.items()):
            if info.get("released", False):
                sumo_id = info["sumo_id"]
                try:
                    # Check if vehicle still exists in simulation
                    traci.vehicle.getPosition(sumo_id)
                except traci.exceptions.TraCIException:
                    to_cleanup.append(object_id)

        for object_id in set(to_cleanup):
            if object_id in self.active_vehicles:
                del self.active_vehicles[object_id]

    def _add_vehicle(self, sumo_id, vehicle_class, entry_region, t_position):
        """
        Add a new vehicle to SUMO with a dynamic queue-aware spawn position.

        Spawn strategy:
          1. Uses the actual normalized camera position (t_position).
          2. Checks existing vehicles on the route. Ensures new vehicles are 
             placed behind the rearmost vehicle (end of the queue).
          3. If the preferred lane's queue extends past QUEUE_TOO_LONG_THRESHOLD,
             try the adjacent lane first (overflow logic).
        """
        vtype_id = f"vType_{vehicle_class}"

        if entry_region and entry_region in OPPOSITE_DIRECTION:
            exit_region = OPPOSITE_DIRECTION[entry_region]
        else:
            entry_region = "north"
            exit_region = "south"

        route_id = f"route_{entry_region}_to_{exit_region}"
        edge_id  = f"{entry_region}_to_center"

        # Lane preferences: motorcycle=0 (inner), everything else=1 (outer)
        preferred_lane = 0 if vehicle_class == "motorcycle" else 1
        # Both lanes are eligible for overflow (motorcycle → car lane and vice-versa)
        lanes_to_try = [preferred_lane, 1 - preferred_lane]

        dims       = VTYPE_DIMENSIONS.get(vehicle_class, VTYPE_DIMENSIONS["car"])
        veh_length = float(dims["length"])
        veh_mingap = float(dims["minGap"])

        # If the queue on the preferred lane reaches further than this from the
        # edge start, consider the lane "too long" and try the adjacent one.
        QUEUE_TOO_LONG_THRESHOLD = 50.0

        def _safe_spawn_pos(lane_id):
            """Return (safe_pos, lane_length) or (None, None) on TraCI error."""
            try:
                lane_len = traci.lane.getLength(lane_id)
            except traci.exceptions.TraCIException:
                return None, None

            mapped_pos = t_position * lane_len

            try:
                veh_ids = traci.lane.getLastStepVehicleIDs(lane_id)
            except traci.exceptions.TraCIException:
                veh_ids = []

            occupied = []
            for vid in veh_ids:
                try:
                    front = traci.vehicle.getLanePosition(vid)
                    vlen  = traci.vehicle.getLength(vid)
                    occupied.append(front - vlen)
                except traci.exceptions.TraCIException:
                    pass

            if occupied:
                last_rear = min(occupied)
                # Cap the spawn position to be behind the queue, 
                # but don't place it ahead of the actual camera-mapped position.
                safe = min(mapped_pos, last_rear - veh_mingap - veh_length)
            else:
                safe = mapped_pos

            safe = max(0.1, min(safe, lane_len - 0.1))
            return safe, lane_len

        chosen_lane = None
        chosen_pos  = None
        fallback    = None  # (lane_idx, pos) if all lanes are too congested

        for lane_idx in lanes_to_try:
            lane_id = f"{edge_id}_{lane_idx}"
            pos, _  = _safe_spawn_pos(lane_id)
            if pos is None:
                continue

            if pos >= QUEUE_TOO_LONG_THRESHOLD:
                # Queue is short enough on this lane — use it
                chosen_lane = lane_idx
                chosen_pos  = pos
                break
            else:
                # Queue is long; remember as fallback, try the adjacent lane
                if fallback is None:
                    fallback = (lane_idx, pos)

        if chosen_lane is None:
            # All lanes congested — use the least-backed-up one as fallback
            if fallback is not None:
                chosen_lane, chosen_pos = fallback
            else:
                chosen_lane = preferred_lane
                chosen_pos  = 1.0  # Last resort: very start of edge

        lane_id = f"{edge_id}_{chosen_lane}"

        try:
            traci.vehicle.add(
                vehID=sumo_id,
                routeID=route_id,
                typeID=vtype_id,
                depart="now",
                departSpeed="7",
            )
            # Disable SUMO's autonomous driving — position is fixed by moveTo
            traci.vehicle.setSpeedMode(sumo_id, 0)
            traci.vehicle.setLaneChangeMode(sumo_id, 0)
            traci.vehicle.setSpeed(sumo_id, 7.0)

            # Hard-place the vehicle at the queue-aware spawn position
            traci.vehicle.moveTo(sumo_id, lane_id, chosen_pos)

            print(f"  [+] Spawned {sumo_id} ({vehicle_class}) on "
                  f"{lane_id} at pos={chosen_pos:.1f} m "
                  f"({'preferred' if chosen_lane == preferred_lane else 'overflow'} lane)")
            return True
        except traci.exceptions.TraCIException as e:
            print(f"  [!] Failed to add {sumo_id}: {e}")
            return False

    def _move_vehicle(self, sumo_id, edge_id, t_position, speed_kmh, vehicle_class):
        """
        Place a vehicle directly on a lane at a specific position.
        Uses traci.vehicle.moveTo() which is impossible to go off-road.

        Lane assignment:
          - motorcycle → lane 0 (inside)
          - car/bus/truck → lane 1 (outside)

        :param sumo_id: Vehicle ID in SUMO
        :param edge_id: Target edge (e.g. 'north_to_center')
        :param t_position: Normalized position along edge (0.0 = start, 1.0 = end)
        :param speed_kmh: Speed in km/h
        :param vehicle_class: 'car', 'motorcycle', 'bus', 'truck'
        """
        try:
            # Lane 0 = inside (motorcycle), Lane 1 = outside (car/bus/truck)
            lane_index = 0 if vehicle_class == "motorcycle" else 1
            lane_id = f"{edge_id}_{lane_index}"

            # Get the lane length and compute absolute position in meters
            lane_length = traci.lane.getLength(lane_id)
            pos = t_position * lane_length
            pos = max(0.1, min(pos, lane_length - 0.1))  # Stay inside the lane

            # moveTo places the vehicle EXACTLY on the lane — cannot be off-road
            traci.vehicle.moveTo(sumo_id, lane_id, pos)

            # Set speed to match detected speed
            speed_ms = max(speed_kmh / 3.6, 0.1)
            traci.vehicle.setSpeed(sumo_id, speed_ms)

        except traci.exceptions.TraCIException:
            # Vehicle may have been removed by SUMO
            pass


# ============================================================
# Main Real-Time Processing Loop
# ============================================================

def process_video_realtime(video_path, output_dir, use_gui=True, sumo_port=None):
    """
    Main real-time processing loop:
    1. Open video & set up SUMO network
    2. Start SUMO via TraCI
    3. For each frame: detect vehicles → map positions → update SUMO
    4. Cleanup on exit
    """

    # --- Load YOLO model ---
    model_name = "model/yolo11x.pt"
    model = YOLO(model_name)
    model.verbose = False

    # --- Open video ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Error opening video file: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    step_length = 1.0 / fps

    print(f"Video: {width}x{height} @ {fps:.2f} FPS (step_length={step_length:.4f}s)")

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
        traci.close()
        return

    # --- Create edge position mapper ---
    edge_mapper = EdgePositionMapper()
    vehicle_mgr = VehicleManager(edge_mapper)

    # --- Tracking state ---
    track_data = {}  # object_id -> list of (frame, cx, cy, speed, label, entry, region)
    frame_count = 0          # next frame index to read from video
    processed_count = 0      # frames actually run through YOLO
    total_skipped = 0        # cumulative skipped frames

    # Maximum frames to skip in one burst to avoid huge jumps
    MAX_SKIP_PER_BURST = 10

    # --- Process video with YOLO ---
    print("\n=== Starting real-time digital twin ===")
    print("Press 'q' in the OpenCV window to stop.\n")

    import time as _time
    wall_start = _time.monotonic()  # wall-clock start reference

    try:
        while True:
            # ----------------------------------------------------------------
            # Frame-skip logic: compare wall-clock time to video playback time.
            # If we are behind real-time, skip frames to catch up.
            # ----------------------------------------------------------------
            wall_elapsed = _time.monotonic() - wall_start
            expected_frame = int(wall_elapsed * fps)  # frame index we SHOULD be at
            frames_behind = expected_frame - frame_count

            if frames_behind > 1:
                skip_n = min(frames_behind - 1, MAX_SKIP_PER_BURST)
                # Read & discard skip_n frames
                for _ in range(skip_n):
                    ret_skip = cap.grab()  # grab without decode — fast
                    if not ret_skip:
                        break
                    frame_count += 1
                    total_skipped += 1

                video_time_at_skip = frame_count / fps
                print(
                    f"[FRAME-SKIP] t={video_time_at_skip:.2f}s | "
                    f"skipped {skip_n} frame(s) | "
                    f"total skipped so far: {total_skipped} | "
                    f"wall={wall_elapsed:.2f}s"
                )

            # Read the next frame for processing
            ret, frame = cap.read()
            if not ret:
                print("End of video.")
                break
            frame_count += 1
            processed_count += 1
            current_time = frame_count / fps

            # Run YOLO tracking on this frame
            results_list = model.track(
                source=frame,
                imgsz=1920,
                conf=0.4,
                show=False,
                stream=False,
                verbose=False,
                persist=True,
                tracker="botsort.yaml",
            )
            results = results_list[0] if results_list else None

            display_frame = frame.copy()

            # Draw region overlays
            draw_polygonal_region(display_frame, regions)

            if results is not None:
                # --- Process each detected vehicle ---
                for box in results.boxes:
                    if box.id is None:
                        continue

                    object_id = int(box.id[0])
                    cls = int(box.cls[0])
                    label = model.names[cls]

                    if cls not in TRACKED_CLASS_IDS:
                        continue

                    # Bounding box center
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2

                    # Detect which region this vehicle is in
                    region = detect_region(cx, cy, regions)

                    # --- Calculate speed ---
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

                    # Store tracking data
                    track_data.setdefault(object_id, []).append(
                        (frame_count, cx, cy, speed, label, entry_point, region)
                    )

                    # Determine vehicle class and entry region
                    vehicle_class = VEHICLE_CLASSES.get(cls, "car")
                    effective_entry = entry_point if entry_point else (
                        track_data[object_id][0][5] if track_data[object_id] else None
                    )

                    # --- Update vehicle position in SUMO ---
                    vehicle_mgr.update_vehicle(
                        object_id=object_id,
                        px_cx=cx,
                        px_cy=cy,
                        speed_kmh=speed if speed > 0 else DEFAULT_SPEED_KMH,
                        vehicle_class=vehicle_class,
                        entry_region=effective_entry,
                        current_region=region,
                        frame_count=frame_count,
                    )

                    # --- Draw on frame ---
                    in_sumo = "◉" if object_id in vehicle_mgr.active_vehicles else ""
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        display_frame,
                        f"ID:{object_id} {label} {speed:.1f}km/h {in_sumo}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2,
                    )

            # --- Release vehicles that left the camera to auto-drive ---
            vehicle_mgr.release_lost_vehicles(frame_count)
            vehicle_mgr.cleanup_pending_vehicles(frame_count)

            # --- Advance SUMO simulation to match video time ---
            try:
                traci.simulationStep(current_time)
            except traci.exceptions.TraCIException as e:
                print(f"TraCI simulation step error: {e}")
                break

            # --- HUD overlay ---
            active_count = len(vehicle_mgr.active_vehicles)
            cv2.putText(
                display_frame,
                f"Frame: {frame_count} | Time: {current_time:.1f}s | "
                f"Active in SUMO: {active_count} | Total added: {vehicle_mgr.total_added} | "
                f"Skipped: {total_skipped}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2,
            )

            cv2.imshow("Real-Time Traffic Digital Twin", display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\nUser pressed 'q' — stopping.")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        print(f"\n=== Session Summary ===")
        print(f"Frames read (video):  {frame_count}")
        print(f"Frames processed:    {processed_count}")
        print(f"Frames skipped:      {total_skipped}")
        print(f"Vehicles added:      {vehicle_mgr.total_added}")
        print(f"Vehicles released:   {vehicle_mgr.total_removed}")
        print(f"Parked filtered:     {vehicle_mgr.total_filtered}")
        print(f"Still active:        {len(vehicle_mgr.active_vehicles)}")
        print(f"Still pending:       {len(vehicle_mgr.pending_vehicles)}")
        print(f"Total tracked:       {len(track_data)}")

        cap.release()
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
