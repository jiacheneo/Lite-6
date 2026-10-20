"""Task 2.8c geometry and report export (mm, fixed-axis XYZ radians)."""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

TCP_RPY_DEG = [180., -90., 0.]
LOCATIONS = [(200, 300), (200, -300), (350, 200), (350, -200)]
TAG_OFFSET = [25., 0., 75.]
# Fixed TCP measurement waypoints, in execution order (mm).
WAYPOINTS = [
    (200, 300, 250), (200, 300, 150), (200, 300, 50),
    (200, -300, 250), (200, -300, 150), (200, -300, 50),
    (350, 200, 250), (350, 200, 150), (350, 200, 50),
    (350, -200, 250), (350, -200, 150), (350, -200, 50),
]
# Optional report labels: replace with 'center'/'edge' for your physical board.
REGIONS = ['unclassified'] * 4


def pose_matrix(pose):
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError("Expected a finite six-element SDK pose.")
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler('xyz', pose[3:]).as_matrix()
    matrix[:3, 3] = pose[:3]
    return matrix


def tag_truth(tcp_pose, tcp_offset, flange_tag_offset=TAG_OFFSET):
    """T_world_tag = T_world_tcp @ inv(T_flange_tcp) @ p_flange_tag."""
    flange = pose_matrix(tcp_pose) @ np.linalg.inv(pose_matrix(tcp_offset))
    return (flange @ np.r_[flange_tag_offset, 1.])[:3]


def measurement_route(start_pose):
    """Go straight to the first point; lift between measurement columns."""
    rpy = np.deg2rad(TCP_RPY_DEG).tolist()
    travel_z = max(start_pose[2], max(point[2] for point in WAYPOINTS))
    route = []
    previous_xy = list(WAYPOINTS[0][:2])
    for x, y, z in WAYPOINTS:
        if [x, y] != previous_xy:
            route.append((previous_xy + [travel_z] + rpy, None))
            route.append(([x, y, travel_z] + rpy, None))
        route.append(([x, y, z] + rpy, LOCATIONS.index((x, y))))
        previous_xy = [x, y]
    return route


def export_report(folder, rows, metadata):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if rows:
        with (folder / 'samples.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    groups = {'all': rows}
    for key in ('region', 'location', 'tcp_target_z_mm'):
        if not rows or key not in rows[0]:
            continue
        for value in sorted({row[key] for row in rows}, key=str):
            groups[f'{key}={value}'] = [row for row in rows if row[key] == value]
    stats = {}
    for name, group in groups.items():
        errors = [row['distance_error_mm'] for row in group]
        stats[name] = {'count': len(group),
                       'mean_error_mm': float(np.mean(errors)) if errors else None,
                       'max_error_mm': float(np.max(errors)) if errors else None}
        heights = [row['truth_z_mm'] for row in group if 'truth_z_mm' in row]
        if heights:
            stats[name]['tag_height_range_mm'] = [float(min(heights)), float(max(heights))]
    with (folder / 'summary.json').open('w') as stream:
        json.dump({'metadata': metadata, 'complete': len(rows) == len(WAYPOINTS),
                   'statistics': stats}, stream, indent=2, allow_nan=False)
