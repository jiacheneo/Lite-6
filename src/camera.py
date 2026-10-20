"""
Camera wrapper for the Intel RealSense L515.

Runtime behavior:
- use RealSense intrinsics directly
- provide live RGB/depth for the control station
- expose a student-facing extrinsic calibration entrypoint that uses
  AprilTags + solvePnP at runtime
"""
import time
from itertools import permutations
from copy import copy
from datetime import datetime, timezone

import cv2
import numpy as np
import pyrealsense2 as rs
from pupil_apriltags import Detector
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage

COLOR_W, COLOR_H = 1280, 720
DEPTH_W, DEPTH_H = 1024, 768
TAG_SIZE_MM = 25.0
BLOCK_SIZES_MM = {'small': 25.0, 'large': 38.0}
BLOCK_SIZE_TOLERANCE = 0.25  # Allow noisy/expanded depth edges on both sides.
BLOCK_MAX_ASPECT_RATIO = 1.50
BLOCK_MIN_RECT_FILL = 0.60
# Initial OpenCV HSV thresholds (H: 0..179, S/V: 0..255); tune in lab lighting.
BLOCK_COLOR_HUE_RANGES = {
    'red': ((0, 1), (170, 179)),
    'orange': ((2, 19),),
    'yellow': ((20, 34),),
    'green': ((35, 89),),
    'blue': ((90, 114),),
    'purple': ((115, 169),),
}
BLOCK_COLOR_MIN_SATURATION = 50
BLOCK_COLOR_MIN_VALUE = 40
BLOCK_COLOR_ERODE_PX = 2  # 5x5 erosion removes two pixels around the contour.
BLOCK_COLOR_MIN_PIXELS = 25
# Margins are named by their position in the top-down view (millimeters).
WORKSPACE_MARGIN_TOP_MM = 100.0     # +world X
WORKSPACE_MARGIN_BOTTOM_MM = 150.0  # -world X
WORKSPACE_MARGIN_LEFT_MM = 150.0    # +world Y
WORKSPACE_MARGIN_RIGHT_MM = 150.0   # -world Y
# Known workspace tag centers in millimeters in the robot/world frame.
# Students can adjust these for their physical station.
# TODO: student lab - adjust thest to match new workspace tag locations.
TAG_WORLD_POINTS = {
    1: (100.0,  300.0, 0.0),
    2: (100.0, -300.0, 0.0),
    3: (400.0, -300.0, 0.0),
    4: (400.0, 300.0, 0.0),
}

# Task 2.8a: example grid intersections in mm; adjust to the physical board.
GRID_WORLD_POINTS = [
    (x, y, 0.0)
    for x in range(100, 401, 50)
    for y in range(-300, 301, 50)
]

class Camera:
    """Holds live frame data and owns the RealSense pipeline."""

    def __init__(self):
        self.video_frame = np.zeros((COLOR_H, COLOR_W, 3), dtype=np.uint8)
        self.depth_frame_raw = np.zeros((COLOR_H, COLOR_W), dtype=np.uint16)
        self.workspace_depth_frame_raw = np.zeros((COLOR_H, COLOR_W), dtype=np.uint16)
        self.depth_frame_rgb = np.zeros((COLOR_H, COLOR_W, 3), dtype=np.uint8)
        self.tag_image_frame = np.zeros((COLOR_H, COLOR_W, 3), dtype=np.uint8)
        self.workspace_frame = np.zeros((COLOR_H, COLOR_W, 3), dtype=np.uint8)
        self.block_height_band_mm = (15, 60)
        self.block_mask = np.zeros((COLOR_H, COLOR_W), dtype=np.uint8)
        self.block_contours = []
        self.block_detections = []
        self.detector_measurement_enabled = False
        self.detector_measurement_snapshot = None
        self.block_rejections = []
        self.block_mask_frame = np.zeros((COLOR_H, COLOR_W, 3), dtype=np.uint8)

        # Camera Calibration: extrinsic_matrix maps world-frame homogeneous points
        # to camera-frame homogeneous points; extrinsic_matrix_inv does the reverse.
        # Task 1.6 manual measurement (mm), not an AprilTag calibration.
        self.camera_calibrated = True
        self.intrinsic_matrix = np.eye(3, dtype=np.float64)
        self.extrinsic_matrix = np.array([
            [0.0, -1.0, 0.0, 0.0],
            [-0.963630, 0.0, 0.267238, 224.716],
            [-0.267238, 0.0, -0.963630, 1023.269],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)
        self.extrinsic_matrix_inv = np.linalg.inv(self.extrinsic_matrix)
        # Workspace view: raw image pixel -> top-down workspace pixel.
        self.workspace_homography = None
        # Inverse lets mouse clicks in workspace view map back to raw image pixels.
        self.workspace_homography_inv = None
        self.workspace_size = None  # (width, height), set on calibration
        self.workspace_pixels_per_mm = None
        self.workspace_table_plane = None
        self.depth_bias_model = None
        self.workspace_depth_flat_mm = None
        self.workspace_height_mm = None

        # Click inspection state used by the GUI.
        self.last_click = np.array([0, 0], dtype=int)
        self.new_click = False

        # AprilTag state
        self.tag_world_points = dict(TAG_WORLD_POINTS)
        self.tag_detections = []
        # Atomic reference to matching color/depth/detections for stationary tests.
        self.tag_measurement_snapshot = None
        self.extrinsics_from_tags = False
        self.tag_detect_interval = 6  # ~5 Hz at 30 FPS
        self.depth_scale_mm = 1.0   # raw depth units -> mm; overwritten once the pipeline starts
        self._pipeline_running = False
        self.camera_connected = False

        if len(rs.context().query_devices()) == 0:
            print("WARNING: No RealSense device found - running without video.")
            return

        try:
            self.pipeline = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, 30)
            cfg.enable_stream(rs.stream.depth, DEPTH_W, DEPTH_H, rs.format.z16, 30)
            profile = self.pipeline.start(cfg)
            self.align = rs.align(rs.stream.color)
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale_mm = depth_sensor.get_depth_scale() * 1000.0
            # L515 visual_preset: 0=Custom, 1=Default (deprecated), 2=No Ambient,
            # 3=Low Ambient, 4=Max Range, 5=Short Range.
            # Short Range lowers the firmware depth floor toward the 250 mm spec minimum.
            # Otherwise the depth cut off from 411 mm.
            if depth_sensor.supports(rs.option.visual_preset):
                try:
                    depth_sensor.set_option(rs.option.visual_preset, 5)
                except RuntimeError:
                    pass
            if depth_sensor.supports(rs.option.min_distance):
                depth_sensor.set_option(rs.option.min_distance, 0)

            intr = (
                profile.get_stream(rs.stream.color)
                .as_video_stream_profile()
                .intrinsics
            )
            self.intrinsic_matrix = np.array([
                [intr.fx, 0.0, intr.ppx],
                [0.0, intr.fy, intr.ppy],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)

            self.tag_detector = Detector(families="tagStandard41h12")

            self.spatial_filter = rs.spatial_filter()
            self.spatial_filter.set_option(rs.option.filter_magnitude, 2)
            self.spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
            self.spatial_filter.set_option(rs.option.filter_smooth_delta, 20)

            self.temporal_filter = rs.temporal_filter()
            self.temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.1)
            self.temporal_filter.set_option(rs.option.filter_smooth_delta, 90)
            self.temporal_filter.set_option(rs.option.holes_fill, 3)

            self.hole_filling = rs.hole_filling_filter()
            self.hole_filling.set_option(rs.option.holes_fill, 2)

            self._pipeline_running = True
            self.camera_connected = True
        except RuntimeError as exc:
            print(
                f"WARNING: Camera unavailable ({exc})\n"
                "  -> Connect the L515 via USB 3.2 for video."
            )

    def stop(self):
        if not self._pipeline_running:
            return
        self.pipeline.stop()
        self._pipeline_running = False

    def colorize_depth_frame(self, depth_lo_mm=300.0, depth_hi_mm=1200.0, frame=None, in_mm=False):
        # Jet colormap (blue=near, red=far); pixels outside range masked black.
        if frame is None:
            frame = self.depth_frame_raw
        depth_mm = frame.astype(np.float32)
        if not in_mm:
            depth_mm *= self.depth_scale_mm
        in_range = (depth_mm >= depth_lo_mm) & (depth_mm <= depth_hi_mm)

        normalized = np.zeros(depth_mm.shape, dtype=np.uint8)
        normalized[in_range] = np.clip(
            (depth_mm[in_range] - depth_lo_mm) / (depth_hi_mm - depth_lo_mm) * 255.0,
            0, 255,
        ).astype(np.uint8)

        colorized = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        colorized = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)
        colorized[~in_range] = 0

        self.depth_frame_rgb = colorized

    def draw_tags_in_rgb_image(self):
        self.tag_image_frame = self.video_frame.copy()
        for detection in self.tag_detections:
            corners = np.round(detection.corners).astype(np.int32)
            center = tuple(np.round(detection.center).astype(int))

            cv2.polylines(self.tag_image_frame, [corners], True, (0, 255, 0), 2)
            cv2.circle(self.tag_image_frame, center, 5, (255, 0, 0), -1)
            cv2.putText(
                self.tag_image_frame, str(detection.tag_id),
                (center[0] + 8, center[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )

    def estimate_extrinsics_from_tags(self):
        """Estimate the world-to-camera transform from AprilTag corners."""
        if not self.camera_connected:
            return False, "Camera offline - cannot calibrate."

        matches = [
            detection for detection in self.tag_detections
            if detection.tag_id in self.tag_world_points
        ]
        if len(matches) < 4:
            return False, "Calibration needs at least four known tags."

        world_points = np.array(
            [self.tag_world_points[d.tag_id] for d in matches], dtype=np.float64
        )
        image_points = np.array(
            [d.center for d in matches], dtype=np.float64
        )
        # Centers give an initial pose so each detected corner can be matched
        # to its world corner, regardless of 90-degree tag rotations.
        ok, rvec, tvec = cv2.solvePnP(
            world_points, image_points, self.intrinsic_matrix, None
        )
        if not ok:
            return False, "Could not estimate camera pose from tags."

        half = TAG_SIZE_MM / 2.0
        offsets = np.array([
            [-half, -half, 0], [half, -half, 0],
            [half, half, 0], [-half, half, 0],
        ], dtype=np.float64)
        corner_world = []
        corner_image = []
        for detection, center in zip(matches, world_points):
            world_corners = center + offsets
            projected, _ = cv2.projectPoints(
                world_corners, rvec, tvec, self.intrinsic_matrix, None
            )
            projected = projected.reshape(4, 2)
            detected = np.asarray(detection.corners, dtype=np.float64)
            order = min(
                permutations(range(4)),
                key=lambda p: np.sum((projected - detected[list(p)]) ** 2),
            )
            corner_world.extend(world_corners)
            corner_image.extend(detected[list(order)])

        ok, rvec, tvec = cv2.solvePnP(
            np.asarray(corner_world), np.asarray(corner_image),
            self.intrinsic_matrix, None,
        )
        if not ok:
            return False, "Could not estimate camera pose from tag corners."

        rotation, _ = cv2.Rodrigues(rvec)
        extrinsic = np.eye(4, dtype=np.float64)
        extrinsic[:3, :3] = rotation
        extrinsic[:3, 3] = tvec.ravel()
        self.extrinsic_matrix = extrinsic
        self.extrinsic_matrix_inv = np.linalg.inv(extrinsic)
        self.extrinsics_from_tags = True
        self.camera_calibrated = True
        self._update_workspace_transform()
        print("World-to-camera extrinsic matrix:\n", extrinsic)
        return True, f"Camera calibrated from {len(matches)} tags."

    def active_depth_bias(self):
        model = getattr(self, 'depth_bias_model', None)
        return model if model is not None and model.matches(self) else None

    def correct_depth_bias(self):
        """Fit a frozen frame inside the four selected table corners."""
        from depth_bias import fit_depth_bias
        table = self.workspace_table_plane
        if table is None or table[0] is not self.workspace_homography:
            raise ValueError('Flatten the table with four corners first.')
        if not self.extrinsics_from_tags:
            raise ValueError('Calibrate from board tags first.')
        snapshot = copy(self)
        snapshot.depth_frame_raw = self.depth_frame_raw.copy()
        for field in ('intrinsic_matrix', 'extrinsic_matrix'):
            setattr(snapshot, field, getattr(self, field).copy())
        corners = table[3]
        # Include the corner sampling windows in the fitting region.
        offsets = np.array([[-4, -4], [-4, 4], [4, -4], [4, 4]])
        workspace_points = (corners[:, None, :] + offsets).reshape(-1, 1, 2).astype(np.float32)
        points = cv2.perspectiveTransform(workspace_points, self.workspace_homography_inv)
        mask = np.zeros(snapshot.depth_frame_raw.shape, np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(np.rint(points).astype(np.int32)), 255)
        model = fit_depth_bias(snapshot.depth_frame_raw, mask, snapshot.intrinsic_matrix,
                               snapshot.extrinsic_matrix, snapshot.depth_scale_mm)
        snapshot.depth_bias_model = model
        frozen = cv2.warpPerspective(snapshot.depth_frame_raw, table[0],
                                     snapshot.workspace_output_size(),
                                     flags=cv2.INTER_NEAREST)
        snapshot.fit_workspace_table_plane(corners, frozen)
        if table[0] is not self.workspace_homography or not model.matches(self):
            raise ValueError('Calibration changed. Reopen the table fit and try again.')
        self.depth_bias_model = model
        self.workspace_table_plane = snapshot.workspace_table_plane
        return model.diagnostics

    def clear_depth_bias(self):
        table = self.workspace_table_plane
        self.depth_bias_model = None
        if table is not None and table[0] is self.workspace_homography:
            self.fit_workspace_table_plane(table[3], table[4])

    def depth_to_camera_point(self, x, y, depth_raw):
        """Back-project with the pinhole model; return None for unavailable depth.

        Depth is aligned to the color image and measures camera Z, not range.
        Lens distortion is not corrected in this manual-calibration model.
        """
        if not self.camera_connected or not np.all(np.isfinite([x, y, depth_raw])):
            return None
        z = float(depth_raw) * self.depth_scale_mm
        if not np.isfinite(z) or z <= 0:
            return None
        model = self.active_depth_bias()
        if model is not None:
            u, v = int(round(x)), int(round(y))
            h, w = model.bias_mm.shape
            if not (0 <= u < w and 0 <= v < h):
                return None
            z -= float(model.bias_mm[v, u])
            if not np.isfinite(z) or z <= 0:
                return None
        ray = np.linalg.solve(self.intrinsic_matrix, np.array([x, y, 1.0]))
        return ray * z

    def camera_to_world(self, camera_point):
        """Convert camera XYZ (mm) to world XYZ (mm), or None if unavailable."""
        if not self.camera_calibrated or camera_point is None:
            return None
        point = np.asarray(camera_point, dtype=np.float64)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            return None
        return (self.extrinsic_matrix_inv @ np.append(point, 1.0))[:3]

    def world_to_image(self, world_point):
        """Project world XYZ (mm) to float color pixels (u, v), without distortion.

        Return None when unavailable or behind the camera. Off-image pixels are
        retained so manual projection errors can still be measured.
        """
        if not self.camera_connected or not self.camera_calibrated:
            return None
        point = np.asarray(world_point, dtype=np.float64)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            return None
        camera_point = (self.extrinsic_matrix @ np.append(point, 1.0))[:3]
        if camera_point[2] <= 0:
            return None
        pixel = self.intrinsic_matrix @ camera_point
        return pixel[:2] / pixel[2]

    def draw_grid_in_rgb_image(self, world_points):
        """Return an RGB copy with projected world grid points (mm) marked."""
        frame = self.video_frame.copy()
        for world_point in world_points:
            pixel = self.world_to_image(world_point)
            if pixel is None:
                continue
            center = tuple(int(value) for value in np.rint(pixel))
            cv2.drawMarker(
                frame, center, (0, 255, 0),
                markerType=cv2.MARKER_CROSS, markerSize=12,
                thickness=1, line_type=cv2.LINE_AA,
            )
        return frame

    def image_to_world(self, x, y):
        """Pixel (x, y) -> world-frame (X, Y, Z) in mm using live depth."""
        if not self.camera_connected or not self.camera_calibrated:
            return None
        if not np.all(np.isfinite([x, y])):
            return None
        height, width = self.depth_frame_raw.shape
        if not (0 <= x < width and 0 <= y < height):
            return None
        # Use the same integer pixel for both the depth sample and its ray.
        ix, iy = int(x), int(y)
        camera_point = self.depth_to_camera_point(ix, iy, self.depth_frame_raw[iy, ix])
        return self.camera_to_world(camera_point)

    def sample_board_plane(self, board_mask, step_px=20):
        """Sample an empty-board mask: rows are u, v, depth_mm, X, Y, Z.

        Z is the signed error against the known world plane Z=0. Missing
        depth produces NaN XYZ, never zero error. Use a frozen camera snapshot
        when collecting a report so every sample uses the same frame/pose.
        """
        if not self.camera_connected or not self.camera_calibrated:
            raise ValueError("Connect and calibrate the camera first.")
        if board_mask.shape != self.depth_frame_raw.shape or step_px < 1:
            raise ValueError("Invalid board mask or sampling interval.")
        rows = []
        height, width = board_mask.shape
        for v in range(0, height, step_px):
            for u in range(0, width, step_px):
                if not board_mask[v, u]:
                    continue
                point = self.image_to_world(u, v)
                xyz = point if point is not None else (np.nan, np.nan, np.nan)
                depth_mm = float(self.depth_frame_raw[v, u]) * self.depth_scale_mm
                rows.append((u, v, depth_mm, *xyz))
        samples = np.asarray(rows, dtype=np.float64).reshape(-1, 6)
        if not len(samples) or not np.any(np.all(np.isfinite(samples[:, 3:]), axis=1)):
            raise ValueError("No valid board samples. Check the selection and depth image.")
        return samples

    def workspace_pixel_to_image(self, x_px, y_px):
        """Map a workspace-view pixel back to the original image pixel."""
        if self.workspace_homography_inv is None:
            return None
        point = cv2.perspectiveTransform(
            np.array([[[x_px, y_px]]], dtype=np.float32),
            self.workspace_homography_inv,
        )
        return tuple(point[0, 0])

    def _update_workspace_transform(self):
        """Capture the workspace homography when Calibrate succeeds."""
        detections = [
            d for d in self.tag_detections if d.tag_id in self.tag_world_points
        ]
        if len(detections) < 4:
            return

        source = np.array([d.center for d in detections], dtype=np.float32)
        world = np.array(
            [self.tag_world_points[d.tag_id][:2] for d in detections],
            dtype=np.float32,
        )
        low = world.min(axis=0) - [WORKSPACE_MARGIN_BOTTOM_MM, WORKSPACE_MARGIN_RIGHT_MM]
        high = world.max(axis=0) + [WORKSPACE_MARGIN_TOP_MM, WORKSPACE_MARGIN_LEFT_MM]
        height, width = self.video_frame.shape[:2]
        scale = min((width - 1) / (high[1] - low[1]),
                    (height - 1) / (high[0] - low[0]))
        self.workspace_pixels_per_mm = float(scale)
        self.block_detections = []
        self.block_rejections = []
        # Crop the output canvas to these bounds rather than exposing extra
        # world coordinates to fill the original 16:9 image. Keep one px/mm scale.
        self.workspace_size = (int(round((high[1] - low[1]) * scale)) + 1,
                               int(round((high[0] - low[0]) * scale)) + 1)
        # Image horizontal follows -world Y; vertical follows -world X.
        destination = np.column_stack((
            (high[1] - world[:, 1]) * scale,
            (high[0] - world[:, 0]) * scale,
        )).astype(np.float32)

        self.workspace_table_plane = None
        self.depth_bias_model = None
        self.workspace_depth_flat_mm = None
        self.workspace_height_mm = None
        self.workspace_homography, _ = cv2.findHomography(source, destination)
        self.workspace_homography_inv = np.linalg.inv(self.workspace_homography)

    def workspace_output_size(self):
        """Return the shared canvas size for all top-down images."""
        size = getattr(self, 'workspace_size', None)
        if size is not None and self.workspace_homography is not None:
            return size
        height, width = self.video_frame.shape[:2]
        return width, height

    def update_workspace_frame(self, frame=None):
        """Apply the last explicit calibration, or show RGB before calibration."""
        if frame is None:
            frame = self.video_frame
        if self.workspace_homography is None:
            self.workspace_frame = frame.copy()
        else:
            self.workspace_frame = cv2.warpPerspective(
                frame, self.workspace_homography, self.workspace_output_size()
            )

    def fit_workspace_table_plane(self, corners, depth_raw):
        """Fit depth = a*u + b*v + c from four 9x9 table-window medians.

        Input is a frozen top-down depth image in sensor units. The fitted
        plane is used only for segmentation/display, never for world XYZ.
        """
        homography = self.workspace_homography
        if homography is None:
            raise ValueError('Calibrate the camera first.')
        points = np.asarray(corners, dtype=float)
        if points.shape != (4, 2) or not np.all(np.isfinite(points)):
            raise ValueError('Select four table corners.')
        points = np.rint(points).astype(int)
        design = np.column_stack((points, np.ones(4)))
        if len(np.unique(points, axis=0)) != 4 or np.linalg.matrix_rank(design) < 3:
            raise ValueError('Select four distinct corners spanning the table.')
        h, w = depth_raw.shape
        corrected = self._workspace_depth_mm(depth_raw, homography)
        medians = []
        for u, v in points:
            if not (4 <= u < w - 4 and 4 <= v < h - 4):
                raise ValueError('Select corners at least four pixels inside the image.')
            patch = corrected[v-4:v+5, u-4:u+5]
            valid = patch[(patch > 0) & np.isfinite(patch)]
            if valid.size < 41:
                raise ValueError('Not enough valid depth near a corner. Select a nearby table point.')
            medians.append(float(np.median(valid)))
        coefficients = np.linalg.lstsq(design, medians, rcond=None)[0]
        v, u = np.indices(depth_raw.shape, dtype=np.float32)
        plane = coefficients[0] * u + coefficients[1] * v + coefficients[2]
        # Publish the matching plane/reference together for the video thread.
        self.workspace_table_plane = (homography, plane, float(np.mean(medians)),
                                      points.copy(), depth_raw.copy(), self.active_depth_bias())

    def _workspace_depth_mm(self, depth_raw, homography):
        """Subtract the fitted camera-Z bias once, keeping raw depth intact."""
        depth_mm = depth_raw.astype(np.float32) * self.depth_scale_mm
        model = self.active_depth_bias()
        if model is not None:
            h, w = depth_raw.shape
            bias = cv2.warpPerspective(model.bias_mm, homography, (w, h),
                                       flags=cv2.INTER_NEAREST,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=float('nan'))
            depth_mm -= bias
        return np.where((depth_raw > 0) & (depth_mm > 0), depth_mm, np.nan)

    def update_workspace_depth_frame(self):
        """Warp depth after calibration and optionally remove the table ramp."""
        homography = self.workspace_homography
        if homography is None:
            depth = self.depth_frame_raw.copy()
        else:
            depth = cv2.warpPerspective(
                self.depth_frame_raw, homography, self.workspace_output_size(),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
        self.workspace_depth_frame_raw = depth
        model = self.workspace_table_plane
        if (model is None or model[0] is not homography
                or model[5] is not self.active_depth_bias()):
            self.workspace_depth_flat_mm = None
            self.workspace_height_mm = None
            return
        _, plane, reference = model[:3]
        depth_mm = self._workspace_depth_mm(depth, homography)
        height_mm = np.where(depth > 0, plane - depth_mm, np.nan)
        self.workspace_height_mm = height_mm
        self.workspace_depth_flat_mm = reference - height_mm

    def update_block_contours(self):
        """Threshold valid heights, remove speckles, and find external contours.

        Results are candidates in workspace pixels, followed by size/shape
        filtering. Touching blocks can form a single contour.
        """
        height_mm = self.workspace_height_mm
        self.block_detections = []
        self.block_rejections = []
        if height_mm is None:
            width, height = self.workspace_output_size()
            self.block_mask = np.zeros((height, width), dtype=np.uint8)
            self.block_contours = []
            self.block_mask_frame = cv2.cvtColor(self.block_mask, cv2.COLOR_GRAY2RGB)
            cv2.putText(self.block_mask_frame, 'Calibrate, then Flatten Table first',
                        (16, 28), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
            return
        low, high = self.block_height_band_mm
        selected = np.isfinite(height_mm) & (height_mm >= low) & (height_mm <= high)
        mask = selected.astype(np.uint8) * 255
        self.block_mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8),
            borderType=cv2.BORDER_CONSTANT, borderValue=0,
        )
        self.block_contours, _ = cv2.findContours(
            self.block_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.block_mask_frame = cv2.cvtColor(self.block_mask, cv2.COLOR_GRAY2RGB)
        self.update_block_geometry()

    def update_block_geometry(self):
        """Measure blocks and retain the reason each other candidate was rejected."""
        self.block_detections = []
        self.block_rejections = []
        scale = getattr(self, 'workspace_pixels_per_mm', None)
        if self.workspace_height_mm is None or scale is None or scale <= 0:
            return
        min_area = 0.5 * min(BLOCK_SIZES_MM.values())**2
        max_area = 1.5 * max(BLOCK_SIZES_MM.values())**2
        detections, rejections = [], []
        for contour in self.block_contours:
            area_mm2 = cv2.contourArea(contour) / scale**2
            rect = cv2.minAreaRect(contour)
            sides = np.sort(np.asarray(rect[1], dtype=float) / scale)
            box = cv2.boxPoints(rect)
            aspect = float(sides[1] / sides[0]) if sides[0] > 0 else float('inf')
            fill = float(area_mm2 / np.prod(sides)) if sides[0] > 0 else 0.0
            candidate = dict(
                center_px=tuple(float(x) for x in rect[0]),
                side_lengths_mm=tuple(float(x) for x in sides),
                area_mm2=float(area_mm2), aspect_ratio=aspect, rect_fill=fill,
                box_px=box, contour=contour,
            )
            matches = [(name, nominal) for name, nominal in BLOCK_SIZES_MM.items()
                       if np.all(np.abs(sides - nominal) <= BLOCK_SIZE_TOLERANCE * nominal)]
            reason = None
            if area_mm2 < min_area:
                reason = ('area_small', f'area {area_mm2:.0f} < {min_area:.0f} mm2')
            elif area_mm2 > max_area:
                reason = ('area_large', f'area {area_mm2:.0f} > {max_area:.0f} mm2')
            elif aspect > BLOCK_MAX_ASPECT_RATIO:
                reason = ('aspect', f'aspect {aspect:.2f} > {BLOCK_MAX_ASPECT_RATIO:.2f}')
            elif fill < BLOCK_MIN_RECT_FILL:
                reason = ('fill', f'fill {fill:.2f} < {BLOCK_MIN_RECT_FILL:.2f}')
            elif not matches:
                reason = ('size', f'size outside +/-{BLOCK_SIZE_TOLERANCE*100:.0f}%')
            if reason is not None:
                rejections.append(dict(candidate, reason=reason[0], reason_text=reason[1]))
                continue
            size, nominal = min(matches, key=lambda item: abs(sides.mean() - item[1]))
            du, dv = box[1] - box[0]
            # Workspace u follows -world Y, v follows -world X.
            yaw = float(np.degrees(np.arctan2(-du, -dv)) % 90.0)
            if np.isclose(yaw, 90.0, atol=1e-4):
                yaw = 0.0
            detections.append(dict(candidate, size=size, nominal_size_mm=nominal,
                                   orientation_deg=yaw))
        # A spatial order makes the list predictable; these are not tracked IDs.
        self.block_detections = sorted(detections, key=lambda d: (d['center_px'][1], d['center_px'][0]))
        self.block_rejections = rejections

    def update_block_colors(self):
        """Classify median RGB inside eroded, depth-supported block contours.

        Warp the clean RGB frame, never the annotated Workspace image. Taking
        the median in RGB before HSV conversion avoids averaging red hues near
        0 and 179 into an unrelated hue.
        """
        for detection in self.block_detections:
            detection.update(color='unknown', color_rgb=None, color_hsv=None,
                             color_sample_count=0, color_reason='insufficient_pixels')
        if not self.block_detections or self.workspace_homography is None:
            return
        rgb = cv2.warpPerspective(self.video_frame, self.workspace_homography,
                                  self.workspace_output_size())
        radius = BLOCK_COLOR_ERODE_PX
        kernel = np.ones((2*radius+1, 2*radius+1), dtype=np.uint8)
        height, width = rgb.shape[:2]
        for detection in self.block_detections:
            contour = detection['contour']
            x, y, w, h = cv2.boundingRect(contour)
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(width, x+w), min(height, y+h)
            if x1 <= x0 or y1 <= y0:
                continue
            mask = np.zeros((y1-y0, x1-x0), dtype=np.uint8)
            local_contour = np.rint(contour - [x0, y0]).astype(np.int32)
            cv2.drawContours(mask, [local_contour], -1, 255, cv2.FILLED)
            # External contours enclose holes too: retain only valid mask pixels.
            mask &= self.block_mask[y0:y1, x0:x1]
            mask = cv2.erode(mask, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
            pixels = rgb[y0:y1, x0:x1][mask != 0]
            detection['color_sample_count'] = len(pixels)
            if len(pixels) < BLOCK_COLOR_MIN_PIXELS:
                continue
            median_rgb = np.rint(np.median(pixels, axis=0)).astype(np.uint8)
            hsv = cv2.cvtColor(median_rgb.reshape(1, 1, 3), cv2.COLOR_RGB2HSV)[0, 0]
            detection['color_rgb'] = tuple(int(v) for v in median_rgb)
            detection['color_hsv'] = tuple(int(v) for v in hsv)
            hue, saturation, value = hsv
            if value < BLOCK_COLOR_MIN_VALUE:
                detection['color_reason'] = 'too_dark'
            elif saturation < BLOCK_COLOR_MIN_SATURATION:
                detection['color_reason'] = 'low_saturation'
            else:
                detection['color_reason'] = 'unmatched_hue'
                for color, ranges in BLOCK_COLOR_HUE_RANGES.items():
                    if any(low <= hue <= high for low, high in ranges):
                        detection['color'] = color
                        detection['color_reason'] = None
                        break

    def update_block_positions(self):
        """Back-project each visible top-face center using original aligned depth.

        Workspace pixels must first pass through the inverse homography. The
        existing image_to_world path applies depth units and optional bias once,
        then camera intrinsics and extrinsics; flattened depth is not camera Z.
        """
        for detection in self.block_detections:
            detection.update(world_position_mm=None, image_center_px=None,
                             depth_pixel=None, position_reason='uncalibrated')
            if (self.workspace_homography_inv is None
                    or not self.camera_calibrated or not self.camera_connected):
                continue
            pixel = self.workspace_pixel_to_image(*detection['center_px'])
            if pixel is None or not np.all(np.isfinite(pixel)):
                detection['position_reason'] = 'invalid_projection'
                continue
            x, y = (float(value) for value in pixel)
            detection['image_center_px'] = (x, y)
            height, width = self.depth_frame_raw.shape
            if not (0 <= x < width and 0 <= y < height):
                detection['position_reason'] = 'outside_image'
                continue
            # Match image_to_world: use the same integer pixel for depth and ray.
            detection['depth_pixel'] = (int(x), int(y))
            world = self.image_to_world(x, y)
            if world is None or not np.all(np.isfinite(world)):
                detection['position_reason'] = 'invalid_depth'
                continue
            detection['world_position_mm'] = tuple(float(value) for value in world)
            detection['position_reason'] = None

    def publish_detector_measurement(self, captured_at):
        """Publish a complete frame atomically, only while the measurement UI is open."""
        if not self.detector_measurement_enabled:
            return
        from copy import deepcopy
        table = self.workspace_table_plane
        if (not self.camera_calibrated or table is None
                or table[0] is not self.workspace_homography or self.workspace_height_mm is None):
            self.detector_measurement_snapshot = None
            return
        bias = self.active_depth_bias()
        settings = dict(
            intrinsics=self.intrinsic_matrix.tolist(), extrinsics=self.extrinsic_matrix.tolist(),
            homography=self.workspace_homography.tolist(), depth_scale_mm=self.depth_scale_mm,
            table_corners_px=table[3].tolist(), table_reference_mm=table[2],
            height_band_mm=list(self.block_height_band_mm),
            depth_bias=bias.to_dict() if bias is not None else None,
            sizes_mm=BLOCK_SIZES_MM, size_tolerance=BLOCK_SIZE_TOLERANCE,
            max_aspect_ratio=BLOCK_MAX_ASPECT_RATIO, min_rect_fill=BLOCK_MIN_RECT_FILL,
            hue_ranges=BLOCK_COLOR_HUE_RANGES, min_saturation=BLOCK_COLOR_MIN_SATURATION,
            min_value=BLOCK_COLOR_MIN_VALUE, erode_px=BLOCK_COLOR_ERODE_PX,
            min_color_pixels=BLOCK_COLOR_MIN_PIXELS)
        fields = ('center_px', 'box_px', 'size', 'side_lengths_mm', 'orientation_deg',
                  'color', 'color_hsv', 'color_rgb', 'color_reason', 'world_position_mm',
                  'position_reason', 'image_center_px')
        detections = [{key: (value.tolist() if isinstance(value, np.ndarray) else value)
                       for key in fields if (value := d.get(key)) is not None}
                      for d in self.block_detections]
        self.detector_measurement_snapshot = dict(
            captured_at=captured_at, frame_time=datetime.now(timezone.utc).isoformat(),
            detections=deepcopy(detections), settings=deepcopy(settings),
            rgb=self.video_frame.copy(), depth=self.depth_frame_raw.copy(),
            preview=self.workspace_frame.copy())

    def draw_block_contours(self):
        """Show accepted blocks in green and explain rejected candidates in orange."""
        if self.workspace_height_mm is None:
            return
        cv2.drawContours(self.workspace_frame, self.block_contours, -1, (128, 128, 128), 1)
        detections = getattr(self, 'block_detections', [])
        h, w = self.workspace_frame.shape[:2]

        def text(label, x, y, scale=.5, color=(255, 255, 255)):
            text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
            point = (max(0, min(x, w-text_width-2)), max(14, min(y, h-4)))
            cv2.putText(self.workspace_frame, label, point,
                        cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3)
            cv2.putText(self.workspace_frame, label, point,
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)

        for rejected in getattr(self, 'block_rejections', []):
            # Keep tiny speckles from covering the workspace in diagnostic text.
            if rejected['area_mm2'] < 100:
                continue
            box = np.rint(rejected['box_px']).astype(np.int32)
            cv2.polylines(self.workspace_frame, [box], True, (255, 165, 0), 1)
            x, y = int(box[:, 0].min()), int(box[:, 1].max()) + 16
            text(rejected['reason_text'], x, y, color=(255, 165, 0))
            a, b = rejected['side_lengths_mm']
            text(f'{a:.1f} x {b:.1f} mm', x, y+17, color=(255, 165, 0))

        for detection in detections:
            box = np.rint(detection['box_px']).astype(np.int32)
            center = np.rint(detection['center_px']).astype(int)
            cv2.polylines(self.workspace_frame, [box], True, (0, 255, 0), 2)
            yaw = np.radians(detection['orientation_deg'])
            length = detection['side_lengths_mm'][0] * self.workspace_pixels_per_mm * .45
            tip = np.rint(center + length * np.array([-np.sin(yaw), -np.cos(yaw)])).astype(int)
            cv2.line(self.workspace_frame, tuple(center), tuple(tip), (255, 255, 0), 2)
            cv2.circle(self.workspace_frame, tuple(center), 3, (255, 255, 0), -1)
            a, b = detection['side_lengths_mm']
            x = int(box[:, 0].min())
            y = max(14, min(int(box[:, 1].max()) + 16, h-38))
            text(f"{detection.get('color', 'unknown')} {detection['size']} {detection['nominal_size_mm']:g} mm | {detection['orientation_deg']:.1f} deg", x, y)
            hsv = detection.get('color_hsv')
            hue_label = f' | H={hsv[0]}' if hsv is not None else ''
            text(f'{a:.1f} x {b:.1f} mm{hue_label}', x, y+17)
            world = detection.get('world_position_mm')
            position_label = ('XYZ unavailable' if world is None else
                              f'XYZ ({world[0]:.1f}, {world[1]:.1f}, {world[2]:.1f}) mm')
            text(position_label, x, y+34)
        low, high = self.block_height_band_mm
        text(f'Height {low}-{high} mm | Contours: {len(self.block_contours)} | Blocks: {len(detections)}',
             16, 28, .65)

    def _to_qimage(self, frame):
        """Convert a NumPy RGB image into a Qt image for display."""
        try:
            h, w = frame.shape[:2]
            return QImage(frame.data, w, h, frame.strides[0], QImage.Format_RGB888).copy()
        except Exception:
            return None


class VideoThread(QThread):
    """Polls the RealSense pipeline at ~30 Hz and emits Qt signals for the GUI."""

    updateFrame = pyqtSignal(QImage, QImage, QImage, QImage, QImage)

    def __init__(self, camera, parent=None):
        QThread.__init__(self, parent=parent)
        self.camera = camera
        self._running = True
        self._frame_counter = 0

    def stop(self):
        self._running = False

    def run(self):
        if not self.camera.camera_connected:
            while self._running:
                time.sleep(0.5)
            return

        while self._running:
            start = time.time()

            try:
                frames = self.camera.pipeline.wait_for_frames(timeout_ms=100)
            except RuntimeError:
                if not self._running:
                    break
                continue

            aligned = self.camera.align.process(frames)
            captured_at = time.monotonic()
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            depth_frame = self.camera.spatial_filter.process(depth_frame)
            depth_frame = self.camera.temporal_filter.process(depth_frame)
            depth_frame = self.camera.hole_filling.process(depth_frame)

            bgr = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data())

            self.camera.video_frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self.camera.depth_frame_raw = depth
            self.camera.update_workspace_depth_frame()
            flat = self.camera.workspace_depth_flat_mm
            self.camera.colorize_depth_frame(
                frame=self.camera.workspace_depth_frame_raw if flat is None else flat,
                in_mm=flat is not None,
            )

            if self._frame_counter % self.camera.tag_detect_interval == 0:
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                self.camera.tag_detections = self.camera.tag_detector.detect(gray)
                self.camera.tag_measurement_snapshot = (
                    captured_at, self.camera.video_frame.copy(),
                    self.camera.depth_frame_raw.copy(), tuple(self.camera.tag_detections),
                )
            self._frame_counter += 1

            self.camera.draw_tags_in_rgb_image()
            grid_frame = self.camera.draw_grid_in_rgb_image(GRID_WORLD_POINTS)
            self.camera.update_workspace_frame(grid_frame)
            self.camera.update_block_contours()
            self.camera.update_block_colors()
            self.camera.update_block_positions()
            self.camera.draw_block_contours()
            self.camera.publish_detector_measurement(captured_at)

            rgb_qt   = self.camera._to_qimage(grid_frame)
            depth_qt = self.camera._to_qimage(self.camera.depth_frame_rgb)
            tag_qt   = self.camera._to_qimage(self.camera.tag_image_frame)
            workspace_qt = self.camera._to_qimage(self.camera.workspace_frame)

            mask_qt = self.camera._to_qimage(self.camera.block_mask_frame)
            if rgb_qt and depth_qt and tag_qt and workspace_qt and mask_qt:
                self.updateFrame.emit(rgb_qt, depth_qt, tag_qt, workspace_qt, mask_qt)

            elapsed = time.time() - start
            time.sleep(max(0.0, 1 / 30 - elapsed))


if __name__ == '__main__':
    import sys

    from PyQt5.QtCore import Qt, pyqtSlot
    from PyQt5.QtGui import QPixmap
    from PyQt5.QtWidgets import QApplication, QLabel

    app = QApplication(sys.argv)
    camera = Camera()
    print(f"Intrinsic matrix:\n{camera.intrinsic_matrix}")

    wins = {}
    for title in ("RGB", "Depth", "Tags", "Workspace", "Mask"):
        label = QLabel(title)
        label.setWindowTitle(title)
        label.setAlignment(Qt.AlignCenter)
        label.resize(640, 360)
        label.show()
        wins[title] = label

    @pyqtSlot(QImage, QImage, QImage, QImage, QImage)
    def on_frame(rgb, depth, tags, workspace, mask):
        for label, image in zip(wins.values(), (rgb, depth, tags, workspace, mask)):
            label.setPixmap(
                QPixmap.fromImage(image).scaled(
                    label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )

    thread = VideoThread(camera)
    thread.updateFrame.connect(on_frame)
    thread.start()

    result = app.exec_()
    thread.stop()
    thread.wait(500)
    camera.stop()
    sys.exit(result)
