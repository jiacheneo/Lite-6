"""Bias recovery and pipeline integration on independent tilted-camera geometry."""
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from test_board_plane import camera_module
from board_plane import export_board_report


def synthetic_camera():
    c = camera_module.Camera.__new__(camera_module.Camera)
    c.camera_connected = c.camera_calibrated = c.extrinsics_from_tags = True
    c.intrinsic_matrix = np.array([[250., 0, 120], [0, 250., 80], [0, 0, 1.]])
    c.extrinsic_matrix_inv = np.eye(4)
    c.extrinsic_matrix_inv[:3, :3] = Rotation.from_euler('x', 160, degrees=True).as_matrix()
    c.extrinsic_matrix_inv[:3, 3] = [200, 0, 1000]
    c.extrinsic_matrix = np.linalg.inv(c.extrinsic_matrix_inv)
    c.depth_scale_mm = .25
    c.workspace_homography = np.array([[1., 0, 8], [0, 1., 4], [0, 0, 1.]])
    c.workspace_homography_inv = np.linalg.inv(c.workspace_homography)
    c.workspace_table_plane = c.depth_bias_model = None
    c.video_frame = np.full((160, 240, 3), 128, np.uint8)
    v, u = np.indices((160, 240))
    rays = np.stack(((u-120)/250, (v-80)/250, np.ones_like(u)), axis=-1)
    world_rays = rays @ c.extrinsic_matrix_inv[:3, :3].T
    true_depth = -1000 / world_rays[:, :, 2]
    rx, ry = rays[:, :, 0], rays[:, :, 1]
    bias = 7 + 8*rx + 20*ry + 30*rx*ry + 40*ry*ry
    c.depth_frame_raw = np.rint((true_depth + bias)/.25).astype(np.uint16)
    c.depth_frame_raw[30:60, 60:90] -= 200  # Outlier object, 50 mm closer.
    c.depth_frame_raw[80, 140] = 0
    c.update_workspace_depth_frame()
    c.fit_workspace_table_plane([(28, 24), (228, 24), (228, 144), (28, 144)],
                                c.workspace_depth_frame_raw)
    return c, world_rays, true_depth, bias


class DepthBiasTests(unittest.TestCase):
    def test_surface_recovery_rejects_objects_and_repeated_fit_does_not_accumulate(self):
        c, _, _, bias = synthetic_camera()
        original = c.depth_frame_raw.copy()
        info = c.correct_depth_bias()
        self.assertGreater(info['rejected'], 0)
        self.assertLess(info['after_rmse_mm'], .15)
        np.testing.assert_allclose(c.depth_bias_model.bias_mm[20:140, 20:220],
                                   bias[20:140, 20:220], atol=.2)
        first = c.depth_bias_model.bias_mm.copy()
        c.correct_depth_bias()
        np.testing.assert_allclose(c.depth_bias_model.bias_mm, first, equal_nan=True)
        np.testing.assert_array_equal(c.depth_frame_raw, original)

    def test_correction_preserves_depth_coverage_outside_the_fit_corners(self):
        c, _, _, bias = synthetic_camera()
        original = c.depth_frame_raw.copy()
        c.update_workspace_depth_frame()
        valid_before = np.isfinite(c.workspace_height_mm)
        c.correct_depth_bias()
        c.update_workspace_depth_frame()
        # The fitting quadrilateral must not crop originally valid depth.
        np.testing.assert_array_equal(np.isfinite(c.workspace_height_mm), valid_before)
        np.testing.assert_array_equal(np.isfinite(c.workspace_depth_flat_mm), valid_before)
        np.testing.assert_allclose(c.depth_bias_model.bias_mm, bias, atol=.2)
        np.testing.assert_array_equal(c.depth_frame_raw, original)

    def test_world_coordinates_at_table_and_stack_heights_use_bias_once(self):
        c, world_rays, _, bias = synthetic_camera()
        c.correct_depth_bias()
        for u, v in [(0, 0), (40, 40), (120, 80), (200, 120), (239, 159)]:
            for height in [0, 25, 38, 76, 114]:
                expected_depth = (height - 1000) / world_rays[v, u, 2]
                raw = (expected_depth + bias[v, u]) / c.depth_scale_mm
                actual = c.camera_to_world(c.depth_to_camera_point(u, v, raw))
                expected = c.extrinsic_matrix_inv[:3, 3] + world_rays[v, u]*expected_depth
                np.testing.assert_allclose(actual, expected, atol=.2)
        self.assertIsNone(c.depth_to_camera_point(-1, 0, 4000))
        self.assertIsNone(c.depth_to_camera_point(120, 80, 0))

    def test_corrected_height_baseline_and_raw_restoration(self):
        c, _, _, _ = synthetic_camera()
        c.update_workspace_depth_frame()
        uncorrected = c.workspace_height_mm.copy()
        c.correct_depth_bias()
        c.update_workspace_depth_frame()
        _, plane, reference = c.workspace_table_plane[:3]
        # Evaluate at an independent image pixel and its translated workspace pixel.
        u, v = 120, 100
        x, y = u+8, v+4
        corrected_depth = c.depth_frame_raw[v, u]*.25 - c.depth_bias_model.bias_mm[v, u]
        self.assertAlmostEqual(float(c.workspace_height_mm[y, x]),
                               float(plane[y, x] - corrected_depth), delta=.001)
        self.assertAlmostEqual(float(c.workspace_depth_flat_mm[y, x]),
                               reference - float(c.workspace_height_mm[y, x]), places=3)
        self.assertTrue(np.isnan(c.workspace_height_mm[84, 148]))
        self.assertTrue(np.isnan(c.workspace_height_mm[0, 0]))
        c.clear_depth_bias()
        c.update_workspace_depth_frame()
        np.testing.assert_allclose(c.workspace_height_mm, uncorrected, equal_nan=True)

    def test_fit_failure_keeps_existing_model_and_calibration_invalidates_it(self):
        c, _, _, _ = synthetic_camera()
        c.correct_depth_bias()
        model = c.depth_bias_model
        plane = c.workspace_table_plane
        c.depth_frame_raw += 800  # Implausible 200 mm error.
        with self.assertRaisesRegex(ValueError, '100 mm'):
            c.correct_depth_bias()
        self.assertIs(c.depth_bias_model, model)
        self.assertIs(c.workspace_table_plane, plane)
        c.extrinsic_matrix = c.extrinsic_matrix.copy()
        c.extrinsic_matrix[0, 3] += 1
        self.assertIsNone(c.active_depth_bias())
        c.update_workspace_depth_frame()
        self.assertIsNone(c.workspace_height_mm)

    def test_export_records_the_applied_model_and_preserves_raw_capture(self):
        c, _, _, _ = synthetic_camera()
        c.correct_depth_bias()
        with tempfile.TemporaryDirectory() as root:
            folder, stats = export_board_report(c, [(30, 70), (210, 70), (210, 130), (30, 130)], [], root)
            self.assertLess(stats['rms_z_mm'], .2)
            np.testing.assert_array_equal(np.load(folder/'depth_raw.npy'), c.depth_frame_raw)
            np.testing.assert_allclose(np.load(folder/'depth_bias_mm.npy'),
                                       c.depth_bias_model.bias_mm, equal_nan=True)


if __name__ == '__main__':
    unittest.main()
