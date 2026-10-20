"""Synthetic Task 2.8b measurements; never start a RealSense device."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import cv2
import numpy as np

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))
from board_plane import board_mask, error_statistics, export_board_report

spec = importlib.util.spec_from_file_location('camera_board_test', SRC / 'camera.py')
camera_module = importlib.util.module_from_spec(spec)
with patch.dict('sys.modules', {'pyrealsense2': ModuleType('pyrealsense2')}):
    spec.loader.exec_module(camera_module)


class BoardPlaneTests(unittest.TestCase):
    def setUp(self):
        self.camera = camera_module.Camera.__new__(camera_module.Camera)
        self.camera.camera_connected = self.camera.camera_calibrated = True
        self.camera.video_frame = np.full((80, 100, 3), 180, np.uint8)
        # Downward-facing camera 1000 mm above the board, half-mm depth units.
        self.camera.depth_frame_raw = np.full((80, 100), 1980, np.uint16)
        self.camera.depth_scale_mm = 0.5
        self.camera.intrinsic_matrix = np.array([[100., 0, 50], [0, 100., 40], [0, 0, 1]])
        self.camera.extrinsic_matrix_inv = np.diag([1., -1., -1., 1.])
        self.camera.extrinsic_matrix_inv[2, 3] = 1000
        self.camera.extrinsic_matrix = np.linalg.inv(self.camera.extrinsic_matrix_inv)
        self.polygon = [(0, 0), (99, 0), (99, 79), (0, 79)]

    def test_z_error_uses_world_transform_and_invalid_depth_is_missing(self):
        self.camera.depth_frame_raw[20, 20] = 0
        mask = board_mask((80, 100), self.polygon, [])
        before = self.camera.depth_frame_raw.copy()
        samples = self.camera.sample_board_plane(mask, 20)
        stats = error_statistics(samples)
        self.assertEqual(stats['sample_count'], 20)
        self.assertEqual(stats['valid_count'], 19)
        self.assertEqual(stats['invalid_count'], 1)
        self.assertAlmostEqual(stats['mean_signed_z_mm'], 10)
        self.assertAlmostEqual(stats['max_absolute_z_mm'], 10)
        np.testing.assert_array_equal(before, self.camera.depth_frame_raw)

    def test_exclusion_removes_object_and_signed_errors_do_not_cancel_mae(self):
        self.camera.depth_frame_raw[:40] = 2040  # world Z = -20 mm
        self.camera.depth_frame_raw[40:] = 1960  # world Z = +20 mm
        self.camera.depth_frame_raw[40, 40] = 1000  # excluded object
        excluded = [[(35, 35), (45, 35), (45, 45), (35, 45)]]
        mask = board_mask((80, 100), self.polygon, excluded)
        samples = self.camera.sample_board_plane(mask, 20)
        self.assertFalse(np.any((samples[:, 0] == 40) & (samples[:, 1] == 40)))
        stats = error_statistics(samples)
        self.assertAlmostEqual(stats['mean_absolute_z_mm'], 20)
        self.assertAlmostEqual(stats['rms_z_mm'], 20)
        self.assertAlmostEqual(stats['max_absolute_z_mm'], 20)
        self.assertLess(abs(stats['mean_signed_z_mm']), 2)

    def test_export_saves_reproducible_report(self):
        with tempfile.TemporaryDirectory() as root:
            output, stats = export_board_report(self.camera, self.polygon, [], root)
            data = np.loadtxt(output / 'samples.csv', delimiter=',', skiprows=1)
            np.testing.assert_allclose(data[:, 5], 10)
            metadata = json.loads((output / 'summary.json').read_text())
            self.assertEqual(metadata['valid_count'], stats['valid_count'])
            self.assertEqual(metadata['depth_scale_mm'], 0.5)
            self.assertIsNotNone(cv2.imread(str(output / 'board_z_error.png')))
            np.testing.assert_array_equal(np.load(output / 'depth_raw.npy'), self.camera.depth_frame_raw)

    def test_no_valid_samples_is_an_error(self):
        self.camera.depth_frame_raw[:] = 0
        with self.assertRaisesRegex(ValueError, 'No valid board samples'):
            self.camera.sample_board_plane(np.ones((80, 100), np.uint8))


if __name__ == '__main__':
    unittest.main()
