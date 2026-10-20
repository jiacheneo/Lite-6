"""World positions from warped block centers and original aligned depth."""
import unittest

import cv2
import numpy as np

from test_depth_bias import synthetic_camera


class BlockPositionTests(unittest.TestCase):
    def setUp(self):
        self.camera, self.world_rays, _, self.bias = synthetic_camera()
        c = self.camera
        c.workspace_homography = np.array([
            [1.2, .1, 8.], [-.05, .9, 4.], [.0002, -.0001, 1.]])
        c.workspace_homography_inv = np.linalg.inv(c.workspace_homography)

    def detect_at(self, u, v):
        c = self.camera
        center = cv2.perspectiveTransform(
            np.array([[[u, v]]], np.float32), c.workspace_homography)[0, 0]
        c.block_detections = [dict(center_px=tuple(center), size='small', color='yellow')]
        c.update_block_positions()
        return c.block_detections[0]

    def test_inverse_warp_depth_units_and_world_transform(self):
        c = self.camera
        u, v = 120, 100
        c.depth_frame_raw[v, u] = 3600  # 900 mm with 0.25 mm raw units.
        # Neither a warped depth nor a flattened height may be used for XYZ.
        c.workspace_depth_frame_raw[:] = 1
        c.workspace_depth_flat_mm = np.full((160, 240), 2., np.float32)
        c.workspace_height_mm = np.full((160, 240), 3., np.float32)
        original = c.depth_frame_raw.copy()
        detection = self.detect_at(u+.4, v+.4)
        expected = c.extrinsic_matrix_inv[:3, 3] + self.world_rays[v, u]*900
        np.testing.assert_allclose(detection['world_position_mm'], expected, atol=1e-6)
        np.testing.assert_allclose(detection['image_center_px'], (u+.4, v+.4), atol=1e-4)
        self.assertEqual(detection['depth_pixel'], (u, v))
        self.assertIsNone(detection['position_reason'])
        np.testing.assert_array_equal(c.depth_frame_raw, original)

    def test_active_bias_applied_once_at_elevated_top(self):
        # Fit using the fixture calibration before changing its display warp.
        c, _, _, _ = synthetic_camera()
        c.correct_depth_bias()
        c.workspace_homography = self.camera.workspace_homography
        c.workspace_homography_inv = self.camera.workspace_homography_inv
        self.camera = c
        u, v = 120, 100
        for top_height in (25., 38., 76.):
            with self.subTest(top_height=top_height):
                depth = (top_height-1000) / self.world_rays[v, u, 2]
                c.depth_frame_raw[v, u] = round((depth+self.bias[v, u])/c.depth_scale_mm)
                detection = self.detect_at(u+.4, v+.4)
                expected = c.extrinsic_matrix_inv[:3, 3] + self.world_rays[v, u]*depth
                np.testing.assert_allclose(detection['world_position_mm'], expected, atol=.3)

    def test_missing_depth_clears_old_position_and_preserves_detection(self):
        c = self.camera
        detection = self.detect_at(120.4, 100.4)
        self.assertIsNotNone(detection['world_position_mm'])
        c.depth_frame_raw[100, 120] = 0
        c.update_block_positions()
        self.assertIsNone(detection['world_position_mm'])
        self.assertEqual(detection['position_reason'], 'invalid_depth')
        self.assertEqual(detection['size'], 'small')
        self.assertEqual(detection['color'], 'yellow')
        self.assertEqual(len(c.block_detections), 1)

    def test_outside_centers_are_not_clamped(self):
        for u, v in [(-.4, 80), (120, -.4), (240.4, 80), (120, 160.4)]:
            with self.subTest(u=u, v=v):
                detection = self.detect_at(u, v)
                self.assertIsNone(detection['world_position_mm'])
                self.assertIsNone(detection['depth_pixel'])
                self.assertEqual(detection['position_reason'], 'outside_image')

    def test_missing_calibration_or_camera_clears_old_fields(self):
        c = self.camera
        for attr, value in [('workspace_homography_inv', None),
                            ('camera_calibrated', False), ('camera_connected', False)]:
            with self.subTest(attr=attr):
                detection = self.detect_at(120.4, 100.4)
                previous = getattr(c, attr)
                setattr(c, attr, value)
                c.update_block_positions()
                self.assertIsNone(detection['world_position_mm'])
                self.assertIsNone(detection['image_center_px'])
                self.assertIsNone(detection['depth_pixel'])
                self.assertEqual(detection['position_reason'], 'uncalibrated')
                setattr(c, attr, previous)
        c.block_detections = []
        c.update_block_positions()
        self.assertEqual(c.block_detections, [])


if __name__ == '__main__':
    unittest.main()
