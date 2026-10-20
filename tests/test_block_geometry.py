"""Metric block geometry and world yaw from independent synthetic world corners."""
import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from test_board_plane import camera_module


def rectangle(width, height, yaw_deg, scale, center=(200, 220)):
    world = np.array([[-width/2, -height/2], [width/2, -height/2],
                      [width/2, height/2], [-width/2, height/2]])
    a = np.radians(yaw_deg)
    rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    world = world @ rotation.T
    pixels = -world[:, ::-1] * scale + center
    return pixels.astype(np.float32).reshape(-1, 1, 2)


class BlockGeometryTests(unittest.TestCase):
    def setUp(self):
        self.camera = camera_module.Camera.__new__(camera_module.Camera)
        c = self.camera
        c.video_frame = np.zeros((480, 640, 3), np.uint8)
        c.workspace_frame = c.video_frame.copy()
        c.workspace_height_mm = np.zeros((480, 640), np.float32)
        c.workspace_pixels_per_mm = 2.
        c.block_height_band_mm = (15, 60)

    def test_metric_sizes_and_world_orientation_across_scales_and_quarter_turns(self):
        c = self.camera
        for scale in [.9, 719/550, 2.]:
            c.workspace_pixels_per_mm = scale
            for size, name in [(25, 'small'), (38, 'large')]:
                for angle in [0, 13, 45, 89, 90, 113]:
                    with self.subTest(scale=scale, size=size, angle=angle):
                        c.block_contours = [rectangle(size, size, angle, scale)]
                        c.update_block_geometry()
                        self.assertEqual(len(c.block_detections), 1)
                        d = c.block_detections[0]
                        self.assertEqual(d['size'], name)
                        self.assertEqual(d['nominal_size_mm'], size)
                        np.testing.assert_allclose(d['side_lengths_mm'], [size, size], atol=.002)
                        np.testing.assert_allclose(d['center_px'], [200, 220], atol=.002)
                        self.assertAlmostEqual(d['area_mm2'], size*size, delta=.02)
                        error = (d['orientation_deg'] - angle + 45) % 90 - 45
                        self.assertAlmostEqual(error, 0, delta=.002)
                        self.assertTrue(0 <= d['orientation_deg'] < 90)

    def test_rejects_noise_merged_blocks_and_poor_rectangle_fill(self):
        c = self.camera
        l_shape = np.array([[0, 0], [38, 0], [38, 10], [10, 10], [10, 38], [0, 38]], np.float32)
        contours = [rectangle(12, 12, 0, 2), rectangle(50, 25, 15, 2),
                    rectangle(76, 38, 30, 2), rectangle(60, 60, 0, 2),
                    (l_shape*2 + 100).reshape(-1, 1, 2),
                    np.array([[[10., 10.]]], np.float32)]
        for contour in contours:
            c.block_contours = [contour]
            c.update_block_geometry()
            self.assertEqual(c.block_detections, [])

    def test_noisy_side_lengths_pass_relaxed_limits_but_merged_pair_still_fails(self):
        c = self.camera
        # Modest edge expansion previously failed the 20% / 1.25 checks.
        for width, height, expected in [(31, 25, 'small'), (44, 34, 'large')]:
            c.block_contours = [rectangle(width, height, 23, 2)]
            c.update_block_geometry()
            self.assertEqual(len(c.block_detections), 1)
            self.assertEqual(c.block_detections[0]['size'], expected)
            self.assertEqual(c.block_rejections, [])
        c.block_contours = [rectangle(50, 25, 0, 2)]
        c.update_block_geometry()
        self.assertEqual(c.block_detections, [])
        self.assertEqual(c.block_rejections[0]['reason'], 'aspect')

    def test_rejection_reasons_include_measured_geometry_and_clear_when_invalid(self):
        c = self.camera
        for width, height, reason in [(12, 12, 'area_small'), (60, 60, 'area_large'),
                                      (40, 25, 'aspect'), (18, 18, 'size')]:
            c.block_contours = [rectangle(width, height, 0, 2)]
            c.update_block_geometry()
            self.assertEqual(c.block_detections, [])
            self.assertEqual(c.block_rejections[0]['reason'], reason)
            np.testing.assert_allclose(c.block_rejections[0]['side_lengths_mm'],
                                       sorted([width, height]), atol=.002)
        # findContours produces integer pixels in the live drawing path.
        c.block_contours = [np.rint(contour).astype(np.int32) for contour in c.block_contours]
        c.draw_block_contours()
        self.assertTrue(np.any(np.all(c.workspace_frame == [255, 165, 0], axis=2)))
        c.workspace_height_mm = None
        c.update_block_contours()
        self.assertEqual(c.block_rejections, [])

    def test_raster_height_pipeline_and_overlay(self):
        c = self.camera
        for size, angle, center in [(25, 0, (120, 220)), (38, 37, (320, 220))]:
            box = np.rint(rectangle(size, size, angle, 2, center)).astype(np.int32)
            cv2.fillConvexPoly(c.workspace_height_mm, box, float(size))
        c.workspace_height_mm[40, 40] = 25
        c.update_block_contours()
        self.assertEqual(len(c.block_detections), 2)
        by_size = {d['size']: d for d in c.block_detections}
        np.testing.assert_allclose(by_size['small']['side_lengths_mm'], [25, 25], atol=.8)
        np.testing.assert_allclose(by_size['large']['side_lengths_mm'], [38, 38], atol=.8)
        self.assertAlmostEqual(by_size['large']['orientation_deg'], 37, delta=1)
        c.draw_block_contours()
        self.assertTrue(np.any(np.all(c.workspace_frame == [0, 255, 0], axis=2)))
        self.assertFalse(c.video_frame.any())
        c.workspace_height_mm = None
        c.update_block_contours()
        self.assertEqual(c.block_detections, [])

    def test_scale_is_recorded_from_calibration_and_missing_scale_gives_no_sizes(self):
        c = self.camera
        c.block_contours = [rectangle(25, 25, 0, 2)]
        c.workspace_pixels_per_mm = None
        c.update_block_geometry()
        self.assertEqual(c.block_detections, [])
        c.tag_world_points = camera_module.TAG_WORLD_POINTS
        c.tag_detections = [SimpleNamespace(tag_id=i, center=[320-y*.5, 350-x*.5])
                            for i, (x, y, _) in c.tag_world_points.items()]
        c._update_workspace_transform()
        x_span = 300 + camera_module.WORKSPACE_MARGIN_TOP_MM + camera_module.WORKSPACE_MARGIN_BOTTOM_MM
        y_span = 600 + camera_module.WORKSPACE_MARGIN_LEFT_MM + camera_module.WORKSPACE_MARGIN_RIGHT_MM
        self.assertAlmostEqual(c.workspace_pixels_per_mm, min(639/y_span, 479/x_span))
        self.assertEqual(c.block_detections, [])


if __name__ == '__main__':
    unittest.main()
