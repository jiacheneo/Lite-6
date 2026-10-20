"""RGB/HSV block colors, clean-frame sampling and invalid-pixel handling."""
import unittest

import cv2
import numpy as np

from test_board_plane import camera_module


def hsv_rgb(hue, saturation=220, value=220):
    return cv2.cvtColor(np.array([[[hue, saturation, value]]], np.uint8),
                        cv2.COLOR_HSV2RGB)[0, 0]


class BlockColorTests(unittest.TestCase):
    def setUp(self):
        c = camera_module.Camera.__new__(camera_module.Camera)
        self.camera = c
        c.video_frame = np.zeros((160, 240, 3), np.uint8)
        c.workspace_homography = np.array([[1., 0, 8], [0, 1., 4], [0, 0, 1.]])
        c.workspace_size = (200, 140)
        c.workspace_pixels_per_mm = 2.
        c.block_height_band_mm = (15, 60)
        c.workspace_height_mm = np.zeros((140, 200), np.float32)
        c.workspace_height_mm[44:94, 58:108] = 25
        # GUI annotations must never be the source of color samples.
        c.workspace_frame = np.full((140, 200, 3), [0, 255, 0], np.uint8)
        c.update_block_contours()
        self.assertEqual(len(c.block_detections), 1)

    def test_six_colors_and_both_red_hue_ranges(self):
        c = self.camera
        for hue, color in [(0, 'red'), (179, 'red'), (15, 'orange'), (19, 'orange'),
                           (20, 'yellow'), (24, 'yellow'), (30, 'yellow'),
                           (60, 'green'), (110, 'blue'), (145, 'purple')]:
            with self.subTest(color=color, hue=hue):
                c.video_frame[40:90, 50:100] = hsv_rgb(hue)
                before = c.video_frame.copy()
                c.update_block_colors()
                d = c.block_detections[0]
                self.assertEqual(d['color'], color)
                self.assertEqual(d['color_sample_count'], 46*46)
                self.assertIsNone(d['color_reason'])
                np.testing.assert_array_equal(d['color_rgb'], hsv_rgb(hue))
                np.testing.assert_array_equal(c.video_frame, before)

    def test_red_wrap_and_highlights_do_not_create_a_false_hue(self):
        c = self.camera
        c.video_frame[40:90, 50:75] = hsv_rgb(0)
        c.video_frame[40:90, 75:100] = hsv_rgb(179)
        c.video_frame[50:60, 60:70] = [255, 255, 255]
        c.update_block_colors()
        self.assertEqual(c.block_detections[0]['color'], 'red')

    def test_erodes_edges_excludes_depth_holes_and_ignores_overlay(self):
        c = self.camera
        c.video_frame[:] = hsv_rgb(110)
        c.video_frame[42:88, 52:98] = hsv_rgb(15)
        c.video_frame[60:70, 70:80] = hsv_rgb(110)
        c.block_mask[64:74, 78:88] = 0
        c.update_block_colors()
        self.assertEqual(c.block_detections[0]['color'], 'orange')
        self.assertEqual(c.block_detections[0]['color_sample_count'], 46*46 - 14*14)

    def test_unknown_retains_geometry_and_clears_old_color(self):
        c = self.camera
        for rgb, reason in [([180, 180, 180], 'low_saturation'), ([20, 0, 0], 'too_dark')]:
            c.video_frame[40:90, 50:100] = rgb
            c.update_block_colors()
            self.assertEqual(len(c.block_detections), 1)
            self.assertEqual(c.block_detections[0]['color'], 'unknown')
            self.assertEqual(c.block_detections[0]['color_reason'], reason)
        c.video_frame[40:90, 50:100] = [255, 0, 0]
        c.update_block_colors()
        self.assertEqual(c.block_detections[0]['color'], 'red')
        c.block_mask[:] = 0
        c.block_mask[60:63, 70:73] = 255
        c.update_block_colors()
        d = c.block_detections[0]
        self.assertEqual(d['color'], 'unknown')
        self.assertEqual(d['color_reason'], 'insufficient_pixels')
        self.assertEqual(d['color_sample_count'], 0)
        self.assertIsNone(d['color_rgb'])
        self.assertIsNone(d['color_hsv'])

    def test_labeled_overlay_can_be_displayed_and_empty_detections_are_valid(self):
        c = self.camera
        c.video_frame[40:90, 50:100] = hsv_rgb(145)
        c.update_block_colors()
        c.draw_block_contours()
        self.assertIsNotNone(c._to_qimage(c.workspace_frame))
        c.workspace_height_mm = None
        c.update_block_contours()
        c.update_block_colors()
        self.assertEqual(c.block_detections, [])


if __name__ == '__main__':
    unittest.main()
