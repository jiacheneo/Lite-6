"""Height-band segmentation and external contours without camera hardware."""
import unittest

import cv2
import numpy as np

from test_board_plane import camera_module


class BlockContourTests(unittest.TestCase):
    def setUp(self):
        self.camera = camera_module.Camera.__new__(camera_module.Camera)
        self.camera.video_frame = np.zeros((120, 180, 3), np.uint8)
        self.camera.workspace_frame = self.camera.video_frame.copy()
        self.camera.workspace_height_mm = np.zeros((120, 180), np.float32)
        self.camera.block_height_band_mm = (15, 60)

    def test_inclusive_band_excludes_table_invalid_and_out_of_range_heights(self):
        c = self.camera
        for index, height in enumerate([15, 60, 14.9, 60.1, np.nan, np.inf, -50]):
            c.workspace_height_mm[50:65, index*25:index*25+15] = height
        before = c.workspace_height_mm.copy()
        c.update_block_contours()
        self.assertEqual(len(c.block_contours), 2)
        self.assertEqual(c.block_mask.dtype, np.uint8)
        self.assertEqual(set(np.unique(c.block_mask)), {0, 255})
        self.assertEqual(c.block_mask[55, 5], 255)
        self.assertEqual(c.block_mask[55, 30], 255)
        self.assertFalse(c.block_mask[:, 50:].any())
        np.testing.assert_array_equal(c.workspace_height_mm, before)

    def test_opening_removes_speckles_and_external_retrieval_ignores_holes(self):
        c = self.camera
        c.workspace_height_mm[45:90, 30:80] = 38
        c.workspace_height_mm[60:70, 45:55] = np.nan
        c.workspace_height_mm[40, 120] = 25
        c.workspace_height_mm[70:72, 130:132] = 25
        c.update_block_contours()
        self.assertEqual(len(c.block_contours), 1)
        self.assertFalse(c.block_mask[:, 100:].any())
        self.assertEqual(c.block_mask[65, 50], 0)
        self.assertEqual(cv2.boundingRect(c.block_contours[0]), (30, 45, 50, 45))
        c.draw_block_contours()
        np.testing.assert_array_equal(c.workspace_frame[80, 30], [128, 128, 128])
        self.assertFalse(c.video_frame.any())

    def test_height_range_changes_and_missing_map_clear_previous_contours(self):
        c = self.camera
        c.workspace_height_mm[50:70, 30:50] = 25
        c.workspace_height_mm[50:70, 50:70] = 38  # Touching regions merge.
        c.update_block_contours()
        self.assertEqual(len(c.block_contours), 1)
        c.block_height_band_mm = (30, 60)
        c.update_block_contours()
        self.assertEqual(cv2.boundingRect(c.block_contours[0]), (50, 50, 20, 20))
        c.workspace_height_mm = None
        c.update_block_contours()
        self.assertEqual(c.block_contours, [])
        self.assertFalse(c.block_mask.any())
        self.assertEqual(c.block_mask_frame.shape, c.video_frame.shape)


if __name__ == '__main__':
    unittest.main()
