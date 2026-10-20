"""Task 3.5 persistence, failure accounting, reports and GUI capture (no hardware)."""
import csv
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from PyQt5.QtWidgets import QApplication, QInputDialog
from detector_measurement import (MeasurementSession, trial_plan, make_row, statistics,
                                  confusion, export_report)
from ui.detector_measurement import DetectorMeasurementDialog


def snapshot(detections):
    return dict(captured_at=time.monotonic(), frame_time='2026-10-07T00:00:00+00:00',
                detections=detections, settings={'test_data': True},
                rgb=np.zeros((60, 80, 3), np.uint8),
                preview=np.zeros((60, 80, 3), np.uint8),
                depth=np.full((60, 80), 1000, np.uint16))


def detection(color='red', world=(153., -246., 25.)):
    return dict(color=color, size='small', world_position_mm=world,
                center_px=(30., 30.), orientation_deg=0.)


class MeasurementTests(unittest.TestCase):
    def test_plan_covers_all_combinations_in_requested_order(self):
        plan = trial_plan()
        self.assertEqual(len(plan), 180)
        self.assertEqual([plan[i]['truth_color'] for i in range(0, 18, 3)],
                         ['red', 'orange', 'yellow', 'green', 'blue', 'purple'])
        self.assertEqual(plan[18]['nominal_size_mm'], 38)
        self.assertEqual(plan[36]['point'], 'P2')
        self.assertEqual(plan[-1]['point'], 'P5')
        self.assertEqual(len({(r['point'], r['truth_size'], r['truth_color'], r['repetition'])
                              for r in plan}), 180)

    def test_miss_unknown_and_invalid_depth_keep_distinct_denominators(self):
        trial = trial_plan()[0]
        rows = [make_row(trial, detection(), 1, '', ''),
                make_row(trial, detection('orange'), 1, '', ''),
                make_row(trial, detection('unknown', None), 1, '', ''),
                make_row(trial, None, 0, '', '')]
        stats = statistics(rows)
        self.assertEqual(stats['mean_error_mm'], 5)
        self.assertEqual(stats['valid_positions'], 2)
        self.assertEqual(stats['invalid_positions'], 1)
        self.assertEqual(stats['missed'], 1)
        self.assertEqual(stats['color_accuracy'], .25)
        np.testing.assert_array_equal(confusion(rows)[0], [1, 1, 0, 0, 0, 0, 1, 1])
        invalid = make_row(trial, detection(world=(float('nan'), 2, 3)), 1, '', '')
        self.assertFalse(invalid['position_valid'])

    def test_save_resume_undo_and_rejected_nonconsecutive_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = MeasurementSession.create(Path(tmp) / 'run', 'LED')
            session.record(snapshot([detection()]), 0)
            capture = session.folder / (session.rows[0]['evidence'] + '.npz')
            self.assertEqual(np.load(capture)['depth_raw'].dtype, np.uint16)
            resumed = MeasurementSession.load(session.folder)
            self.assertEqual(resumed.next_trial['trial_id'], 2)
            resumed.undo()
            self.assertEqual(MeasurementSession.load(session.folder).next_trial['trial_id'], 1)
            self.assertTrue(capture.exists())
            resumed.record(snapshot([]), None)
            with (session.folder / 'samples.csv').open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]['predicted_color'], 'missed')
            resumed.data['rows'][0]['trial_id'] = 8
            (session.folder / 'session.json').write_text(json.dumps(resumed.data))
            with self.assertRaises(ValueError):
                MeasurementSession.load(session.folder)

    def test_failed_authoritative_save_keeps_progress_and_csv_can_be_recovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = MeasurementSession.create(Path(tmp) / 'run', 'LED')
            with patch('detector_measurement.atomic_text', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    session.record(snapshot([]), None)
            self.assertEqual(len(session.rows), 0)
            self.assertEqual(len(MeasurementSession.load(session.folder).rows), 0)
            session.record(snapshot([]), None)
            (session.folder / 'samples.csv').write_text('interrupted')
            MeasurementSession.load(session.folder)
            self.assertIn('missed', (session.folder / 'samples.csv').read_text())

    def test_reports_with_all_misses_and_empty_color_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = MeasurementSession.create(Path(tmp) / 'run', 'synthetic test')
            session.record(snapshot([]), None)
            output = export_report(session)
            summary = json.loads((output / 'summary.json').read_text())
            self.assertIsNone(summary['overall']['mean_error_mm'])
            self.assertEqual(summary['overall']['missed'], 1)
            self.assertIn('PARTIAL', (output / 'report.md').read_text())
            for name in ('position_error.png', 'position_error.pdf',
                         'color_confusion.png', 'color_confusion.pdf'):
                self.assertGreater((output / name).stat().st_size, 1000)


class DialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.camera = SimpleNamespace(camera_connected=True)
        self.dialog = DetectorMeasurementDialog(self.camera, output_root=Path(self.tmp.name))
        self.dialog.last_recorded_at = time.monotonic() - 10

    def tearDown(self):
        self.dialog.reject()
        self.assertFalse(self.camera.detector_measurement_enabled)
        self.tmp.cleanup()

    def test_open_is_ready_and_first_record_saves_without_setup(self):
        self.assertIn('Trial 1/180', self.dialog.prompt.text())
        self.assertIsNone(self.dialog.session)
        self.assertEqual(list(Path(self.tmp.name).iterdir()), [])
        self.camera.detector_measurement_snapshot = snapshot([detection()])
        self.dialog.record()
        self.assertEqual(len(self.dialog.session.rows), 1)
        self.assertTrue((self.dialog.session.folder / 'samples.csv').is_file())
        self.assertIn('placement 2/3', self.dialog.prompt.text())
        with self.assertRaises(ValueError):
            self.dialog.record()

    def test_ambiguous_contours_allow_explicit_miss(self):
        self.camera.detector_measurement_snapshot = snapshot([detection(), detection('blue')])
        with patch.object(QInputDialog, 'getItem', return_value=('Missed — none of these is the tested block', True)):
            self.dialog.record()
        self.assertEqual(self.dialog.session.rows[0]['predicted_color'], 'missed')
        self.assertEqual(self.dialog.session.rows[0]['detection_count'], 2)
        self.assertTrue(self.dialog.timer.isActive())

    def test_selection_keeps_original_frame_when_camera_updates(self):
        self.camera.detector_measurement_snapshot = snapshot([detection(), detection('blue')])
        def choose(*args):
            self.camera.detector_measurement_snapshot = snapshot([detection('green')])
            return '#2: blue small', True
        with patch.object(QInputDialog, 'getItem', side_effect=choose):
            self.dialog.record()
        self.assertEqual(self.dialog.session.rows[0]['predicted_color'], 'blue')

    def test_cancel_selection_does_not_advance_or_create_files(self):
        self.camera.detector_measurement_snapshot = snapshot([detection(), detection('blue')])
        with patch.object(QInputDialog, 'getItem', return_value=('', False)):
            self.dialog.record()
        self.assertIsNone(self.dialog.session)
        self.assertEqual(self.dialog.next_trial['trial_id'], 1)
        self.assertTrue(self.dialog.timer.isActive())
        self.assertEqual(list(Path(self.tmp.name).iterdir()), [])

    def test_explicit_miss_ignores_unrelated_contour(self):
        self.camera.detector_measurement_snapshot = snapshot([detection()])
        self.dialog.record_miss()
        self.assertEqual(self.dialog.session.rows[0]['predicted_color'], 'missed')

    def test_csv_failure_after_json_commit_prevents_duplicate(self):
        self.dialog.session = MeasurementSession.create(Path(self.tmp.name) / 'run', 'test')
        self.camera.detector_measurement_snapshot = snapshot([detection()])
        with patch.object(self.dialog.session, 'write_csv', side_effect=OSError('CSV unavailable')):
            with self.assertRaises(OSError):
                self.dialog.record()
        self.assertEqual(len(self.dialog.session.rows), 1)
        self.assertEqual(MeasurementSession.load(self.dialog.session.folder).next_trial['trial_id'], 2)
        with self.assertRaises(ValueError):
            self.dialog.record()

    def test_final_trial_exports_and_undo_returns_to_it(self):
        self.dialog.session = MeasurementSession.create(Path(self.tmp.name) / 'run', 'test')
        rows = [make_row(t, None, 0, '', '') for t in trial_plan()[:-1]]
        self.dialog.session.save(rows)
        self.camera.detector_measurement_snapshot = snapshot([])
        with patch.object(self.dialog, 'export') as export:
            self.dialog.record()
            export.assert_called_once_with()
        self.assertIsNone(self.dialog.next_trial)
        self.dialog.undo()
        self.assertEqual(self.dialog.next_trial['trial_id'], 180)

    def test_stale_frame_is_not_recorded_as_miss(self):
        self.camera.detector_measurement_snapshot = snapshot([])
        self.camera.detector_measurement_snapshot['captured_at'] -= 30
        with self.assertRaises(ValueError):
            self.dialog.record()
        self.assertIsNone(self.dialog.session)


class SnapshotTests(unittest.TestCase):
    def test_published_frame_is_detached_and_uncalibrated_frame_clears_it(self):
        from test_board_plane import camera_module
        camera = camera_module.Camera.__new__(camera_module.Camera)
        camera.detector_measurement_enabled = True
        camera.camera_calibrated = True
        camera.workspace_homography = np.eye(3)
        camera.intrinsic_matrix = camera.extrinsic_matrix = np.eye(3)
        camera.depth_scale_mm = 1.
        camera.workspace_table_plane = (camera.workspace_homography, None, 900.,
                                        np.array([[1, 1], [9, 1], [9, 9], [1, 9]]))
        camera.workspace_height_mm = np.zeros((10, 10))
        camera.active_depth_bias = lambda: None
        camera.block_height_band_mm = (15, 60)
        camera.block_detections = [detection()]
        camera.video_frame = np.zeros((10, 10, 3), np.uint8)
        camera.depth_frame_raw = np.full((10, 10), 900, np.uint16)
        camera.workspace_frame = camera.video_frame.copy()
        camera.publish_detector_measurement(time.monotonic())
        saved = camera.detector_measurement_snapshot
        json.dumps(saved['settings'])
        json.dumps(saved['detections'])
        camera.block_detections[0]['color'] = 'blue'
        camera.depth_frame_raw[:] = 0
        self.assertEqual(saved['detections'][0]['color'], 'red')
        self.assertTrue(np.all(saved['depth'] == 900))
        camera.workspace_table_plane = None
        camera.publish_detector_measurement(time.monotonic())
        self.assertIsNone(camera.detector_measurement_snapshot)

if __name__ == '__main__':
    unittest.main()
