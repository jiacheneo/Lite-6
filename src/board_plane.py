"""Task 2.8b report export, using the camera's unflattened depth pipeline."""
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


def board_mask(shape, board_polygon, exclusions):
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.asarray(board_polygon, dtype=np.int32)], 255)
    for polygon in exclusions:
        cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32)], 0)
    return mask


def error_statistics(samples):
    valid = np.all(np.isfinite(samples[:, 3:]), axis=1)
    z = samples[valid, 5]
    if not z.size:
        raise ValueError("No valid depth samples in the selected board region.")
    return {
        'sample_count': len(samples),
        'valid_count': int(valid.sum()),
        'invalid_count': int((~valid).sum()),
        'mean_signed_z_mm': float(z.mean()),
        'mean_absolute_z_mm': float(np.abs(z).mean()),
        'rms_z_mm': float(np.sqrt(np.mean(z*z))),
        'max_absolute_z_mm': float(np.abs(z).max()),
        'min_z_mm': float(z.min()),
        'max_z_mm': float(z.max()),
    }


def _error_colors(z, limit):
    """Blue (-), white (0), red (+), returned in OpenCV BGR order."""
    t = np.clip(np.asarray(z) / limit, -1, 1)
    colors = np.stack((1 - np.maximum(t, 0), 1 - np.abs(t),
                       1 + np.minimum(t, 0)), axis=-1)
    return (colors * 255).astype(np.uint8)


def export_board_report(camera, polygon, exclusions, output_root, step_px=20):
    """Export a frozen camera snapshot and spatial signed-error map."""
    mask = board_mask(camera.depth_frame_raw.shape, polygon, exclusions)
    samples = camera.sample_board_plane(mask, step_px)
    stats = error_statistics(samples)
    stamp = datetime.now().astimezone()
    output = Path(output_root) / stamp.strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    np.savetxt(output / 'samples.csv', samples, delimiter=',',
               header='u_px,v_px,depth_mm,x_world_mm,y_world_mm,z_error_mm',
               comments='')
    metadata = {
        **stats,
        'exported_at': stamp.isoformat(),
        'reference_plane': 'world Z = 0 mm',
        'step_px': step_px,
        'board_polygon_image_px': np.asarray(polygon).tolist(),
        'excluded_polygons_image_px': [np.asarray(p).tolist() for p in exclusions],
        'intrinsic_matrix': camera.intrinsic_matrix.tolist(),
        'extrinsic_matrix_world_to_camera': camera.extrinsic_matrix.tolist(),
        'depth_scale_mm': float(camera.depth_scale_mm),
        'depth_processing': 'Aligned and filtered depth; no flattening for XYZ.',
        'depth_bias_correction': (camera.active_depth_bias().to_dict()
                                  if camera.active_depth_bias() is not None else None),
        'samples_depth_column': 'Raw camera depth in mm; XYZ uses recorded intrinsics, extrinsics and optional bias correction.',
    }
    (output / 'summary.json').write_text(json.dumps(metadata, indent=2) + '\n')
    np.save(output / 'depth_raw.npy', camera.depth_frame_raw)
    if camera.active_depth_bias() is not None:
        np.save(output / 'depth_bias_mm.npy', camera.active_depth_bias().bias_mm)

    # Draw samples in image coordinates so the selected board and exclusions
    # remain recognizable. Do not interpolate across holes or excluded objects.
    bgr = cv2.cvtColor(camera.video_frame, cv2.COLOR_RGB2BGR)
    view = (bgr * 0.5).astype(np.uint8)
    limit = max(1.0, stats['max_absolute_z_mm'])
    for u, v, _, x, y, z in samples:
        center = (int(u), int(v))
        if np.all(np.isfinite([x, y, z])):
            color = tuple(int(c) for c in _error_colors(z, limit))
            cv2.circle(view, center, max(2, step_px // 3), color, -1)
        else:
            cv2.drawMarker(view, center, (128, 128, 128), cv2.MARKER_TILTED_CROSS, 7, 1)
    cv2.polylines(view, [np.asarray(polygon, np.int32)], True, (0, 255, 0), 2)
    for excluded in exclusions:
        cv2.polylines(view, [np.asarray(excluded, np.int32)], True, (0, 255, 255), 2)
    h, w = view.shape[:2]
    figure = np.full((h + 115, w + 200, 3), 255, np.uint8)
    figure[65:65+h, :w] = view
    bar = _error_colors(np.linspace(limit, -limit, h), limit)
    figure[65:65+h, w+25:w+55] = bar[:, None, :]

    def label(text, xy, scale=0.55):
        cv2.putText(figure, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (25, 25, 25), 1, cv2.LINE_AA)

    label('Board-plane Z error: measured world Z - 0 mm', (12, 25), 0.7)
    label(f"Valid {stats['valid_count']}/{stats['sample_count']}   "
          f"MAE {stats['mean_absolute_z_mm']:.2f} mm   "
          f"RMS {stats['rms_z_mm']:.2f} mm   Max |Z| {stats['max_absolute_z_mm']:.2f} mm",
          (12, 49))
    label(f'+{limit:.2f} mm', (w+65, 80))
    label('0 mm', (w+65, 65+h//2))
    label(f'-{limit:.2f} mm', (w+65, 60+h))
    label('Image pixels: u right, v down. Gray crosses: invalid depth.', (12, h+88))
    label('Green: board. Yellow: excluded.', (12, h+108))
    for name, data in [('board_z_error.png', figure), ('rgb.png', bgr), ('board_mask.png', mask)]:
        if not cv2.imwrite(str(output / name), data):
            raise OSError(f'Could not save {name}')
    return output, stats
