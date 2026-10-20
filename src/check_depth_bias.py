"""Offline spatial holdout check of a saved Task 2.8b empty-board capture."""
import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from depth_bias import depth_geometry, fit_depth_bias, z_statistics


def check_report(source, output_root):
    source = Path(source)
    metadata = json.loads((source / 'summary.json').read_text())
    depth = np.load(source / 'depth_raw.npy', allow_pickle=False)
    mask = cv2.imread(str(source / 'board_mask.png'), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != depth.shape:
        raise ValueError('Missing or incompatible board_mask.png.')
    intrinsic = np.array(metadata['intrinsic_matrix'])
    extrinsic = np.array(metadata['extrinsic_matrix_world_to_camera'])
    scale = metadata['depth_scale_mm']
    v, u = np.indices(depth.shape)
    training_tiles = ((u//100 + v//100) % 2 == 0)
    model = fit_depth_bias(depth, np.where(training_tiles, mask, 0),
                           intrinsic, extrinsic, scale)
    selected = ((mask != 0) & ~training_tiles & (u % 20 == 0)
                & (v % 20 == 0) & (depth > 0))
    pixels = np.c_[u[selected], v[selected]]
    measured = depth[selected].astype(float) * scale
    rays, slope, origin = depth_geometry(pixels, intrinsic, extrinsic)
    supported = np.array([model.bias_mm(x, y) is not None for x, y in pixels])
    pixels, rays = pixels[supported], rays[supported]
    measured, slope = measured[supported], slope[supported]
    bias = np.c_[np.ones(len(rays)), rays[:, :2]] @ model.coefficients
    before = measured * slope + origin
    after = (measured - bias) * slope + origin
    output = Path(output_root) / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    summary = dict(source=str(source.resolve()),
                   validation='Held-out 100-pixel spatial tiles of ONE frame; not new captures or elevated targets.',
                   test_points_outside_support=int((~supported).sum()),
                   before=z_statistics(before), after=z_statistics(after),
                   model=model.to_dict())
    (output/'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    np.savetxt(output/'validation.csv', np.c_[pixels, measured, bias, before, after],
               delimiter=',', comments='',
               header='u_px,v_px,raw_depth_mm,bias_mm,uncorrected_z_mm,corrected_z_mm')
    # Same color scale on both sides; only held-out points are drawn.
    limit = max(1., np.max(np.abs(before)), np.max(np.abs(after)))
    views = []
    for title, values in [('Before', before), ('After', after)]:
        view = np.full((*depth.shape, 3), 45, np.uint8)
        for (x, y), z in zip(pixels, values):
            t = np.clip(z/limit, -1, 1)
            color = tuple(int(c*255) for c in (1-max(t, 0), 1-abs(t), 1+min(t, 0)))
            cv2.circle(view, (int(x), int(y)), 6, color, -1)
        cv2.putText(view, f'{title}: Z RMS {z_statistics(values)["rmse_mm"]:.2f} mm',
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
        cv2.putText(view, f'Shared scale: blue -{limit:.1f}, white 0, red +{limit:.1f} mm',
                    (20, depth.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
        views.append(view)
    if not cv2.imwrite(str(output/'comparison.png'), np.hstack(views)):
        raise OSError('Could not save comparison.png.')
    (output/'report.md').write_text(
        '# Offline depth-bias check\n\n'
        f'Source: `{source.resolve()}`\n\n'
        'Fit an affine camera-depth bias on alternating 100-pixel image tiles. '
        'Evaluate the other tiles at 20-pixel spacing. Test errors include outliers; '
        'points outside the fitted support hull are counted separately. Raw files are unchanged.\n\n'
        '| Held-out world Z error | Before | After |\n|---|---:|---:|\n'
        + ''.join(f'| {name} (mm) | {summary["before"][key]:.3f} | {summary["after"][key]:.3f} |\n'
                  for name, key in [('Mean', 'mean_mm'), ('RMSE', 'rmse_mm'), ('Max absolute', 'max_abs_mm')])
        + f'\nEvaluated points: {len(before)}; outside support: {int((~supported).sum())}.\n\n'
        '**This is spatial validation within one frozen frame. It does not establish accuracy '
        'on new frames or elevated targets, or distinguish depth bias from extrinsic error.** '
        'Do not apply this saved calibration to a different camera session.\n\n'
        '![Held-out board Z comparison](comparison.png)\n')
    return output, summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('board_report', type=Path)
    parser.add_argument('--output-root', type=Path, default=Path('depth_bias_results'))
    args = parser.parse_args()
    output, summary = check_report(args.board_report, args.output_root)
    print(json.dumps(dict(output=str(output), before=summary['before'], after=summary['after']), indent=2))
