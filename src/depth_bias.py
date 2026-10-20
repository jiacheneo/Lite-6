"""Robust camera-Z bias surface fitted against the known world-Z=0 table."""
from dataclasses import dataclass

import numpy as np


@dataclass
class DepthBiasModel:
    bias_mm: np.ndarray
    coefficients: np.ndarray
    intrinsic: np.ndarray
    extrinsic: np.ndarray
    scale: float
    diagnostics: dict

    def matches(self, camera):
        return (self.bias_mm.shape == camera.depth_frame_raw.shape
                and self.scale == camera.depth_scale_mm
                and np.array_equal(self.intrinsic, camera.intrinsic_matrix)
                and np.array_equal(self.extrinsic, camera.extrinsic_matrix))

    def to_dict(self):
        return dict(model='quadratic_camera_z_bias',
                    formula='b0 + bx*rx + by*ry + bxx*rx^2 + bxy*rx*ry + byy*ry^2',
                    application='Full depth image; extrapolated outside the fitting selection.',
                    coefficients_mm=self.coefficients.tolist(),
                    intrinsic=self.intrinsic.tolist(), extrinsic=self.extrinsic.tolist(),
                    depth_scale_mm=self.scale, diagnostics=self.diagnostics,
                    validation='Fit-frame residuals; validate new frames and elevated targets.')


def fit_depth_bias(depth, mask, intrinsic, extrinsic, scale):
    """Fit measured minus expected depth, rejecting objects with RANSAC.

    The selection must be mostly bare table and controls fitting samples only.
    Evaluate the smooth surface over the full image, extrapolating outside the
    selection. Raw zero depths never participate in the fit.
    """
    v, u = np.indices(depth.shape)
    rays = np.linalg.solve(intrinsic, np.stack((u.ravel(), v.ravel(),
                                              np.ones(depth.size))))
    camera_to_world = np.linalg.inv(extrinsic)
    slopes = (camera_to_world[2, :3] @ rays).reshape(depth.shape)
    expected = np.divide(-camera_to_world[2, 3], slopes,
                         out=np.full(depth.shape, np.nan), where=np.abs(slopes) > 1e-6)
    rx, ry = rays[:2].reshape(2, *depth.shape)
    basis = np.stack((np.ones(depth.shape), rx, ry, rx*rx, rx*ry, ry*ry), axis=-1)
    support = (mask != 0) & np.isfinite(expected) & (expected > 0)
    selected = support & (depth > 0) & (u % 8 == 0) & (v % 8 == 0)
    design = basis[selected]
    residual = depth[selected].astype(float)*scale - expected[selected]
    if len(residual) < 100 or np.linalg.matrix_rank(design) < 6:
        raise ValueError('Select a larger table area with at least 100 valid depth samples.')
    rng = np.random.default_rng(550)
    best = np.zeros(len(residual), dtype=bool)
    for _ in range(300):
        indices = rng.choice(len(residual), 6, replace=False)
        if np.linalg.matrix_rank(design[indices]) < 6:
            continue
        coefficients = np.linalg.solve(design[indices], residual[indices])
        inliers = np.abs(residual - design @ coefficients) <= 3.0
        if inliers.sum() > best.sum():
            best = inliers
    minimum = max(100, int(np.ceil(.6 * len(residual))))
    for _ in range(4):
        if best.sum() < minimum or np.linalg.matrix_rank(design[best]) < 6:
            raise ValueError('Not enough consistent table depth. Clear the board and refit.')
        coefficients = np.linalg.lstsq(design[best], residual[best], rcond=None)[0]
        error = residual - design @ coefficients
        median = np.median(error[best])
        sigma = 1.4826*np.median(np.abs(error[best] - median))
        best = np.abs(error - median) <= max(3., 3*sigma)
    if best.sum() < minimum or np.linalg.matrix_rank(design[best]) < 6:
        raise ValueError('Not enough consistent table depth. Clear the board and refit.')
    coefficients = np.linalg.lstsq(design[best], residual[best], rcond=None)[0]
    bias = (basis @ coefficients).astype(np.float32)
    # Check the entire application domain, including the extrapolated edges.
    if not np.all(np.isfinite(bias)) or np.max(np.abs(bias)) > 100:
        raise ValueError('Bias exceeds 100 mm over the image. Check calibration and spread the table corners farther apart.')
    error = residual - design @ coefficients
    diagnostics = dict(samples=len(residual), inliers=int(best.sum()),
                       rejected=int((~best).sum()),
                       before_rmse_mm=float(np.sqrt(np.mean(residual[best]**2))),
                       after_rmse_mm=float(np.sqrt(np.mean(error[best]**2))))
    return DepthBiasModel(bias, coefficients, intrinsic.copy(), extrinsic.copy(),
                          float(scale), diagnostics)
