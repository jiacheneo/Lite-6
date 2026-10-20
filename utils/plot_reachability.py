"""Task 3.3: plot top-down grasp reachability without hardware.

Requires numpy, scipy and matplotlib. Run from the repository root:
    python utils/plot_reachability.py
    python utils/plot_reachability.py --bounds -100 500 -500 500
"""

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.path import Path as PlotPath
from matplotlib.patches import Patch, PathPatch
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import kinematics as k

DEFAULT_BOARD = ROOT / "board_results_2.8b/20260924_140022_426397/summary.json"
YAW_DEG = (0, 90)
TOOL_OFFSET_MM = 85.0


def board_polygon(report, table_z):
    """Intersect saved boundary pixel rays with the world table plane."""
    metadata = json.loads(report.read_text())
    pixels = np.asarray(metadata["board_polygon_image_px"], dtype=float)
    intrinsic = np.asarray(metadata["intrinsic_matrix"], dtype=float)
    extrinsic = np.asarray(metadata["extrinsic_matrix_world_to_camera"], dtype=float)
    rotation, translation = extrinsic[:3, :3], extrinsic[:3, 3]
    origin = -rotation.T @ translation
    rays = np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(intrinsic).T @ rotation
    if np.any(np.abs(rays[:, 2]) < 1e-10):
        raise ValueError("Board boundary ray is parallel to the table.")
    distance = (table_z - origin[2]) / rays[:, 2]
    if np.any(distance <= 0):
        raise ValueError("Board boundary is behind the camera.")
    return (origin + distance[:, None] * rays)[:, :2]


def sample_grid(polygon, step):
    # Anchor the grid to robot coordinates, so repeated maps are comparable.
    low = np.ceil(polygon.min(axis=0) / step) * step
    high = np.floor(polygon.max(axis=0) / step) * step
    x = np.arange(low[0], high[0] + step / 2, step)
    y = np.arange(low[1], high[1] + step / 2, step)
    X, Y = np.meshgrid(x, y)
    path = PlotPath(np.vstack([polygon, polygon[0]]))
    points = np.c_[X.ravel(), Y.ravel()]
    # Union of +/- radius includes boundary points regardless of winding.
    inside = (path.contains_points(points, radius=1e-7)
              | path.contains_points(points, radius=-1e-7)).reshape(X.shape)
    if not inside.any():
        raise ValueError("No board grid points; check boundary and spacing.")
    return X, Y, inside


def scan(X, Y, inside, block_height, table_z):
    flange_z = table_z + block_height / 2 + TOOL_OFFSET_MM
    reachable = np.zeros((*inside.shape, 2), dtype=bool)
    rows = []
    for iy, ix in np.argwhere(inside):
        x, y = float(X[iy, ix]), float(Y[iy, ix])
        for orientation, yaw in enumerate(YAW_DEG):
            # Fixed-axis xyz: roll pi turns the tool's +Z axis downward.
            pose = [x, y, flange_z, np.pi, 0.0, np.radians(yaw)]
            target_rotation = Rotation.from_euler("xyz", pose[3:]).as_matrix()
            solution = k.IK_geometric(k.DH_STD, pose)
            row = dict(block_height_mm=block_height, x_mm=x, y_mm=y,
                       flange_z_mm=flange_z, roll_deg=180, pitch_deg=0,
                       yaw_deg=yaw, reachable=0, position_error_mm=None,
                       orientation_error_deg=None)
            row.update({f"q{i + 1}_rad": None for i in range(6)})
            if solution is not None:
                solution = np.asarray(solution, dtype=float)
                lo, hi = k.JOINT_LIMITS.T
                if (solution.shape != (6,) or not np.all(np.isfinite(solution))
                        or np.any(solution < lo - 1e-9) or np.any(solution > hi + 1e-9)):
                    raise ValueError(f"Invalid IK solution at {pose}")
                actual = k.FK_dh(k.DH_STD, solution, 6)
                pos_error = float(np.linalg.norm(actual[:3, 3] - pose[:3]))
                rot_error = float(Rotation.from_matrix(
                    actual[:3, :3] @ target_rotation.T).magnitude())
                if pos_error > 1e-6 or rot_error > 1e-6:
                    raise ValueError(f"IK fails FK validation at {pose}")
                reachable[iy, ix, orientation] = True
                row.update(reachable=1, position_error_mm=pos_error,
                           orientation_error_deg=float(np.degrees(rot_error)))
                row.update({f"q{i + 1}_rad": float(q) for i, q in enumerate(solution)})
            rows.append(row)
    union = reachable.any(axis=2)
    total = int(inside.sum())
    summary = dict(block_height_mm=block_height, flange_z_mm=flange_z,
                   grid_points=total,
                   reachable_yaw_0=int(reachable[:, :, 0].sum()),
                   reachable_yaw_90=int(reachable[:, :, 1].sum()),
                   reachable_either=int(union.sum()),
                   reachable_both=int(reachable.all(axis=2).sum()),
                   unreachable=int(total - union.sum()),
                   reachable_percent=float(100 * union.sum() / total),
                   yaw_disagreement_points=int(np.count_nonzero(
                       reachable[:, :, 0] != reachable[:, :, 1])))
    return reachable, rows, summary


def plot_map(output, polygon, X, Y, inside, reachable, summary, step, estimated):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(13, 8), sharex=True, sharey=True)
    colors = ["#dce1e7", "#218a79"]
    cmap = ListedColormap(colors)
    outline = np.vstack([polygon, polygon[0]])
    boundary = PlotPath(outline)
    views = (reachable[:, :, 0], reachable[:, :, 1], reachable.any(axis=2))
    labels = ("Yaw 0°", "Yaw 90°", "Either orientation")
    for ax, values, label in zip(axes, views, labels):
        data = np.ma.array(values.astype(int), mask=~inside)
        xedges = np.r_[X[0] - step / 2, X[0, -1] + step / 2]
        yedges = np.r_[Y[:, 0] - step / 2, Y[-1, 0] + step / 2]
        mesh = ax.pcolormesh(xedges, yedges, data, cmap=cmap, vmin=0, vmax=1,
                             shading="flat", rasterized=True, zorder=2)
        mesh.set_clip_path(PathPatch(boundary, transform=ax.transData))
        ax.plot(outline[:, 0], outline[:, 1], color="#27374a", lw=1.5, zorder=3)
        ax.plot(0, 0, marker="*", color="#c14f24", markersize=12, zorder=4)
        ax.annotate("Base", (0, 0), xytext=(7, 8), textcoords="offset points", fontsize=9)
        count = int(values.sum())
        ax.set_title(f"{label}\n{count:,}/{summary['grid_points']:,} points "
                     f"({100 * count / summary['grid_points']:.1f}%)", pad=12)
        ax.set_xlabel("Robot X (mm)")
        ax.set_aspect("equal")
        ax.set_xlim(min(0, polygon[:, 0].min()) - 35, max(0, polygon[:, 0].max()) + 35)
        ax.set_ylim(min(0, polygon[:, 1].min()) - 35, max(0, polygon[:, 1].max()) + 35)
        ax.grid(alpha=.2, zorder=0)
    axes[0].set_ylabel("Robot Y (mm)")
    size = summary["block_height_mm"]
    fig.suptitle(f"Top-down grasp reachability · {size:g} mm block\n"
                 f"Flange Z = {summary['flange_z_mm']:g} mm  |  "
                 f"{step:g} mm grid  |  Geometric IK", fontsize=16, y=.96)
    fig.legend(handles=[Patch(color=colors[1], label="Reachable"),
                        Patch(color=colors[0], label="Unreachable"),
                        Line2D([0], [0], color="#27374a", label="Board boundary"),
                        Line2D([0], [0], marker="*", color="#c14f24", lw=0,
                               markersize=11, label="Arm base (0, 0)")],
               loc="lower center", bbox_to_anchor=(.5, .065), ncol=4, frameon=False)
    source = "Estimated board outline from saved calibration" if estimated else "User-specified board bounds"
    fig.text(.5, .035, source + " · IK feasibility only; collisions and approach paths are not evaluated.",
             ha="center", fontsize=9, color="#48586a")
    fig.subplots_adjust(left=.065, right=.98, bottom=.17, top=.80, wspace=.18)
    stem = output / f"reachability_{size:g}mm"
    fig.savefig(stem.with_suffix(".png"), dpi=180, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    board = parser.add_mutually_exclusive_group()
    board.add_argument("--board-report", type=Path, default=DEFAULT_BOARD)
    board.add_argument("--bounds", nargs=4, type=float, metavar=("XMIN", "XMAX", "YMIN", "YMAX"))
    parser.add_argument("--step", type=float, default=10)
    parser.add_argument("--table-z", type=float, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "reachability_results")
    args = parser.parse_args()
    if not np.isfinite(args.step) or args.step <= 0 or not np.isfinite(args.table_z):
        parser.error("Spacing must be positive and table height must be finite.")
    if args.bounds:
        xmin, xmax, ymin, ymax = args.bounds
        if not np.all(np.isfinite(args.bounds)) or xmin >= xmax or ymin >= ymax:
            parser.error("Bounds must be finite and increasing.")
        polygon = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
        source = "User-specified rectangular board bounds"
    else:
        polygon = board_polygon(args.board_report, args.table_z)
        source = f"Estimated from selected board polygon in {args.board_report}"
    X, Y, inside = sample_grid(polygon, args.step)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    summary = dict(board_source=source, board_outline_estimated=not bool(args.bounds),
                   board_polygon_xy_mm=polygon.tolist(), step_mm=args.step,
                   table_z_mm=args.table_z, tool_offset_mm=TOOL_OFFSET_MM,
                   roll_deg=180, pitch_deg=0, yaw_deg=list(YAW_DEG),
                   dh_params=k.DH_STD.tolist(), joint_limits_rad=k.JOINT_LIMITS.tolist(),
                   scope="Geometric IK plus joint limits and FK validation; no collision or path checking.",
                   blocks=[])
    rows, maps = [], []
    for size in (38, 25):
        reachable, block_rows, stats = scan(X, Y, inside, size, args.table_z)
        plot_map(output, polygon, X, Y, inside, reachable, stats, args.step, not bool(args.bounds))
        maps.append(reachable.any(axis=2))
        rows.extend(block_rows)
        summary["blocks"].append(stats)
        print(f"{size} mm: {stats['reachable_either']}/{stats['grid_points']} reachable "
              f"({stats['reachable_percent']:.2f}%)", flush=True)
    changed = int(np.count_nonzero(maps[0] != maps[1]))
    summary["size_comparison"] = dict(changed_grid_points=changed,
        changed_percent=float(100 * changed / inside.sum()),
        only_38mm=int(np.count_nonzero(maps[0] & ~maps[1])),
        only_25mm=int(np.count_nonzero(maps[1] & ~maps[0])))
    with (output / "samples.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    lines = ["# Task 3.3: reachability map", "",
             f"Top-down geometric IK on a {args.step:g} mm grid; roll = 180°, pitch = 0°, yaw = 0° / 90°.",
             f"Flange Z = table Z ({args.table_z:g} mm) + block height / 2 + 85 mm. Returned solutions pass joint-limit and FK checks.", "",
             "Board outline: " + source + ".", "",
             "| Block | Flange Z | Yaw 0° | Yaw 90° | Either / sampled | Reachable |",
             "| --- | --- | --- | --- | --- | --- |"]
    for stats in summary["blocks"]:
        lines.append(f"| {stats['block_height_mm']} mm | {stats['flange_z_mm']:g} mm | "
                     f"{stats['reachable_yaw_0']} | {stats['reachable_yaw_90']} | "
                     f"{stats['reachable_either']} / {stats['grid_points']} | {stats['reachable_percent']:.2f}% |")
    lines += ["", f"Block size changes reachability at {changed} grid points "
              f"({summary['size_comparison']['changed_percent']:.2f}%). Percentages refer to sampled points.",
              "IK feasibility only; collisions and approach paths are not evaluated.", "",
              "![38 mm block](reachability_38mm.png)", "", "![25 mm block](reachability_25mm.png)", ""]
    (output / "report.md").write_text("\n".join(lines))
    print(f"Saved plots, report and sample data to {output}")


if __name__ == "__main__":
    main()
