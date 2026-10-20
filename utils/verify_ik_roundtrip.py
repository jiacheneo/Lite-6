"""Task 3.3: verify both IK solvers without hardware or the GUI.

Run from the repository root:
    python utils/verify_ik_roundtrip.py --samples 300 --seed 550
"""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import kinematics as k


def pose_errors(target, actual):
    """Return Euclidean position error (mm) and relative rotation angle (deg)."""
    position = np.linalg.norm(actual[:3, 3] - target[:3, 3])
    orientation = Rotation.from_matrix(
        actual[:3, :3] @ target[:3, :3].T
    ).magnitude()
    return float(position), float(np.degrees(orientation))


def verify(samples=300, seed=550, position_tol_mm=1e-3, orientation_tol_deg=1e-3):
    rng = np.random.default_rng(seed)
    lo, hi = k.JOINT_LIMITS.T
    joints = rng.uniform(lo, hi, size=(samples, 6))
    rows = []
    solvers = {"geometric": k.IK_geometric, "numerical": k.IK_numerical}
    for index, original in enumerate(joints):
        target = k.FK_dh(k.DH_STD, original, 6)
        pose = k.get_pose_from_T(target)
        for name, solve in solvers.items():
            row = {"sample": index, "solver": name}
            row.update({f"input_q{i + 1}_rad": float(q) for i, q in enumerate(original)})
            row.update(dict(zip(
                ("x_mm", "y_mm", "z_mm", "roll_rad", "pitch_rad", "yaw_rad"), pose
            )))
            row.update({f"solution_q{i + 1}_rad": None for i in range(6)})
            row.update(status="no_solution", position_error_mm=None,
                       orientation_error_deg=None, detail="")
            try:
                # Keep the default numerical seed Q_DEFAULT. Do not pass original.
                solution = solve(k.DH_STD, pose)
                if solution is not None:
                    solution = np.asarray(solution, dtype=float)
                    if solution.shape != (6,) or not np.all(np.isfinite(solution)):
                        row["status"] = "invalid_solution"
                    else:
                        row.update({f"solution_q{i + 1}_rad": float(q)
                                    for i, q in enumerate(solution)})
                        pos, rot = pose_errors(target, k.FK_dh(k.DH_STD, solution, 6))
                        row.update(position_error_mm=pos, orientation_error_deg=rot)
                        if np.any(solution < lo - 1e-9) or np.any(solution > hi + 1e-9):
                            row["status"] = "outside_limits"
                        elif pos > position_tol_mm or rot > orientation_tol_deg:
                            row["status"] = "pose_error"
                        else:
                            row["status"] = "pass"
            except Exception as exc:
                row.update(status="exception", detail=f"{type(exc).__name__}: {exc}")
            rows.append(row)
        if (index + 1) % 50 == 0 or index + 1 == samples:
            print(f"Checked {index + 1}/{samples} targets", flush=True)

    summary = {
        "samples": samples, "seed": seed,
        "fk": "FK_dh(DH_STD, q, 6)",
        "numerical_seed_rad": k.Q_DEFAULT.tolist(),
        "dh_params": k.DH_STD.tolist(), "joint_limits_rad": k.JOINT_LIMITS.tolist(),
        "position_tol_mm": position_tol_mm, "orientation_tol_deg": orientation_tol_deg,
        "statistics_scope": "All finite returned poses, including failed checks; None has no error value.",
        "solvers": {},
    }
    for name in solvers:
        selected = [row for row in rows if row["solver"] == name]
        counts = {status: sum(row["status"] == status for row in selected)
                  for status in ("pass", "no_solution", "invalid_solution",
                                 "outside_limits", "pose_error", "exception")}
        stats = {"counts": counts, "success_rate": counts["pass"] / samples,
                 "failed_sample_ids": [row["sample"] for row in selected
                                       if row["status"] != "pass"]}
        for field in ("position_error_mm", "orientation_error_deg"):
            values = [row[field] for row in selected if row[field] is not None]
            stats[field] = {"count": len(values),
                            "mean": float(np.mean(values)) if values else None,
                            "max": float(np.max(values)) if values else None}
        summary["solvers"][name] = stats
    return rows, summary


def write_results(output, rows, summary):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "samples.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    lines = [
        "# Task 3.3: IK round-trip verification", "",
        f"{summary['samples']} targets sampled uniformly within JOINT_LIMITS; seed {summary['seed']}.",
        "Target: FK_dh -> get_pose_from_T. Each solver's answer is checked with FK_dh.",
        "Numerical IK starts from Q_DEFAULT for every target; the original joints are never used as a seed.",
        "No hardware connection or motion is involved.", "",
        f"Pass thresholds: position <= {summary['position_tol_mm']:g} mm, "
        f"orientation <= {summary['orientation_tol_deg']:g} deg, with all joints inside limits.",
        "These are practical test thresholds, not a claim of floating-point-level accuracy.",
        "Orientation error is the angle of the relative rotation, not an Euler-angle difference.", "",
        "| Solver | Pass / total | No solution | Other failures | Position mean / max (mm) | Orientation mean / max (deg) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    def formatted(metric):
        if metric["mean"] is None:
            return "N/A"
        return f"{metric['mean']:.6g} / {metric['max']:.6g}"

    for name, stats in summary["solvers"].items():
        counts = stats["counts"]
        other = summary["samples"] - counts["pass"] - counts["no_solution"]
        lines.append(f"| {name} | {counts['pass']} / {summary['samples']} | "
                     f"{counts['no_solution']} | {other} | "
                     f"{formatted(stats['position_error_mm'])} | "
                     f"{formatted(stats['orientation_error_deg'])} |")
    lines += ["", "Error statistics include all finite returned poses, including failed checks.",
              "Missing solutions have no measurable pose error and are counted separately.",
              "Every target has a known legal solution (the sampled joints), so None indicates a solver failure here.",
              "See samples.csv for input joints, target poses, returned joints, errors and failure details.",
              "Different joint configurations are accepted when they reproduce the target pose within limits.", ""]
    report = "\n".join(lines)
    (output / "report.md").write_text(report)
    print("\n" + report)
    print(f"Results saved to {output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=550)
    parser.add_argument("--output", type=Path, default=Path("ik_roundtrip_results"))
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    rows, summary = verify(args.samples, args.seed)
    write_results(args.output, rows, summary)
    return int(any(stats["counts"]["pass"] != args.samples
                   for stats in summary["solvers"].values()))


if __name__ == "__main__":
    sys.exit(main())
