"""Tasks 2.2/2.3: compare DH, PoX and the connected arm's solver (no motion).

Run from the repository root:
    python src/check_fk.py --ip 192.168.1.152 --samples 500
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from kinematics import DH_STD, JOINT_LIMITS, M, S_list, FK_dh, FK_pox, get_pose_from_T


COMPARISONS = {
    "student": "Student DH vs SDK",
    "firmware": "Firmware DH vs SDK",
    "pox_sdk": "PoX vs SDK",
    "pox_dh": "PoX vs student DH",
}


def pose_errors(mine, sdk):
    """Return translation distance in mm and rotation distance in radians."""
    mine, sdk = np.asarray(mine), np.asarray(sdk)
    r_mine = Rotation.from_euler("xyz", mine[3:]).as_matrix()
    r_sdk = Rotation.from_euler("xyz", sdk[3:]).as_matrix()
    # Rotation magnitude avoids arccos precision loss near zero error.
    angle = Rotation.from_matrix(r_mine.T @ r_sdk).magnitude()
    return float(np.linalg.norm(mine[:3] - sdk[:3])), float(
        angle
    )


def statistics(values):
    return dict(zip(
        ("mean", "median", "max", "p95"),
        map(float, (np.mean(values), np.median(values),
                    np.max(values), np.percentile(values, 95))),
    ))


def collect_diagnostics(arm):
    """Read controller information and compare zero-pose FK without motion."""
    version_code, version = arm.get_version()
    zero_code, zero_sdk = arm.get_forward_kinematics(
        [0.0] * 6, input_is_radian=True, return_is_radian=True,
    )
    zero_mine = get_pose_from_T(FK_dh(DH_STD, np.zeros(6), 6))
    result = {
        "pose_units": "xyz: mm; roll, pitch, yaw: rad",
        "firmware_version_code": version_code,
        "firmware_version": version if version_code == 0 else None,
        # These properties are cached report values, not explicit readbacks.
        "offset_source": "SDK cached report properties after random sweep; freshness not verified",
        "tcp_offset": list(arm.tcp_offset),
        "world_offset": list(arm.world_offset),
        "zero_pose_sdk_code": zero_code,
        "zero_pose_sdk": None,
        "zero_pose_student": zero_mine,
    }
    if zero_code == 0:
        sdk = np.asarray(zero_sdk, dtype=float)
        if sdk.shape == (6,) and np.all(np.isfinite(sdk)):
            result["zero_pose_sdk"] = sdk.tolist()
            position, orientation = pose_errors(zero_mine, sdk)
            result["zero_pose_error"] = {
                "position_mm": position,
                "orientation_deg": float(np.degrees(orientation)),
            }
        else:
            result["zero_pose_warning"] = "SDK returned an invalid zero pose"
    return result


def save_plots(rows, output):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Plots skipped: install matplotlib and rerun to generate figures.")
        return

    for table, label in COMPARISONS.items():
        data = np.array([row[1:] for row in rows if row[0] == table])
        q_deg = np.degrees(data[:, :6])
        position, orientation = data[:, -2], np.degrees(data[:, -1])
        fig, ax = plt.subplots()
        ax.hist(position, bins=30)
        ax.set(xlabel="Position error (mm)", ylabel="Samples", title=label)
        fig.tight_layout()
        fig.savefig(output / f"{table}_position_histogram.png", dpi=160)
        plt.close(fig)

        fig, axes = plt.subplots(2, 6, figsize=(18, 6))
        for j in range(6):
            for i, (errors, unit) in enumerate(
                ((position, "Position error (mm)"),
                 (orientation, "Orientation error (deg)"))
            ):
                axes[i, j].scatter(q_deg[:, j], errors, s=5)
                axes[i, j].set(xlabel=f"Joint {j + 1} (deg)", ylabel=unit)
        fig.suptitle(label)
        fig.tight_layout()
        fig.savefig(output / f"{table}_joint_errors.png", dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", required=True, help="Real arm's IP address")
    parser.add_argument("--samples", type=int, default=500,
                        help="Number of random configurations to attempt")
    parser.add_argument("--seed", type=int, default=550)
    parser.add_argument("--output", type=Path, default=Path("fk_results"))
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")

    from xarm.wrapper import XArmAPI

    rng = np.random.default_rng(args.seed)
    rows, skipped = [], {}
    arm = XArmAPI(args.ip, is_radian=True)
    try:
        code, dh = arm.get_dh_params()
        if code != 0:
            raise RuntimeError(f"get_dh_params failed: code={code}")
        firmware_raw = np.asarray(dh, dtype=float).reshape(-1, 4)
        if firmware_raw.shape != (7, 4):
            raise ValueError(f"Unexpected firmware DH table shape: {firmware_raw.shape}")
        if not np.all(np.isfinite(firmware_raw)):
            raise ValueError("Firmware returned an invalid DH table")
        # Align the Lite 6 seven-row firmware layout with Standard DH:
        # theta_offset and d belong to this row; alpha and a to the next.
        firmware = firmware_raw[:6].copy()
        firmware[:, 2:] = firmware_raw[1:7, 2:]
        for _ in range(args.samples):
            q = rng.uniform(JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
            code, sdk = arm.get_forward_kinematics(q.tolist())
            if code != 0:
                skipped[str(code)] = skipped.get(str(code), 0) + 1
                continue
            sdk = np.asarray(sdk, dtype=float)
            if sdk.shape != (6,) or not np.all(np.isfinite(sdk)):
                raise ValueError(f"Invalid SDK pose: {sdk}")
            student = get_pose_from_T(FK_dh(DH_STD, q, 6))
            firmware_pose = get_pose_from_T(FK_dh(firmware, q, 6))
            pox = get_pose_from_T(FK_pox(q, M, S_list))
            for name, mine, reference in (
                ("student", student, sdk),
                ("firmware", firmware_pose, sdk),
                ("pox_sdk", pox, sdk),
                ("pox_dh", pox, student),
            ):
                errors = pose_errors(mine, reference)
                rows.append([name, *q, *reference, *mine, *errors])
        diagnostics = collect_diagnostics(arm)
        print("Controller and zero-pose diagnostics:")
        print(json.dumps(diagnostics, indent=2))
    finally:
        arm.disconnect()

    if not rows:
        raise RuntimeError(f"No valid SDK samples. Error counts: {skipped}")
    args.output.mkdir(parents=True, exist_ok=True)
    header = ["comparison"] + [f"q{i}_rad" for i in range(1, 7)]
    header += [f"{source}_{axis}" for source in ("reference", "mine")
               for axis in ("x_mm", "y_mm", "z_mm", "roll_rad", "pitch_rad", "yaw_rad")]
    header += ["position_error_mm", "orientation_error_rad"]
    with (args.output / "samples.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)
    summary = {"attempted": args.samples, "valid": len(rows) // len(COMPARISONS),
               "comparisons": COMPARISONS,
               "orientation_error_method": "relative rotation magnitude",
               "pox_M": M.tolist(), "pox_S_list": S_list.tolist(),
               "skipped_codes": skipped, "seed": args.seed,
               "student_dh": DH_STD.tolist(),
               "diagnostics": diagnostics,
               "firmware_dh_raw": firmware_raw.tolist(),
               "firmware_dh": firmware.tolist()}
    for name in COMPARISONS:
        errors = np.array([row[-2:] for row in rows if row[0] == name])
        summary[name] = {"position_mm": statistics(errors[:, 0]),
                         "orientation_deg": statistics(np.degrees(errors[:, 1]))}
    rendered = json.dumps(summary, indent=2)
    (args.output / "summary.json").write_text(rendered + "\n")
    print(rendered)
    save_plots(rows, args.output)
    print(f"Results saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
