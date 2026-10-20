"""Task 3.3: query real-controller IK without enabling or moving the arm.

python utils/verify_ik_vendor.py --ip 192.168.1.152
"""

import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import kinematics as k

POSE_FIELDS = ("x_mm", "y_mm", "z_mm", "roll_rad", "pitch_rad", "yaw_rad")


def load_targets(path):
    targets = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            sample = int(row["sample"])
            pose = np.array([float(row[field]) for field in POSE_FIELDS])
            if not np.all(np.isfinite(pose)):
                raise ValueError(f"Nonfinite target {sample}")
            if sample in targets and not np.array_equal(targets[sample], pose):
                raise ValueError(f"Conflicting poses for sample {sample}")
            targets[sample] = pose
    if not targets:
        raise ValueError("No targets in input CSV")
    return list(targets.items())


def pose_matrix(pose):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", pose[3:]).as_matrix()
    T[:3, 3] = pose[:3]
    return T


def errors(target, actual):
    return (float(np.linalg.norm(actual[:3, 3] - target[:3, 3])),
            float(np.degrees(Rotation.from_matrix(
                actual[:3, :3] @ target[:3, :3].T).magnitude())))


def legal_vendor_angles(raw):
    # Some SDK versions return seven entries, including padding for Lite 6.
    q = np.asarray(raw, dtype=float)
    if q.shape not in ((6,), (7,)) or not np.all(np.isfinite(q)):
        raise ValueError("Expected six finite angles (or seven SDK entries)")
    q = q[:6]
    lo, hi = k.JOINT_LIMITS.T
    minimum = np.ceil((lo - q - 1e-9) / (2 * np.pi))
    maximum = np.floor((hi - q + 1e-9) / (2 * np.pi))
    if np.any(minimum > maximum):
        return None
    turns = np.clip(np.rint((k.Q_DEFAULT - q) / (2 * np.pi)), minimum, maximum)
    return np.clip(q + turns * 2 * np.pi, lo, hi)


def compare_target(sdk, sample, pose):
    pose = np.asarray(pose, dtype=float)
    target = pose_matrix(pose)
    vendor = dict(sample=sample, code=None, status="exception", detail="",
                  raw_angles_rad=None, angles_rad=None,
                  target_position_error_mm=None, target_orientation_error_deg=None)
    vendor_T = None
    try:
        code, raw = sdk.get_inverse_kinematics(
            pose.tolist(), input_is_radian=True, return_is_radian=True)
        vendor["code"] = int(code)
        if code != 0:
            vendor["status"] = "sdk_error"
        else:
            q = legal_vendor_angles(raw)
            vendor["raw_angles_rad"] = np.asarray(raw, dtype=float).tolist()
            if q is None:
                vendor["status"] = "outside_limits"
            else:
                vendor["status"] = "ok"
                vendor["angles_rad"] = q.tolist()
                vendor_T = k.FK_dh(k.DH_STD, q, 6)
                pos, rot = errors(target, vendor_T)
                vendor.update(target_position_error_mm=pos, target_orientation_error_deg=rot)
    except Exception as exc:
        vendor["detail"] = f"{type(exc).__name__}: {exc}"

    rows = []
    for name, solve in (("geometric", k.IK_geometric), ("numerical", k.IK_numerical)):
        row = dict(sample=sample, solver=name, **dict(zip(POSE_FIELDS, map(float, pose))),
                   local_status="no_solution", local_detail="", vendor_status=vendor["status"],
                   vendor_code=vendor["code"], vendor_detail=vendor["detail"],
                   vendor_raw_angles_rad=json.dumps(vendor["raw_angles_rad"]),
                   local_target_position_error_mm=None, local_target_orientation_error_deg=None,
                   vendor_target_position_error_mm=vendor["target_position_error_mm"],
                   vendor_target_orientation_error_deg=vendor["target_orientation_error_deg"],
                   comparison_status="unavailable", position_error_mm=None, orientation_error_deg=None)
        row.update({f"local_q{i + 1}_rad": None for i in range(6)})
        row.update({f"vendor_q{i + 1}_rad": None if vendor["angles_rad"] is None
                    else vendor["angles_rad"][i] for i in range(6)})
        try:
            # Recompute current implementations; preserve the default numerical seed.
            q = solve(k.DH_STD, pose)
            if q is not None:
                q = np.asarray(q, dtype=float)
                lo, hi = k.JOINT_LIMITS.T
                if q.shape != (6,) or not np.all(np.isfinite(q)):
                    row["local_status"] = "invalid_solution"
                elif np.any(q < lo - 1e-9) or np.any(q > hi + 1e-9):
                    row["local_status"] = "outside_limits"
                else:
                    row.update({f"local_q{i + 1}_rad": float(q[i]) for i in range(6)})
                    actual = k.FK_dh(k.DH_STD, q, 6)
                    pos, rot = errors(target, actual)
                    row.update(local_target_position_error_mm=pos,
                               local_target_orientation_error_deg=rot,
                               local_status="ok" if pos <= 1e-3 and rot <= 1e-3 else "pose_error")
                    if vendor_T is not None and row["local_status"] == "ok":
                        pos, rot = errors(vendor_T, actual)
                        row.update(position_error_mm=pos, orientation_error_deg=rot,
                                   comparison_status="within_10mm" if pos <= 10 else "over_10mm")
        except Exception as exc:
            row.update(local_status="exception", local_detail=f"{type(exc).__name__}: {exc}")
        rows.append(row)
    return vendor, rows


def metric(rows, field):
    values = [row[field] for row in rows if row[field] is not None]
    return dict(count=len(values), mean=float(np.mean(values)) if values else None,
                max=float(np.max(values)) if values else None)


def save_results(output, rows, vendors, metadata):
    summary = dict(metadata, completed_targets=len(vendors),
                   vendor_status_counts=dict(Counter(v["status"] for v in vendors)),
                   vendor_error_codes=dict(Counter(str(v["code"]) for v in vendors
                                                  if v["code"] not in (None, 0))),
                   vendor_target_position_mm=metric(vendors, "target_position_error_mm"),
                   vendor_target_orientation_deg=metric(vendors, "target_orientation_error_deg"),
                   vendor_results=vendors, solvers={})
    for name in ("geometric", "numerical"):
        selected = [r for r in rows if r["solver"] == name]
        summary["solvers"][name] = dict(
            local_status_counts=dict(Counter(r["local_status"] for r in selected)),
            comparison_counts=dict(Counter(r["comparison_status"] for r in selected)),
            position_mm=metric(selected, "position_error_mm"),
            orientation_deg=metric(selected, "orientation_error_deg"))
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    def pair(value):
        return "N/A" if value["mean"] is None else f"{value['mean']:.6g} / {value['max']:.6g}"
    lines = ["# Task 3.3: vendor IK comparison", "",
             f"Completed {len(vendors)}/{metadata['requested_targets']} targets. "
             "Compare both local IK answers with vendor IK using the same FK_dh model; numerical seed: Q_DEFAULT.",
             "No motor-enable or motion commands. Position threshold: 10 mm; orientation is reported without a pass threshold.", "",
             "| Solver | Compared | ≤10 mm | >10 mm | Unavailable | Position mean / max (mm) | Orientation mean / max (deg) |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, stats in summary["solvers"].items():
        counts = stats["comparison_counts"]
        lines.append(f"| {name} | {stats['position_mm']['count']} | {counts.get('within_10mm', 0)} | "
                     f"{counts.get('over_10mm', 0)} | {counts.get('unavailable', 0)} | "
                     f"{pair(stats['position_mm'])} | {pair(stats['orientation_deg'])} |")
    lines += ["", f"Vendor status counts: `{summary['vendor_status_counts']}`."]
    for name, stats in summary["solvers"].items():
        lines.append(f"{name} status counts: `{stats['local_status_counts']}`.")
    lines += ["", "Error statistics cover valid pairs only; failures remain in the counts and samples.csv."]
    if metadata.get("interruption"):
        lines += ["", "Partial run: " + metadata["interruption"]]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", required=True)
    parser.add_argument("--input", type=Path, default=ROOT / "ik_roundtrip_results/samples.csv")
    parser.add_argument("--limit", type=int, help="Only query the first N unique targets")
    parser.add_argument("--output", type=Path, help="New output directory (must not exist)")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    targets = load_targets(args.input)
    if args.limit is not None:
        targets = targets[:args.limit]
    output = args.output or ROOT / "ik_vendor_results" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    metadata = dict(ip=args.ip, source_csv=str(args.input.resolve()), requested_targets=len(targets),
                    source_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                    kinematics_sha256=hashlib.sha256(Path(k.__file__).read_bytes()).hexdigest(),
                    dh_params=k.DH_STD.tolist(), joint_limits_rad=k.JOINT_LIMITS.tolist(),
                    numerical_seed_rad=k.Q_DEFAULT.tolist(), position_threshold_mm=10,
                    angle_units="radians; reported orientation errors are degrees",
                    vendor_angle_policy="SDK defaults; use first six entries and legal equivalent 2*pi turns",
                    comparison="FK_dh(local IK) versus FK_dh(vendor IK)")
    from xarm.wrapper import XArmAPI
    sdk, rows, vendors = None, [], []
    try:
        print(f"Connecting to {args.ip}; querying {len(targets)} targets, no motion commands.", flush=True)
        sdk = XArmAPI(args.ip, is_radian=True)
        if not sdk.connected:
            raise ConnectionError("Controller is not connected; check power, network and IP")
        metadata["firmware_version"] = sdk.version
        with (output / "samples.csv").open("w", newline="") as stream:
            writer = None
            for sample, pose in targets:
                if not sdk.connected:
                    raise ConnectionError("Controller disconnected during sweep")
                vendor, comparisons = compare_target(sdk, sample, pose)
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
                    writer.writeheader()
                writer.writerows(comparisons)
                stream.flush()
                rows.extend(comparisons)
                vendors.append(vendor)
                if len(vendors) % 25 == 0 or len(vendors) == len(targets):
                    print(f"Queried {len(vendors)}/{len(targets)} targets", flush=True)
    except (Exception, KeyboardInterrupt) as exc:
        metadata["interruption"] = f"{type(exc).__name__}: {exc}"
        print(metadata["interruption"], file=sys.stderr)
    finally:
        if sdk is not None:
            sdk.disconnect()
        summary = save_results(output, rows, vendors, metadata)
        print(f"Results: {output}")
    if metadata.get("interruption"):
        return 2
    return int(any(s["comparison_counts"].get("within_10mm", 0) != len(targets)
                   for s in summary["solvers"].values()))


if __name__ == "__main__":
    sys.exit(main())
