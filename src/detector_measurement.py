"""Task 3.5: fixed manual-placement trials, durable records and offline reports."""
import argparse
import csv
import io
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

COLORS = ('red', 'orange', 'yellow', 'green', 'blue', 'purple')
POSITIONS = ((150, -250), (150, 250), (350, 250), (350, -250), (250, 0))
SIZES = (('small', 25), ('large', 38))


def trial_plan():
    plan = []
    for p, (x, y) in enumerate(POSITIONS, start=1):
        for size, mm in SIZES:
            for color in COLORS:
                for repeat in range(1, 4):
                    plan.append(dict(
                        trial_id=len(plan) + 1, point=f'P{p}', truth_x_mm=x,
                        truth_y_mm=y, truth_size=size, nominal_size_mm=mm,
                        truth_color=color, repetition=repeat))
    return plan


def atomic_text(path, content):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def make_row(trial, detection, count, frame_time, evidence, note=''):
    world = detection.get('world_position_mm') if detection is not None else None
    valid = world is not None and len(world) == 3 and all(math.isfinite(v) for v in world)
    row = dict(trial, recorded_at=datetime.now(timezone.utc).isoformat(),
               frame_time=frame_time, detection_count=count, detected=detection is not None,
               position_valid=valid, predicted_color='missed', predicted_size='',
               detected_x_mm=None, detected_y_mm=None, detected_z_mm=None,
               dx_mm=None, dy_mm=None, error_xy_mm=None, orientation_deg=None,
               position_reason='', color_reason='', evidence=evidence, note=note)
    if detection is not None:
        row.update(predicted_color=detection.get('color', 'unknown'),
                   predicted_size=detection.get('size', ''),
                   orientation_deg=detection.get('orientation_deg'),
                   position_reason=detection.get('position_reason') or '',
                   color_reason=detection.get('color_reason') or '')
    if valid:
        dx, dy = world[0] - trial['truth_x_mm'], world[1] - trial['truth_y_mm']
        row.update(detected_x_mm=world[0], detected_y_mm=world[1], detected_z_mm=world[2],
                   dx_mm=dx, dy_mm=dy, error_xy_mm=math.hypot(dx, dy))
    return row


class MeasurementSession:
    def __init__(self, folder, data):
        self.folder, self.data = Path(folder), data

    @classmethod
    def create(cls, folder, lighting):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=False)
        session = cls(folder, dict(schema=1, lighting=lighting, plan=trial_plan(), rows=[]))
        session.save([])
        return session

    @classmethod
    def load(cls, folder):
        folder = Path(folder)
        data = json.loads((folder / 'session.json').read_text(encoding='utf-8'))
        if data.get('schema') != 1 or data.get('plan') != trial_plan():
            raise ValueError('This session uses a different test plan or format.')
        rows = data.get('rows', [])
        if len(rows) > len(data['plan']) or any(
                any(row.get(k) != v for k, v in trial.items())
                for row, trial in zip(rows, data['plan'])):
            raise ValueError('The saved trials are not a consecutive prefix of the test plan.')
        session = cls(folder, data)
        session.write_csv(rows)  # Recover a CSV interrupted after the JSON commit.
        return session

    @property
    def rows(self):
        return self.data['rows']

    @property
    def next_trial(self):
        return self.data['plan'][len(self.rows)] if len(self.rows) < 180 else None

    def write_csv(self, rows):
        stream = io.StringIO(newline='')
        fields = list(make_row(trial_plan()[0], None, 0, '', ''))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        atomic_text(self.folder / 'samples.csv', stream.getvalue())

    def save(self, rows):
        updated = dict(self.data, rows=rows)
        atomic_text(self.folder / 'session.json', json.dumps(updated, indent=2, allow_nan=False))
        self.data = updated  # JSON is the authoritative transaction, CSV is a derived copy.
        self.write_csv(rows)

    def record(self, snapshot, selected, note=''):
        import numpy as np
        if self.next_trial is None:
            raise ValueError('All trials are already recorded.')
        detections = snapshot['detections']
        detection = None if selected is None else detections[selected]
        evidence = Path('captures') / f'{self.next_trial["trial_id"]:03d}_{uuid4().hex[:10]}'
        path = self.folder / evidence
        path.parent.mkdir(exist_ok=True)
        np.savez_compressed(str(path) + '.npz', rgb=snapshot['rgb'], depth_raw=snapshot['depth'])
        atomic_text(str(path) + '.json', json.dumps(dict(
            settings=snapshot['settings'], detections=detections, selected=selected,
            frame_time=snapshot['frame_time']), indent=2, allow_nan=False))
        row = make_row(self.next_trial, detection, len(detections), snapshot['frame_time'],
                       str(evidence), note)
        self.save(self.rows + [row])

    def undo(self):
        if self.rows:
            self.save(self.rows[:-1])  # Retain old captures for audit; they are no longer counted.


def statistics(rows):
    errors = [r['error_xy_mm'] for r in rows if r['position_valid']]
    n = len(rows)
    return dict(trials=n, detected=sum(r['detected'] for r in rows),
                missed=sum(not r['detected'] for r in rows), valid_positions=len(errors),
                invalid_positions=sum(r['detected'] and not r['position_valid'] for r in rows),
                mean_error_mm=sum(errors) / len(errors) if errors else None,
                max_error_mm=max(errors) if errors else None,
                color_accuracy=sum(r['predicted_color'] == r['truth_color'] for r in rows) / n if n else None,
                size_accuracy=sum(r['predicted_size'] == r['truth_size'] for r in rows) / n if n else None)


def confusion(rows):
    import numpy as np
    labels = COLORS + ('unknown', 'missed')
    counts = np.zeros((6, 8), dtype=int)
    for row in rows:
        predicted = row['predicted_color']
        counts[COLORS.index(row['truth_color']), labels.index(predicted if predicted in labels else 'unknown')] += 1
    return counts


def export_report(session):
    """Use Agg directly so export is safe inside an existing Qt application."""
    import numpy as np
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib.patches import Rectangle

    rows = session.rows
    if not rows:
        raise ValueError('Record at least one trial before exporting.')
    output = session.folder / ('report_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir()
    session.write_csv(rows)
    (output / 'samples.csv').write_bytes((session.folder / 'samples.csv').read_bytes())
    groups = {f'{point}_{size}': [r for r in rows if r['point'] == point and r['truth_size'] == size]
              for point in ('P1', 'P2', 'P3', 'P4', 'P5') for size, _ in SIZES}
    stats = dict(overall=statistics(rows), by_point_size={k: statistics(v) for k, v in groups.items()},
                 center=statistics([r for r in rows if r['point'] == 'P5']),
                 outer=statistics([r for r in rows if r['point'] != 'P5']))
    atomic_text(output / 'summary.json', json.dumps(stats, indent=2, allow_nan=False))
    averages = [s['mean_error_mm'] for s in stats['by_point_size'].values() if s['mean_error_mm'] is not None]
    norm = Normalize(0, max(averages + [1.]))
    fig = Figure(figsize=(10, 7), constrained_layout=True)
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 2)
    for ax, (size, mm) in zip(axes, SIZES):
        # These are known tag extents, not an invented physical board outline.
        ax.add_patch(Rectangle((100, -300), 300, 600, fill=False, linestyle='--', color='gray'))
        ax.scatter([0], [0], marker='s', color='black', label='Arm base')
        for i, (x, y) in enumerate(POSITIONS):
            group = stats['by_point_size'][f'P{i+1}_{size}']
            mean = group['mean_error_mm']
            if mean is not None:
                ax.scatter([x], [y], c=[mean], norm=norm, cmap='viridis', s=160, edgecolors='black')
            else:
                ax.scatter([x], [y], marker='o', facecolors='none', edgecolors='gray', s=160)
            if group['missed']:
                ax.scatter([x], [y], marker='x', color='red', s=180)
            value = f'{mean:.2f} mm' if mean is not None else 'no valid XY'
            ax.annotate(f'P{i+1}: {value}\nn={group["trials"]}, valid={group["valid_positions"]}\nmiss={group["missed"]}',
                        (x, y), xytext=(0, 12), textcoords='offset points', ha='center', fontsize=8)
        ax.set(xlim=(-35, 460), ylim=(-355, 370), xlabel='World X (mm)', ylabel='World Y (mm)',
               title=f'{size}: {mm} mm')
        ax.set_aspect('equal')
        ax.grid(alpha=.2)
    fig.colorbar(ScalarMappable(norm=norm, cmap='viridis'), ax=list(axes), label='Mean XY error (mm)')
    fig.suptitle(f'Position error: {len(rows)}/180 trials\nDashed: tag extents; red x: missed detections')
    for extension in ('png', 'pdf'):
        fig.savefig(output / f'position_error.{extension}', dpi=180)

    counts = confusion(rows)
    totals = counts.sum(axis=1, keepdims=True)
    percents = np.divide(counts * 100., totals, out=np.zeros_like(counts, dtype=float), where=totals != 0)
    fig = Figure(figsize=(10, 7), constrained_layout=True)
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    im = ax.imshow(percents, vmin=0, vmax=100, cmap='Blues')
    for y in range(6):
        for x in range(8):
            label = f'{counts[y, x]}\n{percents[y, x]:.1f}%' if totals[y, 0] else '0\nN/A'
            ax.text(x, y, label, ha='center', va='center', color='white' if percents[y, x] > 50 else 'black')
    ax.set(xticks=range(8), xticklabels=COLORS + ('unknown', 'missed'), yticks=range(6),
           yticklabels=COLORS, xlabel='Predicted color', ylabel='True color',
           title=f'Color confusion: both sizes, {len(rows)}/180 trials')
    fig.colorbar(im, ax=ax, label='Percent of true-color trials')
    for extension in ('png', 'pdf'):
        fig.savefig(output / f'color_confusion.{extension}', dpi=180)
    stream = io.StringIO(newline='')
    writer = csv.writer(stream)
    writer.writerow(['truth / prediction', *COLORS, 'unknown', 'missed'])
    writer.writerows([color, *row] for color, row in zip(COLORS, counts.tolist()))
    atomic_text(output / 'color_confusion.csv', stream.getvalue())
    overall = stats['overall']
    mean, maximum = overall['mean_error_mm'], overall['max_error_mm']
    error_text = f'{mean:.3f} / {maximum:.3f} mm' if mean is not None else 'N/A (no valid positions)'
    atomic_text(output / 'report.md', f'''# Task 3.5 detector measurements

Status: {'COMPLETE' if len(rows) == 180 else 'PARTIAL'} ({len(rows)}/180 trials).
Lighting (operator supplied): {session.data['lighting']}

Five known grid positions; each has 25 mm then 38 mm blocks, with colors
red, orange, yellow, green, blue, purple and three independent placements each.
Truth is the manually aligned block center. Position statistics use detected top-face
XY; no Z or orientation truth is asserted. The operator keeps one block on the board
and verifies the selected contour when extra detections exist. A single frame is
recorded per placement. Repeated frames are not counted as independent placements.

- Valid positions: {overall['valid_positions']}; missed: {overall['missed']}; detected with invalid XYZ: {overall['invalid_positions']}.
- Mean / maximum XY error: {error_text}.
- Color accuracy over all recorded trials (including misses): {overall['color_accuracy']:.1%}.
- Size accuracy over all recorded trials (including misses): {overall['size_accuracy']:.1%}.

![Position error](position_error.png)

Markers show mean Euclidean XY error across colors and repeats at each point and size.
Unsampled/invalid points have no assigned error; misses are shown separately.
Dashed rectangle marks tag extents, not a measured physical board boundary.
P5 (250, 0) is the test center; the other four points are 269.3 mm away.
Only two radial distances are sampled, so a continuous distance trend is not established.
Regional counts and errors are in summary.json. Calibration and detector settings
for each frame are retained in the measurement folder's captures directory; check these before
combining runs made with different settings or lighting.

![Color confusion](color_confusion.png)

Rows are true colors; columns are predictions. Percentages are normalized by all
recorded trials of that true color, including Unknown and Missed. Both sizes are pooled.
Missing trials are not counted as misses. Raw counts are in color_confusion.csv.
''')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session', type=Path, help='Folder containing session.json')
    args = parser.parse_args()
    print(export_report(MeasurementSession.load(args.session)))
