"""
UI layout for the Arm Lab control station.
Python-only - no .ui file needed. Edit this file to change the layout.

Design roles
--------------------------------
Labels  : section · muted · value · status
Buttons : danger · primary · ghost · compact  (default = secondary via base QSS)
Cards   : QFrame[role="card"]  (white, rounded, subtle border)
Video   : QFrame#videoCard     (dark, rounded - frames the camera feed)
"""

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QPainter
from PyQt5.QtWidgets import (
    QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QCheckBox,
    QRadioButton, QFrame, QSlider, QProgressBar, QWidget, QSizePolicy, QLineEdit, QGridLayout, QSpinBox,
)
from lite6arm import JOINT_NAMES

# Narrow labels for the joint readout and the IK result columns, where the full
# JOINT_NAMES would not fit. Same order as JOINT_NAMES.
JOINT_LABELS_SHORT = ('Base', 'Shldr', 'Elbow', 'F.Roll', 'W.Ptch', 'W.Roll')


class ToggleSwitch(QWidget):
    """Smooth sliding toggle. Emits stateChanged(Qt.Checked / Qt.Unchecked)."""

    stateChanged = pyqtSignal(int)
    _OFF = (189, 189, 189)   # #bdbdbd
    _ON  = (91,  95,  196)   # #5b5fc4 - matches theme accent

    def __init__(self, parent=None):
        super().__init__(parent)
        self._checked = False
        self._pos = 0.0        # 0.0 = off, 1.0 = on
        self.setFixedSize(46, 24)
        self.setCursor(Qt.PointingHandCursor)
        self._timer = QTimer(self)
        self._timer.setInterval(16)  # ~60 fps
        self._timer.timeout.connect(self._step)

    def _step(self):
        target = 1.0 if self._checked else 0.0
        self._pos += 0.25 if self._pos < target else -0.25
        self._pos = max(0.0, min(1.0, self._pos))
        self.update()
        if abs(self._pos - target) < 0.01:
            self._pos = target
            self._timer.stop()

    def isChecked(self):
        return self._checked

    def setChecked(self, val):
        val = bool(val)
        if self._checked != val:
            self._checked = val
            self._timer.start()
            self.stateChanged.emit(Qt.Checked if val else Qt.Unchecked)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.setChecked(not self._checked)

    def paintEvent(self, _event):
        t = self._pos
        r = int(self._OFF[0] + (self._ON[0] - self._OFF[0]) * t)
        g = int(self._OFF[1] + (self._ON[1] - self._OFF[1]) * t)
        b = int(self._OFF[2] + (self._ON[2] - self._OFF[2]) * t)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(r, g, b))
        p.drawRoundedRect(0, 3, 46, 18, 9, 9)
        p.setBrush(QColor('#ffffff'))
        p.drawEllipse(int(t * 24), 1, 22, 22)


def _label(text, role=None, mono=False, bold=False, pt=None, align=None, min_w=None):
    w = QLabel(text)
    f = w.font()
    changed = False
    if mono:  f.setFamily('Ubuntu Mono, Consolas, monospace'); changed = True
    if pt:    f.setPointSize(pt);  changed = True
    if bold:  f.setBold(True);     changed = True
    if changed: w.setFont(f)
    if align is not None: w.setAlignment(align)
    if min_w: w.setMinimumWidth(min_w)
    if role:  w.setProperty('role', role)
    return w


def _button(text, role=None, name=None):
    w = QPushButton(text)
    if name: w.setObjectName(name)
    if role: w.setProperty('role', role)
    return w


def _card(title=None, padding=(16, 14, 16, 14), spacing=10):
    """White rounded card. Returns (QFrame, QVBoxLayout)."""
    frame = QFrame()
    frame.setProperty('role', 'card')
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(*padding)
    layout.setSpacing(spacing)
    if title:
        title_label = _label(title, role='section', align=Qt.AlignVCenter)
        title_label.setFixedHeight(24)
        layout.addWidget(title_label)
    return frame, layout


def _row(*widgets, spacing=6):
    """Horizontal row of equal-stretch widgets."""
    h = QHBoxLayout()
    h.setSpacing(spacing)
    for w in widgets:
        h.addWidget(w)
    return h


class Ui_MainWindow:

    def setupUi(self, MainWindow):
        MainWindow.setWindowTitle('Arm Lab')
        MainWindow.resize(1260, 934)
        MainWindow.setMinimumSize(1260, 820)

        central = QWidget()
        MainWindow.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setSpacing(12)
        root.setContentsMargins(14, 14, 14, 14)

        # --- LEFT COLUMN ---
        video_col = QVBoxLayout()
        video_col.setSpacing(10)
        root.addLayout(video_col, stretch=1)

        # Video card - dark bezel
        video_card = QFrame()
        video_card.setObjectName('videoCard')
        self.videoCard = video_card
        vc = QVBoxLayout(video_card)
        vc.setContentsMargins(8, 8, 8, 8)
        vc.setSpacing(0)
        self.videoDisplay = QLabel()
        self.videoDisplay.setObjectName('videoDisplay')
        self.videoDisplay.setMinimumSize(320, 180)
        self.videoDisplay.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.videoDisplay.setAlignment(Qt.AlignCenter)
        self.videoDisplay.setCursor(Qt.CrossCursor)
        self.videoDisplay.setMouseTracking(True)
        vc.addWidget(self.videoDisplay)
        video_col.addWidget(video_card, stretch=1)

        # Control bar - transparent row (no card background)
        bar_row = QHBoxLayout()
        bar_row.setSpacing(14)
        bar_row.setContentsMargins(6, 4, 6, 4)
        video_col.addLayout(bar_row)

        for obj, text, checked in [('radioVideo',     'RGB',       True),
                                    ('radioDepth',     'Depth',     False),
                                    ('radioTags',      'Tags',      False),
                                    ('radioWorkspace', 'Workspace', False),
                                    ('radioMask',      'Mask',      False)]:
            rb = QRadioButton(text)
            rb.setObjectName(obj)
            rb.setChecked(checked)
            bar_row.addWidget(rb)
            setattr(self, obj, rb)

        bar_row.addStretch()
        bar_row.addWidget(_label('pixel', role='muted'))
        self.rdoutMousePixels = _label('(u, v)', role='value')
        self.rdoutMousePixels.setObjectName('rdoutMousePixels')
        bar_row.addWidget(self.rdoutMousePixels)
        bar_row.addSpacing(20)
        bar_row.addWidget(_label('depth (mm)', role='muted'))
        self.rdoutMouseDepth = _label('-', role='value')
        self.rdoutMouseDepth.setObjectName('rdoutMouseDepth')
        bar_row.addWidget(self.rdoutMouseDepth)
        bar_row.addSpacing(20)
        bar_row.addWidget(_label('world (mm)', role='muted'))
        self.rdoutMouseWorld = _label('(x, y, z)', role='value')
        self.rdoutMouseWorld.setObjectName('rdoutMouseWorld')
        bar_row.addWidget(self.rdoutMouseWorld)

        # Direct control panel - left half only, right half reserved
        dc_row = QHBoxLayout()
        dc_row.setSpacing(8)
        video_col.addLayout(dc_row)

        self.SliderFrame = QFrame()
        self.SliderFrame.setObjectName('SliderFrame')
        self.SliderFrame.setProperty('active', False)
        dc_row.addWidget(self.SliderFrame, stretch=5)

        # --- IK Control Station ---
        self.IKFrame = QFrame()
        self.IKFrame.setObjectName('IKFrame')
        self.IKFrame.setMaximumWidth(566)   # IK card (two solver columns) + Cartesian Jog
        dc_row.addWidget(self.IKFrame, stretch=5)

        # IK Solver (left) and Cartesian Jog (right) sit side by side
        ikf = QHBoxLayout(self.IKFrame)
        ikf.setContentsMargins(0, 0, 0, 0)
        ikf.setSpacing(8)

        # --- IK Solver card ---
        iks_frame, iks = _card('IK Solver  ·  World Frame', padding=(10, 6, 10, 6), spacing=4)
        ikf.addWidget(iks_frame, stretch=5)

        # Top: inputs (left) and joint results (right) side by side
        top = QHBoxLayout(); top.setSpacing(10); top.setContentsMargins(0, 0, 0, 0)
        iks.addLayout(top)

        in_col = QGridLayout()
        in_col.setContentsMargins(0, 0, 0, 0)
        in_col.setHorizontalSpacing(6)
        in_col.setVerticalSpacing(4)
        self.ik_inputs = {}
        input_axes = [('X', 'mm'), ('Y', 'mm'), ('Z', 'mm'),
                      ('Roll', 'deg'), ('Pitch', 'deg'), ('Yaw', 'deg')]
        for row_idx, (axis, unit) in enumerate(input_axes):
            in_col.setRowMinimumHeight(row_idx, 30)
            in_col.addWidget(_label(axis, role='muted', min_w=34), row_idx, 0)
            le = QLineEdit('0.0')
            le.setObjectName(f'ik_{axis.lower()}')
            le.setAlignment(Qt.AlignRight)
            le.setFixedSize(64, 30)
            self.ik_inputs[axis] = le
            in_col.addWidget(le, row_idx, 1)
            in_col.addWidget(_label(unit, role='muted', min_w=24), row_idx, 2)
        top.addLayout(in_col)

        # Two solver columns side by side; control_station names them at runtime
        # (the first column is the closed form in sim, the SDK on the real arm).
        res_col = QGridLayout()
        res_col.setContentsMargins(0, 0, 0, 0)
        res_col.setHorizontalSpacing(6)
        res_col.setVerticalSpacing(4)
        self.ik_col_headers = []
        for col_idx in (1, 2):
            hdr = _label('', role='muted')
            hdr.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            hdr.setMinimumWidth(78)   # fits the widest reading, "+360.00°"
            self.ik_col_headers.append(hdr)
            res_col.addWidget(hdr, 0, col_idx)
        self.ik_result_labels = []      # column A: geometric (sim) / SDK (real)
        self.ik_result_labels_num = []  # column B: numerical
        for row_idx, joint_name in enumerate(JOINT_LABELS_SHORT):
            res_col.addWidget(_label(joint_name, role='muted', min_w=44), row_idx + 1, 0)
            for col_idx, labels in ((1, self.ik_result_labels),
                                    (2, self.ik_result_labels_num)):
                lbl = _label('-', role='value')
                lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                lbl.setMinimumWidth(78)
                labels.append(lbl)
                res_col.addWidget(lbl, row_idx + 1, col_idx)
        top.addLayout(res_col, stretch=1)

        # Bottom: action buttons
        self.btn_ik_calc = _button('Calculate IK', name='btn_ik_calc')
        self.btn_ik_calc.setFixedSize(112, 30)
        self.btn_ik_clear = _button('Clear', name='btn_ik_clear')
        self.btn_ik_clear.setFixedSize(64, 30)
        self.btn_ik_goto = _button('Go To Pose', role='primary', name='btn_ik_goto')
        self.btn_ik_goto.setFixedSize(104, 30)
        self.btn_ik_goto.setEnabled(False)
        ik_btns = QHBoxLayout()
        ik_btns.setSpacing(6)
        ik_btns.addStretch()
        ik_btns.addWidget(self.btn_ik_calc)
        ik_btns.addWidget(self.btn_ik_clear)
        ik_btns.addWidget(self.btn_ik_goto)
        ik_btns.addStretch()
        iks.addStretch()
        iks.addLayout(ik_btns)

        # --- Cartesian Jog card ---
        cj_frame, cj = _card(padding=(10, 6, 10, 6), spacing=4)
        cj_frame.setObjectName('CartesianJogFrame')
        cj_frame.setFixedWidth(150)
        self.CartesianJogFrame = cj_frame
        ikf.addWidget(cj_frame)

        cj_title = _label('Cartesian Jog', role='section')
        cj_title.setFixedHeight(24)
        cj.addWidget(cj_title)

        self.radioBaseFrame = QRadioButton('Base')
        self.radioBaseFrame.setObjectName('radioBaseFrame')
        self.radioBaseFrame.setChecked(True)
        self.radioEEFrame = QRadioButton('EE')
        self.radioEEFrame.setObjectName('radioEEFrame')

        cj_grid = QGridLayout()
        cj_grid.setContentsMargins(0, 0, 0, 0)
        cj_grid.setHorizontalSpacing(6)
        cj_grid.setVerticalSpacing(4)

        self.cart_jog_buttons = []  # (btn, axis_idx, direction)
        for row_idx, (axis_name, axis_idx) in enumerate([('X', 0), ('Y', 1), ('Z', 2), ('Roll', 3), ('Pitch', 4), ('Yaw', 5)]):
            cj_grid.setRowMinimumHeight(row_idx, 30)
            cj_grid.addWidget(_label(axis_name, role='muted', min_w=34), row_idx, 0)
            btn_pos = _button('+', role='compact')
            btn_neg = _button('-', role='compact')
            btn_pos.setFixedSize(30, 30)
            btn_neg.setFixedSize(30, 30)
            cj_grid.addWidget(btn_pos, row_idx, 1)
            cj_grid.addWidget(btn_neg, row_idx, 2)
            self.cart_jog_buttons.append((btn_neg, axis_idx, -1))
            self.cart_jog_buttons.append((btn_pos, axis_idx,  1))
        cj.addLayout(cj_grid)

        cj.addStretch()
        cj.addLayout(_row(self.radioBaseFrame, self.radioEEFrame, spacing=8))

        sf = QHBoxLayout(self.SliderFrame)
        sf.setContentsMargins(10, 6, 10, 6)
        sf.setSpacing(10)

        dc_header = QHBoxLayout()
        dc_header.setSpacing(10)
        dc_header.setContentsMargins(0, 0, 0, 0)
        dc_title = _label('Direct Control', role='section', align=Qt.AlignVCenter)
        dc_title.setFixedHeight(24)
        dc_header.addWidget(dc_title)
        dc_header.addStretch()
        self.chk_directcontrol = QCheckBox()
        self.chk_directcontrol.setObjectName('chk_directcontrol')
        self.chk_directcontrol.setFixedHeight(24)
        dc_header.addWidget(self.chk_directcontrol)

        left_col = QVBoxLayout()
        left_col.setContentsMargins(0, 0, 0, 0)
        left_col.setSpacing(6)
        left_col.addLayout(dc_header)
        sf.addLayout(left_col, stretch=1)

        self.SliderControls = QWidget()
        self.SliderControls.setObjectName('SliderControls')
        self.SliderControls.setEnabled(False)
        left_col.addWidget(self.SliderControls, stretch=1)

        joints_col = QVBoxLayout(self.SliderControls)
        joints_col.setContentsMargins(0, 0, 0, 0)
        joints_col.setSpacing(6)

        for lbl_text, pos_name, neg_name, prog_name, rdout_name in [
            ('Base',         'btnBasePos',    'btnBaseNeg',    'progBase',     'rdoutBase'),
            ('Shoulder',     'btnShoulderPos','btnShoulderNeg','progShoulder', 'rdoutShoulder'),
            ('Elbow',        'btnElbowPos',   'btnElbowNeg',   'progElbow',    'rdoutElbow'),
            ('Forearm Roll', 'btnWristAPos',  'btnWristANeg',  'progWristA',   'rdoutWristA'),
            ('Wrist Pitch',  'btnWristRPos',  'btnWristRNeg',  'progWristR',   'rdoutWristR'),
            ('Wrist Roll',   'btnWristBPos',  'btnWristBNeg',  'progWristB',   'rdoutWristB'),
        ]:
            row = QHBoxLayout()
            row.setSpacing(8)
            row.addWidget(_label(lbl_text, role='muted', min_w=100))

            prog = QProgressBar()
            prog.setObjectName(prog_name)
            prog.setRange(0, 100); prog.setValue(0)
            prog.setTextVisible(False); prog.setFixedHeight(6)
            row.addWidget(prog, stretch=1)
            setattr(self, prog_name, prog)

            rdout = _label('+0.0°', role='value', align=Qt.AlignCenter, min_w=74)
            rdout.setObjectName(rdout_name)
            row.addWidget(rdout)
            setattr(self, rdout_name, rdout)

            btn_pos = _button('+', role='compact', name=pos_name)
            btn_pos.setFixedSize(30, 30)
            row.addWidget(btn_pos)
            setattr(self, pos_name, btn_pos)

            btn_neg = _button('-', role='compact', name=neg_name)
            btn_neg.setFixedSize(30, 30)
            row.addWidget(btn_neg)
            setattr(self, neg_name, btn_neg)

            joints_col.addLayout(row)

        speed_frame, speed = _card('Speed', padding=(8, 8, 8, 8), spacing=6)
        speed_frame.setObjectName('SpeedFrame')
        speed_frame.setFixedWidth(80)
        self.SpeedFrame = speed_frame
        dc_row.addWidget(speed_frame)

        speed.setAlignment(Qt.AlignHCenter)
        self.sldrMoveTime = QSlider(Qt.Vertical)
        self.sldrMoveTime.setObjectName('sldrMoveTime')
        self.sldrMoveTime.setRange(1, 100)
        self.sldrMoveTime.setValue(20)
        self.sldrMoveTime.setInvertedAppearance(True)
        speed.addWidget(self.sldrMoveTime, stretch=1, alignment=Qt.AlignHCenter)
        self.rdoutMoveTime = _label('20%', role='value', align=Qt.AlignCenter)
        self.rdoutMoveTime.setObjectName('rdoutMoveTime')
        speed.addWidget(self.rdoutMoveTime)

        # Status bar
        stat_frame, stat = _card(padding=(12, 8, 12, 8), spacing=0)
        stat_row = QHBoxLayout()
        stat_row.setSpacing(8)
        stat.addLayout(stat_row)
        stat_row.addWidget(_label('Status', role='statusKey'))
        self.rdoutStatus = QLabel('Waiting for input')
        self.rdoutStatus.setObjectName('rdoutStatus')
        self.rdoutStatus.setProperty('role', 'status')
        self.rdoutStatus.setWordWrap(True)
        stat_row.addWidget(self.rdoutStatus, stretch=1)
        video_col.addWidget(stat_frame)

        # --- RIGHT PANEL ---
        right_frame = QFrame()
        right_frame.setObjectName('rightPanel')
        right_frame.setFixedWidth(290)
        root.addWidget(right_frame)

        rp = QVBoxLayout(right_frame)
        rp.setSpacing(10)
        rp.setContentsMargins(0, 0, 0, 0)

        # --- Emergency Stop - full-width, standalone ---
        self.btn_estop = _button('STOP', role='danger', name='btn_estop')
        rp.addWidget(self.btn_estop)

        # --- Cards: Joint Angles + End Effector (side by side) ---
        ja_frame, ja = _card('Joint Angles', padding=(10, 10, 10, 10), spacing=5)
        for axis, attr in zip(JOINT_LABELS_SHORT,
                              ['rdoutBaseJC', 'rdoutShoulderJC', 'rdoutElbowJC',
                               'rdoutWristAJC', 'rdoutWristRJC', 'rdoutWristBJC']):
            r = QHBoxLayout(); r.setSpacing(4)
            r.addWidget(_label(axis, role='muted', min_w=44))
            v = _label('+0.0°', role='value', min_w=62)   # fits "+360.0°"
            v.setObjectName(attr); setattr(self, attr, v)
            r.addWidget(v, stretch=1)
            ja.addLayout(r)

        ee_frame, ee = _card('End Effector', padding=(10, 10, 10, 10), spacing=5)
        for axis, attr, placeholder in [('X',     'rdoutX',     '+0.00 mm'),
                                        ('Y',     'rdoutY',     '+0.00 mm'),
                                        ('Z',     'rdoutZ',     '+0.00 mm'),
                                        ('Roll',  'rdoutPhi',   '+0.00°'),
                                        ('Pitch', 'rdoutTheta', '+0.00°'),
                                        ('Yaw',   'rdoutPsi',   '+0.00°')]:
            r = QHBoxLayout(); r.setSpacing(4)
            r.addWidget(_label(axis, role='muted', min_w=34))
            # 90px fits the widest reading, "-500.00 mm" (88px); without it Qt
            # squeezes this column and clips the unit off the X/Y/Z readouts.
            v = _label(placeholder, role='value', min_w=90)
            v.setObjectName(attr); setattr(self, attr, v)
            r.addWidget(v, stretch=1)
            ee.addLayout(r)

        je_row = QHBoxLayout()
        je_row.setSpacing(6)
        je_row.addWidget(ja_frame)
        je_row.addWidget(ee_frame)
        rp.addLayout(je_row)

        # --- Card: Arm Control ---
        act_frame, act = _card('Arm Control', padding=(14, 14, 14, 14), spacing=6)
        rp.addWidget(act_frame)

        self.btn_initial_pose = _button('Initial Pose', name='btn_initial_pose')
        self.btn_sleep_arm    = _button('Sleep Arm',    name='btn_sleep_arm')
        act.addLayout(_row(self.btn_initial_pose, self.btn_sleep_arm))

        # --- Card: Gripper Control ---
        # Three gripper buttons on one row get their labels clipped inside the
        # fixed-width right panel, so they wrap onto two rows here instead.
        grip_frame, grip = _card('Gripper Control', padding=(14, 14, 14, 14), spacing=6)
        rp.addWidget(grip_frame)

        self.btn_open_gripper  = _button('Open Gripper',  name='btn_open_gripper')
        self.btn_close_gripper = _button('Close Gripper', name='btn_close_gripper')
        self.btn_stop_gripper  = _button('Stop Gripper',  name='btn_stop_gripper')
        self.btn_open_gripper.setEnabled(False)
        self.btn_close_gripper.setEnabled(False)
        self.btn_stop_gripper.setEnabled(False)
        grip.addLayout(_row(self.btn_open_gripper, self.btn_close_gripper))
        grip.addWidget(self.btn_stop_gripper)

        # --- Card: Waypoint Recorder ---
        pb_frame, pb = _card('Waypoint Recorder', padding=(14, 14, 14, 14), spacing=6)
        rp.addWidget(pb_frame)

        self.btn_add_wp    = _button('Add Waypoint',    name='btn_add_wp')
        self.btn_clear_wps = _button('Clear All', name='btn_clear_wps')
        pb.addLayout(_row(self.btn_add_wp, self.btn_clear_wps))

        self.btn_playback_wps = _button('Playback Waypoints', name='btn_playback_wps')
        pb.addWidget(self.btn_playback_wps)

        # --- Card: Camera ---
        cam_frame, cam = _card('Camera', padding=(14, 14, 14, 14), spacing=6)
        rp.addWidget(cam_frame)

        self.btn_calibrate = _button('Calibrate', name='btn_calibrate')
        cam.addWidget(self.btn_calibrate)
        self.btn_flatten_table = _button('Flatten Table (4 corners)', name='btn_flatten_table')
        cam.addWidget(self.btn_flatten_table)
        self.btn_depth_bias = _button('Correct Depth Bias', name='btn_depth_bias')
        self.btn_depth_bias.setToolTip('Fit the current frame inside the four table corners; keep the area mostly bare table.')
        self.btn_clear_depth_bias = _button('Clear Depth Bias', name='btn_clear_depth_bias')
        cam.addLayout(_row(self.btn_depth_bias, self.btn_clear_depth_bias))
        self.depth_bias_status = _label('Depth bias: OFF', role='ink', pt=10)
        cam.addWidget(self.depth_bias_status)
        height_row = QHBoxLayout()
        height_row.addWidget(QLabel('Height band (mm):'))
        self.height_min = QSpinBox()
        self.height_min.setRange(0, 299)
        self.height_min.setValue(15)
        self.height_max = QSpinBox()
        self.height_max.setRange(1, 300)
        self.height_max.setValue(60)
        height_row.addWidget(self.height_min)
        height_row.addWidget(QLabel('to'))
        height_row.addWidget(self.height_max)
        cam.addLayout(height_row)
        self.btn_board_plane = _button('Measure Board Z (2.8b)', name='btn_board_plane')
        self.btn_board_plane.setToolTip('Measure world-Z error across the empty board.')
        cam.addWidget(self.btn_board_plane)
        self.btn_fixture = _button('Measure Tag Fixture (2.8c)', name='btn_fixture')
        cam.addWidget(self.btn_fixture)
        self.btn_detector_measurement = _button("Measure Detector (3.5)", name="btn_detector_measurement")
        cam.addWidget(self.btn_detector_measurement)

        # --- Card: Mode Select ---
        auto_frame, auto = _card(padding=(14, 14, 14, 14), spacing=6)
        rp.addWidget(auto_frame)

        manual_row = QHBoxLayout()
        manual_row.setSpacing(10)
        manual_row.addWidget(_label('Manual Mode', role='ink', bold=True, pt=11))
        manual_row.addStretch()
        self.chk_manual_mode = ToggleSwitch()
        self.chk_manual_mode.setObjectName('chk_manual_mode')
        manual_row.addWidget(self.chk_manual_mode)
        auto.addLayout(manual_row)

        pick_row = QHBoxLayout()
        pick_row.setSpacing(10)
        pick_lbl = _label('Click Pick & Place', role='ink', bold=True, pt=11)
        pick_row.addWidget(pick_lbl)
        pick_row.addStretch()
        self.chk_pick_place = ToggleSwitch()
        self.chk_pick_place.setObjectName('chk_pick_place')
        pick_row.addWidget(self.chk_pick_place)
        auto.addLayout(pick_row)

        rp.addStretch()

        # Widget groups - controller iterates these; keep in sync with the joint loop above
        self.joint_readouts     = [self.rdoutBaseJC,  self.rdoutShoulderJC, self.rdoutElbowJC,
                                    self.rdoutWristAJC, self.rdoutWristRJC,  self.rdoutWristBJC]
        self.joint_jog_readouts = [self.rdoutBase,    self.rdoutShoulder,   self.rdoutElbow,
                                    self.rdoutWristA,  self.rdoutWristR,    self.rdoutWristB]
        self.joint_progs        = [self.progBase,     self.progShoulder,    self.progElbow,
                                    self.progWristA,   self.progWristR,     self.progWristB]
        self.jog_buttons        = [  # (neg, pos) in joint order
            (self.btnBaseNeg,     self.btnBasePos),
            (self.btnShoulderNeg, self.btnShoulderPos),
            (self.btnElbowNeg,    self.btnElbowPos),
            (self.btnWristANeg,   self.btnWristAPos),
            (self.btnWristRNeg,   self.btnWristRPos),
            (self.btnWristBNeg,   self.btnWristBPos),
        ]
