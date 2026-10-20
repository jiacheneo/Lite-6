"""
Design tokens
--------------------------------
bg       #eef0f5   cool off-white
surface  #ffffff   card surface
border   #dde1ee   default border
black    #000000   emphasis - titles, values, button labels
grey     #656a73   all secondary / default text
accent   #5b5fc4   muted indigo (borders, progress, indicators - not text)
danger   #c41230   stop-sign red
"""

STYLE = """

/* --- Global --- */

QMainWindow, QWidget {
    background-color: #eef0f5;
    color: #656a73;
    font-family: "Inter", "Segoe UI", Ubuntu, sans-serif;
    font-size: 10pt;
}

*:focus { outline: none; }

/* --- Cards --- */

QFrame[role="card"] {
    background-color: #ffffff;
    border: 1px solid #dde1ee;
    border-radius: 12px;
}

/* --- Video card --- */

QFrame#videoCard {
    background-color: #0d0e12;
    border: none;
    border-radius: 14px;
}

QLabel#videoDisplay {
    background-color: transparent;
    border: none;
    color: #656a73;
    font-size: 14pt;
}

/* --- Right panel --- */

QFrame#rightPanel { background: transparent; border: none; }

/* --- Direct control --- */

QFrame#SliderFrame {
    background-color: #ffffff;
    border: 1px solid #dde1ee;
    border-radius: 12px;
}

QWidget#SliderControls         { background: transparent; }
QCheckBox#chk_directcontrol    { background: transparent; padding: 0px; margin: 0px; }

QFrame#SliderFrame[active="false"] {
    background-color: #f4f5f9;
    border-color: #e8eaf2;
}

QFrame#SliderFrame[active="false"] QLabel       { color: #656a73; }
QFrame#SliderFrame[active="false"] QProgressBar { background-color: #e8eaf2; }
QWidget#SliderControls:disabled QLabel          { color: #656a73; }
QWidget#SliderControls:disabled QProgressBar    { background-color: #e8eaf2; }

QFrame#SliderFrame[active="true"] QLabel[role="value"] { color: #000000; font-weight: 700; }

QFrame#CartesianJogFrame:disabled {
    background-color: #f4f5f9;
    border-color: #e8eaf2;
}

QFrame#CartesianJogFrame:disabled QLabel,
QFrame#CartesianJogFrame:disabled QRadioButton        { color: #656a73; }
QFrame#CartesianJogFrame:disabled QLabel[role="section"] { color: #000000; font-weight: 600; }

QFrame#SpeedFrame:disabled {
    background-color: #f4f5f9;
    border-color: #e8eaf2;
}

QFrame#SpeedFrame:disabled QLabel                       { color: #656a73; }
QFrame#SpeedFrame:disabled QLabel[role="section"],
QFrame#SpeedFrame:disabled QLabel[role="value"]         { color: #000000; }

/* --- Typography --- */

QLabel {
    border: none;
    background: transparent;
    color: #656a73;
}

QLabel[role="section"] {
    color: #000000;
    font-size: 10pt;
    font-weight: 600;
    letter-spacing: 0.2px;
    padding-bottom: 2px;
}

QLabel[role="ink"]        { color: #000000; }
QLabel[role="muted"]      { color: #656a73; }
QLabel[role="statusKey"]  { color: #656a73; }
QLabel[role="status"]     { color: #5b5fc4; font-weight: 500; }

QLabel[role="value"] {
    color: #000000;
    font-family: "JetBrains Mono", "Ubuntu Mono", Consolas, monospace;
    font-size: 12pt;
    font-weight: 500;
}

/* --- Buttons: base --- */

QPushButton {
    background-color: #ffffff;
    color: #000000;
    border: 1px solid #d0d4e4;
    border-radius: 8px;
    padding: 5px 12px;
    min-height: 28px;
    font-size: 10pt;
    font-weight: 400;
}

QPushButton:hover    { background-color: #eeeeff; border-color: #5b5fc4; color: #000000; }
QPushButton:pressed  { background-color: #e4e4f8; border-color: #5b5fc4; color: #000000; }
QPushButton:disabled { background-color: #f4f5f9; color: #656a73; border-color: #e4e6f0; }

QPushButton#btn_ik_calc,
QPushButton#btn_ik_goto { font-size: 9.5pt; padding: 4px 10px; min-height: 30px; }

/* --- Primary --- */

QPushButton[role="primary"] {
    background-color: #5b5fc4;
    color: #ffffff;
    border: none;
    font-weight: 600;
    min-height: 32px;
}

QPushButton[role="primary"]:hover    { background-color: #4548a8; color: #ffffff; border: none; }
QPushButton[role="primary"]:pressed  { background-color: #3840a0; color: #ffffff; border: none; }
QPushButton[role="primary"]:disabled { background-color: #b8b9e8; color: #ffffff; border: none; }

/* --- Ghost --- */

QPushButton[role="ghost"] {
    background-color: transparent;
    color: #656a73;
    border: 1px solid #dde1ee;
}

QPushButton[role="ghost"]:hover   { background-color: #f4f5fc; color: #000000; border-color: #b8bfd4; }
QPushButton[role="ghost"]:pressed { background-color: #eeeeff; color: #000000; }

/* --- Compact - jog +/- buttons --- */

QPushButton[role="compact"] {
    background-color: #5b5fc4;
    color: #ffffff;
    border: none;
    border-radius: 7px;
    font-weight: 700;
    font-size: 12pt;
    padding: 0px;
    min-height: 0px;
}

QPushButton[role="compact"]:hover    { background-color: #4548a8; }
QPushButton[role="compact"]:pressed  { background-color: #3840a0; }
QPushButton[role="compact"]:disabled { background-color: #c8c9e8; }

/* --- Danger --- */

QPushButton[role="danger"] {
    background-color: #c41230;
    color: #ffffff;
    border: none;
    font-weight: 700;
    font-size: 13pt;
    border-radius: 8px;
    min-height: 44px;
}

QPushButton[role="danger"]:hover   { background-color: #a80e28; color: #ffffff; border: none; }
QPushButton[role="danger"]:pressed { background-color: #8c0b20; color: #ffffff; border: none; }

/* --- Emergency Stop --- */

QPushButton#btn_estop {
    background-color: #c41230;
    color: #ffffff;
    border: none;
    border-radius: 10px;
    font-weight: 800;
    font-size: 14pt;
    letter-spacing: 3px;
    min-height: 50px;
}

QPushButton#btn_estop:hover   { background-color: #a80e28; }
QPushButton#btn_estop:pressed { background-color: #8c0b20; }

/* --- Progress bars --- */

QProgressBar {
    background-color: #e8eaf2;
    border: none;
    border-radius: 3px;
    max-height: 6px;
}

QProgressBar::chunk { background-color: #5b5fc4; border-radius: 3px; }

/* --- Inputs --- */

QLineEdit {
    background-color: #ffffff;
    color: #000000;
    border: 1px solid #d0d4e4;
    border-radius: 7px;
    padding: 2px 8px;
    selection-background-color: #5b5fc4;
}

QLineEdit:focus { border-color: #5b5fc4; }

/* --- Speed slider --- */

QSlider::groove:vertical   { background: #e0e3ef; width: 5px; border-radius: 2px; }
QSlider::handle:vertical   { background: #5b5fc4; width: 15px; height: 15px; margin: 0 -5px; border-radius: 8px; }
QSlider::sub-page:vertical { background: #5b5fc4; border-radius: 2px; }
QSlider::add-page:vertical  { background: #e0e3ef; border-radius: 2px; }

/* --- Checkbox / Radio --- */

QCheckBox, QRadioButton {
    background: transparent;
    color: #656a73;
    spacing: 6px;
    font-size: 10pt;
}

QCheckBox::indicator, QRadioButton::indicator {
    width: 15px;
    height: 15px;
    border: 1.5px solid #b8bfd4;
    background: #ffffff;
}

QCheckBox::indicator                    { border-radius: 4px; }
QRadioButton::indicator                 { border-radius: 8px; }
QRadioButton#radioBaseFrame::indicator,
QRadioButton#radioEEFrame::indicator    { border-radius: 3px; }

QCheckBox::indicator:hover,
QRadioButton::indicator:hover { border-color: #5b5fc4; }

QCheckBox::indicator:checked,
QRadioButton::indicator:checked {
    background-color: #5b5fc4;
    border-color: #5b5fc4;
}

"""
