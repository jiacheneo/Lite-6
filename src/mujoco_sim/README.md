# MuJoCo Sim (Lite 6)
A cross-platform software bridge of a nominal robot controller to Mujoco simulator via LCM communication protocol.

> Vendored from [ARC Bridge](https://github.com/ARCaD-Lab-UM/arc-bridge) and trimmed
> to the UFACTORY Lite 6: the base `Lcm2MujocoBridge` plus `UfactoryLite6Bridge`, and the 4 LCM types
> those need. The Python package is `lite6_sim`. The generated LCM bindings and
> `lcm_msgs.jar` are git-tracked, so the generation step below is only needed if you edit `lcm_types/`.


## Quick Start
1. Create a conda environment:
    ```sh
    conda env create -f environment_py310.yml
    conda activate env550lab
    ```
1. Install `mujoco-sim` as a Python module in editable and compatible mode (from the repo root):
    ```sh
    pip install -e src/mujoco_sim --no-deps --config-settings editable_mode=compat
    ```
1. Launch it in any terminal with `env550lab` activated:
    ```sh
    mujoco-sim # Ubuntu or Windows
    ```
    Use `--help` to find other launching options.
    If you are using macOS, launch it in this folder:
    ```sh
    mjpython lite6_sim/main.py # macOS
    ```
1. Check the communication status and visualize data:
    ```sh
    550-lcm-spy
    ```

### Re-generating LCM types
Only needed if you change anything in `lcm_types/`. Requires java - type `javac` to verify.
```sh
bash gen_lcm_types.sh # Ubuntu or macOS
gen_lcm_types_win.cmd # Windows
```
:warning: Restart MATLAB to make changes effective. Redo this if any LCM types are changed.


## Supported Robots & Controllers
- [x] UFACTORY Lite 6 (6-DOF arm)

### Lite 6 control modes
The Lite 6 model is **position controlled**: `robot_assets/UfactoryLite6/xml/robot.xml` uses
`<position kp="2000" kv="200">` actuators (the same ones upstream `mujoco_menagerie` ships in
`lite6.xml`), so `mj_data.ctrl` is a joint angle target and the servo gains live in the XML.
`ufactory_lite6_control_t.mode` selects what the bridge servos to:

| mode | target | shaped by | mirrors on the real arm |
| --- | --- | --- | --- |
| 0 | `qj_pos` - joint angles | `speed`, `mvacc` (trapezoidal profile) | `set_servo_angle`, mode 0 |
| 1 | `qj_vel` - joint velocities | `speed` (cap), `mvacc` (ramp) | `vc_set_joint_velocity`, mode 4 |
| 2 | `v_ee` - end-effector twist (m/s, rad/s) | `speed` (joint speed cap), `mvacc` (ramp) | `vc_set_cartesian_velocity`, mode 5 |

Mode 2 runs velocity (differential) IK in the bridge - a small QP
(`lite6_sim/utils/differential_ik.py`, proxsuite, with a damped-least-squares fallback) that maps the
twist to joint velocities subject to the joint speed and travel limits. `ee_frame` picks the tool
frame over the base frame. This stands in for the real arm's firmware, so the upper level streams a
twist and never solves IK itself.

`qj_tau` / `kp` / `kd` are unused under position control; they stay in the message for the generic
torque bridge.


## Add Your Custom Robot

- For a referenced integration, see the pendulum example added in commit [`bd614b0`](https://github.com/ARCaD-Lab-UM/arc-bridge/commit/bd614b0). 
- Sensors including IMU, joint encoder, foot force, CoM position and velocity are parsed automatically. Check [hopper_model.xml](https://github.com/ARCaD-Lab-UM/arc-bridge/blob/main/robot_assets/Hopper_v2/hopper_model.xml) for an example of a floating-based robot with a common sensor pack.

1. Add your robot's MuJoCo XML files in `robot_assets/`:
    ```
    robot_assets/
    └── YourRobot/
        └── your_robot.xml  # Robot and scene definitions
    ```

2. Define LCM message types in `lcm_types/`:
    ```
    lcm_types/
    ├── your_robot_state_t.lcm    # Robot state message
    └── your_robot_control_t.lcm  # Robot control message
    ```

    **Message structure example**:
    ```lcm
    // your_robot_state_t.lcm
    package lcm_msgs;
    struct your_robot_state_t {
        int64_t timestamp;
        double qj_pos[N];           // Joint positions
        double qj_vel[N];           // Joint velocities  
        double qj_tau[N];           // Joint torques
        // Add robot-specific fields as needed
    }

    // your_robot_control_t.lcm  
    package lcm_msgs;
    struct your_robot_control_t {
        int64_t timestamp;
        double qj_tau[N];           // Desired joint torques
        double qj_pos[N];           // Desired joint positions
        double qj_vel[N];           // Desired joint velocities
        double kp[N];               // Proportional gains
        double kd[N];               // Derivative gains
    }
    ```
    Replace `N` with the number of actuated joints of your robot.

3. Create a robot bridge classin `lite6_sim/bridges/your_robot_bridge.py`:
    ```python
    import mujoco
    import numpy as np

    from .lcm2mujoco_bridge import Lcm2MujocoBridge
    from lite6_sim.lcm_msgs import your_robot_state_t, your_robot_control_t
    from lite6_sim.utils import *


    class YourRobotBridge(Lcm2MujocoBridge):
        def __init__(self, mj_model, mj_data, config):
            super().__init__(mj_model, mj_data, config)

        def parse_robot_specific_low_state(self):
            """Add robot-specific state information to low_state message"""
            # Example: Add inertia matrix and bias forces
            temp_inertia_mat = np.zeros((self.mj_model.nv, self.mj_model.nv))
            mujoco.mj_fullM(self.mj_model, temp_inertia_mat, self.mj_data.qM)
            self.low_state.inertia_mat = temp_inertia_mat.tolist()
            self.low_state.bias_force = self.mj_data.qfrc_bias.tolist()
    ```

4. Register your robot to bridge imports in `lite6_sim/bridges/__init__.py`:
    ```python
    # ... existing robots ...
    from .your_robot_bridge import YourRobotBridge
    ```
5. Register your robot to launching configurations in `lite6_sim/config.py`:
    ```python
    robot_path_dict = {
        # ... existing robots ...
        "your_robot": "YourRobot/robot.xml",
    }
    ```

6. Re-generate LCM Types
    ```bash
    bash gen_lcm_types.sh     # Ubuntu/macOS
    gen_lcm_types_win.cmd     # Windows
    ```


## Dependencies
- [LCM](https://github.com/lcm-proj/lcm)
- [Mujoco](https://github.com/google-deepmind/mujoco)
- Java (LCM type generation and visualization)


## Troubleshooting

<details>
    <summary>  
        <b> For macOS users </b>
    </summary>

Use `mjpython` instead of `python` to launch the bridge.
</details>

<details>
    <summary>  
        <b> For Windows users </b>
    </summary>

Use `mujoco-sim --busywait` to avoid inaccurate system clock resolutions.
</details>

<details>
    <summary>  
        <b> LCM messages not found in MATLAB </b>
    </summary>

Restart MATLAB once after generating LCM types.
</details>

<details>
    <summary>  
        <b> GLFW Error </b>
    </summary>

```sh
GLFWError: (65542) b'GLX: No GLXFBConfigs returned'
GLFWError: (65545) b'GLX: Failed to find a suitable GLXFBConfig'
ERROR: could not create window
```
Set NVIDIA GPU as primary renderer (for systems with NVIDIA GPUs)
```
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
```
</details>

<details>
    <summary>
        <b> Unable to install LimX SDK in env550lab </b>
    </summary>

Downgrade `mujoco` to 3.2.2 and `numpy` to 1.21.6 manually.
</details>


## Citation
If you use this software, please consider citing:
```bibtex
@software{zhuang2026arcbridge,
  title={{Agile Robot Control (ARC) Bridge}},
  author={Zhuang, Yulun and Qin, Yue and Shen, Zelin},
  year={2026},
  url={https://github.com/ARCaD-Lab-UM/arc-bridge},
}
```
