# ARX5 + RewACT + gym-hil Integration Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the Franka Panda with an ARX L5 (ARX5) robot arm in gym-hil, add a 3-cube block-tower scene, wire in the rewact policy for dense reward prediction, and verify HIL infrastructure still works end-to-end.

**Architecture:** The ARX L5 uses position actuators (not torque), so we drop the operational-space controller entirely. The base env accepts 7D joint-position actions (6 arm + 1 gripper) matching rewact's output. For teleop, a Jacobian-based differential IK wrapper converts Cartesian commands to joint targets. The rewact policy is loaded as a LeRobot plugin and feeds observations through its preprocessor before calling `select_action`.

**Tech Stack:** MuJoCo, gymnasium, LeRobot (>=0.4), lerobot_policy_rewact, rewact_tools, PyTorch, huggingface_hub

---

## Conventions

- ARX L5 (mujoco_menagerie name) = ARX5 (rewact dataset name) — same robot
- All new files go under `gym_hil/` (not `gym_hil/envs/` for the base env class)
- Existing Franka envs are left untouched — we add ARX5 alongside them
- `control_dt = 0.05` (20 Hz, matching rewact training data)
- Camera resolution: 480×640 (matching rewact checkpoint)

---

## Task 1: Install Dependencies

**Files:**
- Clone: `../rewact/` (sibling to gym-hil)
- Modify: nothing

**Step 1: Clone rewact**

```bash
cd /home/user/Desktop/code
git clone https://github.com/pravsels/rewact
```

**Step 2: Install LeRobot**

```bash
pip install "lerobot>=0.4"
```

rewact is a LeRobot plugin (not a patch). The README describes standalone installable packages — no file replacements needed.

**Step 3: Install rewact packages**

```bash
cd /home/user/Desktop/code/rewact/rewact_tools
pip install -e .
cd /home/user/Desktop/code/rewact/lerobot_policy_rewact
pip install -e .
```

**Step 4: Install gym-hil**

```bash
cd /home/user/Desktop/code/gym-hil
pip install -e .
```

**Step 5: Verify imports work**

```bash
python -c "import lerobot; import lerobot_policy_rewact; import gym_hil; print('All imports OK')"
```

Expected: `All imports OK`

**Step 6: Download rewact checkpoint**

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('villekuosmanen/rewact_build_block_tower_all', local_dir='checkpoints/rewact_block_tower')
"
```

Run this from `/home/user/Desktop/code/gym-hil`.

**Step 7: Commit**

```bash
git add -A && git commit -m "add rewact checkpoint and install dependencies"
```

---

## Task 2: Copy ARX L5 Assets into gym-hil

**Files:**
- Create: `gym_hil/assets/arx_l5/` (directory with XML + meshes)

**Step 1: Copy the ARX L5 model directory**

```bash
cp -r /home/user/Desktop/code/mujoco_menagerie/arx_l5/ /home/user/Desktop/code/gym-hil/gym_hil/assets/arx_l5/
```

This copies `arx_l5.xml`, `scene.xml`, and `assets/*.obj`.

**Step 2: Verify the model loads**

```bash
python -c "
import mujoco
model = mujoco.MjModel.from_xml_path('gym_hil/assets/arx_l5/scene.xml')
print(f'nq={model.nq}, nv={model.nv}, nu={model.nu}')
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    print(f'  joint {i}: {name}')
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print(f'  actuator {i}: {name}')
"
```

Expected output:
```
nq=8, nv=8, nu=7
  joint 0: joint1
  joint 1: joint2
  joint 2: joint3
  joint 3: joint4
  joint 4: joint5
  joint 5: joint6
  joint 6: joint7
  joint 7: joint8
  actuator 0: joint1
  actuator 1: joint2
  actuator 2: joint3
  actuator 3: joint4
  actuator 4: joint5
  actuator 5: joint6
  actuator 6: gripper
```

**Step 3: Commit**

```bash
git add gym_hil/assets/arx_l5/ && git commit -m "add ARX L5 model assets from mujoco_menagerie"
```

---

## Task 3: Create the ARX5 Block Tower Scene XML

**Files:**
- Create: `gym_hil/assets/arx5_block_tower_scene.xml`

**Step 1: Write the scene XML**

This scene includes the ARX L5 model on a table, adds 3 colored cubes with freejoints, a front camera, sensors for block positions/orientations, and an EEF site for observation.

```xml
<mujoco model="ARX5 Block Tower Scene">
  <include file="arx_l5/arx_l5.xml"/>

  <option timestep=".002" noslip_iterations="5" noslip_tolerance="0"/>

  <statistic center="0.15 0 0.15" extent="0.5"/>

  <visual>
    <headlight diffuse=".4 .4 .4" ambient=".5 .5 .5"/>
    <global azimuth="160" elevation="-20" offheight="480" offwidth="640"/>
    <quality offsamples="8"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1=".3 .5 .7" rgb2="0 0 0" width="32" height="512"/>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512"
             rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
    <material name="grid" texture="grid" texrepeat="2 2" texuniform="true" reflectance="0"/>
  </asset>

  <worldbody>
    <!-- Cameras -->
    <camera name="front" pos="0.6 0.0 0.35" quat="0.5963678 0.3799282 0.3799282 0.5963678" fovy="45"/>
    <!-- Lighting -->
    <light pos="0 0 1.5" dir="0 0 -1" diffuse="1 1 1" specular=".3 .3 .3"/>
    <light pos="0 -.3 .3" mode="targetbodycom" target="block1" diffuse=".8 .8 .8" specular=".3 .3 .3"/>
    <!-- Floor -->
    <geom name="floor" size="3 3 0.01" type="plane" material="grid"/>

    <!-- Cube 1: Red (largest, bottom of stack) -->
    <body name="block1" pos="0.2 -0.05 0.025">
      <freejoint name="block1"/>
      <geom name="block1" type="box" size=".025 .025 .025" mass="0.08"
            rgba="0.9 0.15 0.15 1" friction="1.0 0.005 0.0001"/>
    </body>

    <!-- Cube 2: Green (medium) -->
    <body name="block2" pos="0.2 0.05 0.02">
      <freejoint name="block2"/>
      <geom name="block2" type="box" size=".020 .020 .020" mass="0.06"
            rgba="0.15 0.8 0.2 1" friction="1.0 0.005 0.0001"/>
    </body>

    <!-- Cube 3: Blue (smallest, top of stack) -->
    <body name="block3" pos="0.25 0.0 0.015">
      <freejoint name="block3"/>
      <geom name="block3" type="box" size=".015 .015 .015" mass="0.04"
            rgba="0.15 0.3 0.9 1" friction="1.0 0.005 0.0001"/>
    </body>
  </worldbody>

  <!-- EEF site for measuring tool-center-point -->
  <worldbody>
    <!-- This is handled by adding a site to the ARX L5 model; see Step 2 -->
  </worldbody>

  <!-- Block sensors -->
  <sensor>
    <framepos  name="block1_pos"  objtype="geom" objname="block1"/>
    <framequat name="block1_quat" objtype="geom" objname="block1"/>
    <framepos  name="block2_pos"  objtype="geom" objname="block2"/>
    <framequat name="block2_quat" objtype="geom" objname="block2"/>
    <framepos  name="block3_pos"  objtype="geom" objname="block3"/>
    <framequat name="block3_quat" objtype="geom" objname="block3"/>
  </sensor>
</mujoco>
```

**Step 2: Add an EEF site to arx_l5.xml**

The ARX L5 model needs a site between the gripper fingers for EEF pose sensing. Add inside the `<body name="link6">` element, after the gripper finger bodies:

```xml
<site name="eef" pos="0.13 0 0" size="0.001"/>
```

Position `0.13 0 0` is approximately between the two finger tips (link7 at `pos="0.08657 0.024896 ..."` and link8 at `pos="0.08657 -0.0249 ..."`; the finger collision geoms extend ~0.05 further).

Also add sensors to the copied `arx_l5.xml`:

```xml
<sensor>
  <framepos    name="eef_pos"    objtype="site" objname="eef"/>
  <framequat   name="eef_quat"   objtype="site" objname="eef"/>
  <framelinvel name="eef_vel"    objtype="site" objname="eef"/>
  <frameangvel name="eef_angvel" objtype="site" objname="eef"/>
</sensor>
```

**Step 3: Add joint sensors to arx_l5.xml**

The env needs joint position, velocity, and effort (actuator force) sensors. Add to the `arx_l5.xml`:

```xml
<sensor>
  <jointpos name="arx5/joint1_pos" joint="joint1"/>
  <jointpos name="arx5/joint2_pos" joint="joint2"/>
  <jointpos name="arx5/joint3_pos" joint="joint3"/>
  <jointpos name="arx5/joint4_pos" joint="joint4"/>
  <jointpos name="arx5/joint5_pos" joint="joint5"/>
  <jointpos name="arx5/joint6_pos" joint="joint6"/>
  <jointvel name="arx5/joint1_vel" joint="joint1"/>
  <jointvel name="arx5/joint2_vel" joint="joint2"/>
  <jointvel name="arx5/joint3_vel" joint="joint3"/>
  <jointvel name="arx5/joint4_vel" joint="joint4"/>
  <jointvel name="arx5/joint5_vel" joint="joint5"/>
  <jointvel name="arx5/joint6_vel" joint="joint6"/>
  <actuatorfrc name="arx5/joint1_effort" actuator="joint1"/>
  <actuatorfrc name="arx5/joint2_effort" actuator="joint2"/>
  <actuatorfrc name="arx5/joint3_effort" actuator="joint3"/>
  <actuatorfrc name="arx5/joint4_effort" actuator="joint4"/>
  <actuatorfrc name="arx5/joint5_effort" actuator="joint5"/>
  <actuatorfrc name="arx5/joint6_effort" actuator="joint6"/>
</sensor>
```

**Step 4: Verify scene loads with cubes**

```bash
python -c "
import mujoco
m = mujoco.MjModel.from_xml_path('gym_hil/assets/arx5_block_tower_scene.xml')
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
for name in ['block1_pos', 'block2_pos', 'block3_pos']:
    print(f'{name}: {d.sensor(name).data}')
print(f'eef_pos: {d.sensor(\"eef_pos\").data}')
print(f'nq={m.nq}, nu={m.nu}')
"
```

**Step 5: Commit**

```bash
git add gym_hil/assets/ && git commit -m "add ARX5 block tower scene with 3 cubes and sensors"
```

---

## Task 4: Create Arx5GymEnv Base Class

**Files:**
- Create: `gym_hil/arx5_gym_env.py`

This is the core environment class, analogous to `FrankaGymEnv` but for ARX5 with position control.

**Step 1: Write the base environment class**

Key differences from `FrankaGymEnv`:
- **6 arm DOF** (not 7): joints `joint1`–`joint6`, actuators named the same
- **Position actuators**: set `data.ctrl[i] = target_joint_position` directly, no opspace
- **Gripper**: 1 actuator named `gripper` driving `joint7` (coupled to `joint8` via equality constraint)
- **Action space**: 7D `[joint1_pos, ..., joint6_pos, gripper_pos]` — raw joint position targets
- **Observation format**: dict matching rewact keys exactly
- **Camera resolution**: 480×640
- **Control frequency**: 20 Hz (control_dt=0.05)

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from gym_hil.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv

# ARX L5 joint limits (from arx_l5.xml)
_ARX5_JOINT_LIMITS = np.array([
    [-3.14, 3.14],   # joint1
    [0.0,   3.14],   # joint2
    [0.0,   3.14],   # joint3
    [-1.7,  1.7],    # joint4
    [-1.7,  1.7],    # joint5
    [-3.14, 3.14],   # joint6
])
_ARX5_GRIPPER_RANGE = [0.0, 0.044]

# Home position from arx_l5.xml keyframe
_ARX5_HOME = np.array([0, 0.251, 0.314, 0, 0, 0], dtype=np.float64)
_ARX5_GRIPPER_HOME = 0.044  # open


class Arx5GymEnv(MujocoGymEnv):
    """Base class for ARX5 (ARX L5) robot environments with position control."""

    N_ARM_JOINTS = 6
    N_ACTUATORS = 7  # 6 arm + 1 gripper

    def __init__(
        self,
        xml_path: Path | None = None,
        seed: int = 0,
        control_dt: float = 0.05,   # 20 Hz matching rewact training
        physics_dt: float = 0.002,
        render_spec: GymRenderingSpec = GymRenderingSpec(height=480, width=640),
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
    ):
        if xml_path is None:
            xml_path = Path(__file__).parent / "assets" / "arx5_block_tower_scene.xml"

        super().__init__(
            xml_path=xml_path,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            render_spec=render_spec,
        )

        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            "render_fps": int(np.round(1.0 / self.control_dt)),
        }
        self.render_mode = render_mode
        self.image_obs = image_obs

        # Camera IDs
        cam_front = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, "front")
        cam_wrist = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
        self.camera_id = (cam_front, cam_wrist)

        # Cache joint/actuator IDs
        self._arm_joint_names = [f"joint{i}" for i in range(1, 7)]
        self._arm_dof_ids = np.array([
            self._model.joint(name).id for name in self._arm_joint_names
        ])
        self._arm_ctrl_ids = np.array([
            self._model.actuator(name).id for name in self._arm_joint_names
        ])
        self._gripper_ctrl_id = self._model.actuator("gripper").id
        self._gripper_joint_id = self._model.joint("joint7").id
        self._eef_site_id = self._model.site("eef").id

        # Setup spaces
        self._setup_action_space()
        self._setup_observation_space()

        # Initialize renderer
        self._viewer = mujoco.Renderer(
            self.model, height=render_spec.height, width=render_spec.width
        )
        self._viewer.render()

    def _setup_action_space(self):
        low = np.concatenate([_ARX5_JOINT_LIMITS[:, 0], [_ARX5_GRIPPER_RANGE[0]]])
        high = np.concatenate([_ARX5_JOINT_LIMITS[:, 1], [_ARX5_GRIPPER_RANGE[1]]])
        self.action_space = spaces.Box(
            low=low.astype(np.float32),
            high=high.astype(np.float32),
            dtype=np.float32,
        )

    def _setup_observation_space(self):
        state_box = spaces.Box(-np.inf, np.inf, (7,), dtype=np.float32)
        obs_dict = {
            "observation.state": state_box,
            "observation.velocity": state_box,
            "observation.effort": state_box,
            "observation.eef_6d_pose": spaces.Box(-np.inf, np.inf, (6,), dtype=np.float32),
        }
        if self.image_obs:
            h, w = self._render_specs.height, self._render_specs.width
            img_box = spaces.Box(0, 255, (h, w, 3), dtype=np.uint8)
            obs_dict["pixels"] = spaces.Dict({
                "front": img_box,
                "wrist": img_box,
            })
        self.observation_space = spaces.Dict(obs_dict)

    def reset_robot(self):
        self._data.qpos[self._arm_dof_ids] = _ARX5_HOME
        self._data.qpos[self._gripper_joint_id] = _ARX5_GRIPPER_HOME
        self._data.ctrl[self._arm_ctrl_ids] = _ARX5_HOME
        self._data.ctrl[self._gripper_ctrl_id] = _ARX5_GRIPPER_HOME
        mujoco.mj_forward(self._model, self._data)

    def apply_action(self, action):
        """Apply 7D joint position action: [joint1..joint6, gripper]."""
        action = np.asarray(action, dtype=np.float64)
        self._data.ctrl[self._arm_ctrl_ids] = action[:self.N_ARM_JOINTS]
        self._data.ctrl[self._gripper_ctrl_id] = action[self.N_ARM_JOINTS]
        for _ in range(self._n_substeps):
            mujoco.mj_step(self._model, self._data)

    def get_obs_state(self):
        """observation.state: 6 joint positions + 1 gripper position = [7]."""
        qpos = self._data.qpos[self._arm_dof_ids].astype(np.float32)
        grip = np.array([self._data.qpos[self._gripper_joint_id]], dtype=np.float32)
        return np.concatenate([qpos, grip])

    def get_obs_velocity(self):
        """observation.velocity: 6 joint velocities + 1 gripper velocity = [7]."""
        qvel_ids = [self._model.joint(n).dofadr[0] for n in self._arm_joint_names]
        qvel = self._data.qvel[qvel_ids].astype(np.float32)
        grip_vel = np.array(
            [self._data.qvel[self._model.joint("joint7").dofadr[0]]], dtype=np.float32
        )
        return np.concatenate([qvel, grip_vel])

    def get_obs_effort(self):
        """observation.effort: 6 arm actuator forces + 1 gripper force = [7]."""
        efforts = np.array([
            self._data.actuator_force[self._arm_ctrl_ids[i]]
            for i in range(self.N_ARM_JOINTS)
        ], dtype=np.float32)
        grip_effort = np.array(
            [self._data.actuator_force[self._gripper_ctrl_id]], dtype=np.float32
        )
        return np.concatenate([efforts, grip_effort])

    def get_obs_eef_6d_pose(self):
        """observation.eef_6d_pose: 3 position + 3 rotation vector = [6]."""
        pos = self._data.site_xpos[self._eef_site_id].astype(np.float32)
        rot_mat = self._data.site_xmat[self._eef_site_id].reshape(3, 3)
        rot_vec = np.zeros(3)
        mujoco.mju_mat2Quat(rot_quat := np.zeros(4), rot_mat.flatten())
        mujoco.mju_quat2Vel(rot_vec, rot_quat, 1.0)
        return np.concatenate([pos, rot_vec.astype(np.float32)])

    def get_gripper_pose(self):
        return np.array([self._data.qpos[self._gripper_joint_id]], dtype=np.float32)

    def render(self):
        rendered_frames = []
        for cam_id in self.camera_id:
            self._viewer.update_scene(self.data, camera=cam_id)
            rendered_frames.append(self._viewer.render())
        return rendered_frames
```

**Step 2: Verify it instantiates**

```bash
python -c "
from gym_hil.arx5_gym_env import Arx5GymEnv
env = Arx5GymEnv()
env.reset_robot()
print('action_space:', env.action_space)
print('obs_space:', env.observation_space)
print('state:', env.get_obs_state())
print('eef:', env.get_obs_eef_6d_pose())
"
```

**Step 3: Commit**

```bash
git add gym_hil/arx5_gym_env.py && git commit -m "add Arx5GymEnv base class with position control"
```

---

## Task 5: Create Arx5BlockTowerGymEnv (Task Environment)

**Files:**
- Create: `gym_hil/envs/arx5_block_tower_gym_env.py`

This subclass handles:
- Episode reset with randomized cube placement
- Observation computation (rewact-compatible format + cube poses for bookkeeping)
- Reward = 0 by default (rewact provides the reward signal externally)
- Termination after max steps (no manual reward/success check)

```python
from pathlib import Path
from typing import Any, Dict, Literal, Tuple

import mujoco
import numpy as np
from gymnasium import spaces

from gym_hil.arx5_gym_env import Arx5GymEnv
from gym_hil.mujoco_gym_env import GymRenderingSpec

_N_CUBES = 3
_CUBE_SAMPLING_BOUNDS = np.array([[0.12, -0.12], [0.28, 0.12]])


class Arx5BlockTowerGymEnv(Arx5GymEnv):
    """ARX5 block tower environment — reward comes from rewact, not hardcoded."""

    def __init__(
        self,
        seed: int = 0,
        control_dt: float = 0.05,
        physics_dt: float = 0.002,
        render_spec: GymRenderingSpec = GymRenderingSpec(height=480, width=640),
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = True,
    ):
        super().__init__(
            xml_path=Path(__file__).parent.parent / "assets" / "arx5_block_tower_scene.xml",
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            render_spec=render_spec,
            render_mode=render_mode,
            image_obs=image_obs,
        )

        self.n_cubes = _N_CUBES
        self._cube_names = [f"block{i}" for i in range(1, _N_CUBES + 1)]

        # Extend observation space with cube poses (for HIL bookkeeping)
        cube_obs = spaces.Box(-np.inf, np.inf, (_N_CUBES * 7,), dtype=np.float32)
        obs_spaces = dict(self.observation_space.spaces)
        obs_spaces["cube_poses"] = cube_obs
        self.observation_space = spaces.Dict(obs_spaces)

    def reset(self, seed=None, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self._model, self._data)
        self.reset_robot()

        # Randomize cube positions on the table surface
        for i, name in enumerate(self._cube_names):
            xy = self._random.uniform(_CUBE_SAMPLING_BOUNDS[0], _CUBE_SAMPLING_BOUNDS[1])
            block_z = self._model.geom(name).size[2]
            self._data.joint(name).qpos[:3] = [xy[0], xy[1], block_z]
            self._data.joint(name).qpos[3:7] = [1, 0, 0, 0]  # identity quaternion

        mujoco.mj_forward(self._model, self._data)
        return self._compute_observation(), {}

    def step(self, action) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        self.apply_action(action)
        obs = self._compute_observation()
        # Reward is 0 by default — rewact provides dense reward externally
        return obs, 0.0, False, False, {}

    def _compute_observation(self) -> dict:
        obs = {
            "observation.state": self.get_obs_state(),
            "observation.velocity": self.get_obs_velocity(),
            "observation.effort": self.get_obs_effort(),
            "observation.eef_6d_pose": self.get_obs_eef_6d_pose(),
        }

        # Cube poses: [pos(3) + quat(4)] * 3 = 21 floats
        cube_data = []
        for name in self._cube_names:
            pos = self._data.sensor(f"{name}_pos").data.astype(np.float32)
            quat = self._data.sensor(f"{name}_quat").data.astype(np.float32)
            cube_data.extend([pos, quat])
        obs["cube_poses"] = np.concatenate(cube_data)

        if self.image_obs:
            front_view, wrist_view = self.render()
            obs["pixels"] = {"front": front_view, "wrist": wrist_view}

        return obs
```

**Step 2: Update `gym_hil/envs/__init__.py`**

Add the new env class to exports:

```python
from gym_hil.envs.arx5_block_tower_gym_env import Arx5BlockTowerGymEnv
__all__ = ["PandaPickCubeGymEnv", "PandaArrangeBoxesGymEnv", "Arx5BlockTowerGymEnv"]
```

**Step 3: Register new env IDs in `gym_hil/__init__.py`**

Add registrations alongside existing ones:

```python
from gym_hil.arx5_gym_env import Arx5GymEnv

register(
    id="gym_hil/Arx5BlockTowerBase-v0",
    entry_point="gym_hil.envs:Arx5BlockTowerGymEnv",
    max_episode_steps=200,  # 200 steps * 0.05s = 10 seconds
)
```

(Gamepad/keyboard variants are added in Task 7 after the wrappers are updated.)

**Step 4: Verify the task env**

```bash
python -c "
import gymnasium as gym
import gym_hil
env = gym.make('gym_hil/Arx5BlockTowerBase-v0', image_obs=True)
obs, info = env.reset()
print('Obs keys:', list(obs.keys()))
print('state shape:', obs['observation.state'].shape)
print('velocity shape:', obs['observation.velocity'].shape)
print('effort shape:', obs['observation.effort'].shape)
print('eef shape:', obs['observation.eef_6d_pose'].shape)
print('cube_poses shape:', obs['cube_poses'].shape)
if 'pixels' in obs:
    print('front img shape:', obs['pixels']['front'].shape)
    print('wrist img shape:', obs['pixels']['wrist'].shape)
action = env.action_space.sample()
obs, rew, term, trunc, info = env.step(action)
print('Stepped OK, reward:', rew)
env.close()
"
```

**Step 5: Commit**

```bash
git add gym_hil/envs/ gym_hil/__init__.py && git commit -m "add Arx5BlockTowerGymEnv with 3 cubes"
```

---

## Task 6: Update Wrappers for ARX5

**Files:**
- Modify: `gym_hil/wrappers/hil_wrappers.py`
- Modify: `gym_hil/wrappers/factory.py`
- Modify: `gym_hil/__init__.py`

The key challenge: the `InputsControlWrapper` generates 4D Cartesian actions (xyz + gripper) but the ARX5 env expects 7D joint position actions. We need a `CartesianToJointWrapper` that uses differential IK to convert.

**Step 1: Add CartesianToJointWrapper to hil_wrappers.py**

This wrapper sits between `InputsControlWrapper` (which outputs 4D) and the base env (which expects 7D joint). It uses the Jacobian pseudoinverse for differential IK.

```python
class CartesianToJointWrapper(gym.ActionWrapper):
    """Converts 4D Cartesian actions [dx, dy, dz, gripper] to 7D joint position targets.

    Uses Jacobian-pseudoinverse differential IK.
    """

    def __init__(self, env, step_scale=0.02):
        super().__init__(env)
        self.step_scale = step_scale

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 0.044], dtype=np.float32),
            dtype=np.float32,
        )

    def action(self, action):
        base_env = self.env.unwrapped
        dof_ids = base_env._arm_dof_ids
        site_id = base_env._eef_site_id
        model = base_env._model
        data = base_env._data

        dx_cart = action[:3] * self.step_scale

        # Compute Jacobian
        J_pos = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jacSite(model, data, J_pos, None, site_id)
        J = J_pos[:, dof_ids]  # (3, 6)

        # Pseudoinverse IK: dq = J^+ @ dx
        dq = np.linalg.lstsq(J, dx_cart, rcond=None)[0]

        # Current joint positions + delta
        q_current = data.qpos[dof_ids].copy()
        q_target = q_current + dq

        # Clip to joint limits
        from gym_hil.arx5_gym_env import _ARX5_JOINT_LIMITS
        q_target = np.clip(q_target, _ARX5_JOINT_LIMITS[:, 0], _ARX5_JOINT_LIMITS[:, 1])

        gripper_target = action[3]
        return np.concatenate([q_target.astype(np.float32), [gripper_target]])
```

**Step 2: Update factory.py**

Add support for ARX5 env creation:

```python
from gym_hil.envs.arx5_block_tower_gym_env import Arx5BlockTowerGymEnv

def wrap_arx5_env(
    env,
    use_viewer=False,
    use_gamepad=False,
    use_inputs_control=False,
    auto_reset=False,
    show_ui=True,
    reset_delay_seconds=1.0,
    controller_config_path=None,
):
    """Apply wrappers suitable for ARX5 environments."""
    from gym_hil.wrappers.hil_wrappers import (
        CartesianToJointWrapper,
        InputsControlWrapper,
        ResetDelayWrapper,
    )
    from gym_hil.wrappers.viewer_wrapper import PassiveViewerWrapper

    env = CartesianToJointWrapper(env)

    if use_inputs_control:
        env = InputsControlWrapper(
            env,
            x_step_size=1.0,
            y_step_size=1.0,
            z_step_size=1.0,
            use_gripper=True,
            auto_reset=auto_reset,
            use_gamepad=use_gamepad,
            controller_config_path=controller_config_path,
        )

    if use_viewer:
        env = PassiveViewerWrapper(env, show_left_ui=show_ui, show_right_ui=show_ui)

    env = ResetDelayWrapper(env, delay_seconds=reset_delay_seconds)
    return env


def make_env(env_id, **kwargs):
    # ... existing code ...
    elif env_id == "gym_hil/Arx5BlockTowerBase-v0":
        env = Arx5BlockTowerGymEnv(**env_kwargs)
        return wrap_arx5_env(env, **wrapper_kwargs)
```

**Step 3: Register wrapped ARX5 env IDs in `gym_hil/__init__.py`**

```python
register(
    id="gym_hil/Arx5BlockTowerGamepad-v0",
    entry_point="gym_hil.wrappers.factory:make_env",
    max_episode_steps=200,
    kwargs={
        "env_id": "gym_hil/Arx5BlockTowerBase-v0",
        "use_viewer": True,
        "use_inputs_control": True,
        "use_gamepad": True,
    },
)

register(
    id="gym_hil/Arx5BlockTowerKeyboard-v0",
    entry_point="gym_hil.wrappers.factory:make_env",
    max_episode_steps=200,
    kwargs={
        "env_id": "gym_hil/Arx5BlockTowerBase-v0",
        "use_viewer": True,
        "use_inputs_control": True,
        "use_gamepad": False,
    },
)
```

**Step 4: Update InputsControlWrapper gripper command for ARX5 range**

The existing wrapper maps gripper to `[0.0, 2.0]` (Franka). For ARX5, gripper range is `[0.0, 0.044]`. Rather than changing the existing wrapper, the `CartesianToJointWrapper` handles this mapping — the gamepad outputs `[0, 2]` and the wrapper maps to `[0, 0.044]`. Update `CartesianToJointWrapper.action()` to remap:

```python
gripper_raw = action[3]  # from InputsControlWrapper: 0=close, 1=neutral, 2=open
gripper_target = (gripper_raw / 2.0) * 0.044  # map [0,2] → [0, 0.044]
```

**Step 5: Commit**

```bash
git add gym_hil/wrappers/ gym_hil/__init__.py && git commit -m "add ARX5 wrappers with Cartesian-to-joint IK"
```

---

## Task 7: Wire in the RewACT Policy

**Files:**
- Create: `examples/run_rewact.py`

This script:
1. Loads the rewact checkpoint
2. Creates the ARX5 block tower env
3. Runs episodes with the policy acting
4. Uses rewact's predicted reward as the env reward signal

**Step 1: Write the inference script**

```python
"""Run the rewact policy in the ARX5 block tower sim environment."""
import argparse
import torch
import numpy as np
from pathlib import Path

import gym_hil  # register envs
import gymnasium as gym


def load_rewact_policy(checkpoint_path: str, device: str = "cuda"):
    """Load a rewact policy from a LeRobot-style checkpoint."""
    from lerobot.common.policies.factory import make_policy_from_checkpoint

    policy = make_policy_from_checkpoint(checkpoint_path)
    policy.to(device)
    policy.eval()
    return policy


def obs_to_batch(obs: dict, device: str = "cuda") -> dict:
    """Convert gym observation dict to rewact-compatible batch dict."""
    batch = {}

    # State observations — add batch dim and convert to tensor
    for key in ["observation.state", "observation.velocity",
                "observation.effort", "observation.eef_6d_pose"]:
        val = obs[key]
        batch[key] = torch.from_numpy(val).unsqueeze(0).float().to(device)

    # Image observations — convert HWC uint8 → CHW float [0,1], add batch dim
    if "pixels" in obs:
        for cam_name in ["front", "wrist"]:
            img = obs["pixels"][cam_name]  # (H, W, 3) uint8
            img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            batch[f"observation.images.{cam_name}"] = img_t.to(device)

    return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/rewact_block_tower")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    # Load policy
    print(f"Loading rewact policy from {args.checkpoint}...")
    policy = load_rewact_policy(args.checkpoint, device=device)
    print("Policy loaded.")

    # Create environment
    env = gym.make("gym_hil/Arx5BlockTowerBase-v0", image_obs=True)

    episode_returns = []

    for ep in range(args.episodes):
        obs, info = env.reset()
        episode_return = 0.0
        step_count = 0

        policy.reset()

        while True:
            batch = obs_to_batch(obs, device=device)

            with torch.no_grad():
                action_tensor, reward_pred = policy.select_action(batch)

            action = action_tensor.squeeze(0).cpu().numpy()
            reward = reward_pred.item()

            obs, _, terminated, truncated, info = env.step(action)
            episode_return += reward
            step_count += 1

            if terminated or truncated:
                break

        episode_returns.append(episode_return)
        print(f"Episode {ep+1}/{args.episodes}: "
              f"steps={step_count}, return={episode_return:.4f}")

    env.close()

    # Summary
    returns = np.array(episode_returns)
    print(f"\n--- Summary ({args.episodes} episodes) ---")
    print(f"Mean return: {returns.mean():.4f}")
    print(f"Std return:  {returns.std():.4f}")
    print(f"Min return:  {returns.min():.4f}")
    print(f"Max return:  {returns.max():.4f}")

    if returns.std() > 0.001:
        print("PASS: Reward varies across episodes (dense reward working)")
    else:
        print("WARNING: Reward is nearly constant — check policy/obs format")


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add examples/run_rewact.py && git commit -m "add rewact policy inference script"
```

---

## Task 8: ARX5 Teleop Test

**Files:**
- Create: `examples/test_arx5_teleoperation.py`

**Step 1: Write the teleop test**

```python
"""Test teleoperation with the ARX5 block tower environment."""
import argparse
import time

import gymnasium as gym
import numpy as np

import gym_hil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-keyboard", action="store_true")
    args = parser.parse_args()

    env_id = ("gym_hil/Arx5BlockTowerKeyboard-v0" if args.use_keyboard
              else "gym_hil/Arx5BlockTowerGamepad-v0")

    env = gym.make(env_id, image_obs=True)
    obs, _ = env.reset()

    print(f"Env: {env_id}")
    print(f"Action space: {env.action_space}")
    print(f"Obs keys: {list(obs.keys())}")

    dummy_action = np.array([0.0, 0.0, 0.0, 0.022], dtype=np.float32)

    try:
        while True:
            obs, reward, terminated, truncated, info = env.step(dummy_action)

            is_intervention = info.get("is_intervention", False)
            teleop_action = info.get("teleop_action", None)

            if is_intervention:
                print(f"  intervention! teleop_action shape: "
                      f"{teleop_action.shape if teleop_action is not None else 'N/A'}")

            if terminated or truncated:
                print("Episode ended, resetting...")
                obs, _ = env.reset()

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        env.close()


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add examples/test_arx5_teleoperation.py && git commit -m "add ARX5 teleoperation test script"
```

---

## Task 9: Smoke Test

**Step 1: Run 10 episodes with rewact**

```bash
python examples/run_rewact.py --episodes 10
```

**Checklist:**
- [ ] All 10 episodes complete without errors
- [ ] Cube positions change between episodes (randomization works)
- [ ] Robot arm moves (joint positions change from home)
- [ ] Reward varies across episodes (std > 0.001)
- [ ] No NaN in observations or actions

**Step 2: Test teleop**

```bash
python examples/test_arx5_teleoperation.py --use-keyboard
```

**Checklist:**
- [ ] Env launches without errors
- [ ] `info["is_intervention"]` toggles when pressing keys
- [ ] `info["teleop_action"]` has shape `(4,)` during intervention
- [ ] Robot moves in response to keyboard input
- [ ] Cubes are physically stable (don't fly off)

---

## Key Architectural Decisions

| Decision | Rationale |
|----------|-----------|
| Position control (no opspace) | ARX L5 uses position actuators; rewact outputs joint targets directly |
| 7D action space [joint1–6 + gripper] | Matches rewact's 7D output and the ARX L5 actuator layout |
| 480×640 camera resolution | Matches rewact training data (`[3, 480, 640]`) |
| 20 Hz control rate | Matches rewact training FPS (dataset at 20 fps) |
| Jacobian-pseudoinverse IK for teleop | Simplest approach to convert Cartesian gamepad commands to joint targets |
| Reward = 0 in env, rewact provides reward | The rewact policy predicts dense reward — no manual reward function needed |
| `cube_poses` in obs (bookkeeping only) | Available for HIL-RL logging; rewact ignores it (uses cameras) |
| Existing Franka envs untouched | ARX5 is added alongside, not replacing |

## Known Limitations / Future Work

- **EEF rotation teleop**: The current CartesianToJointWrapper only maps XYZ translation + gripper. Rotation control via gamepad is not wired up (matches the existing Franka wrapper behavior).
- **eef_6d_pose dimension**: The rewact checkpoint config says `[7]` but the dataset says `[6]`. We use `[6]` (pos + rotation vector). If the policy errors, try appending gripper width to make it `[7]`.
- **Observation normalization**: The rewact policy expects normalized observations (MEAN_STD). The `obs_to_batch` function in `run_rewact.py` passes raw values. The policy's built-in preprocessor should handle normalization, but if rewards look wrong, verify normalizer stats match.
