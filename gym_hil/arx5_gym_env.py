from pathlib import Path
from typing import Literal

import mujoco
import numpy as np
from gymnasium import spaces

from gym_hil.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv

_ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]
_ARM_ACTUATORS = [f"joint{i}" for i in range(1, 7)]
_GRIPPER_JOINT = "joint7"
_GRIPPER_ACTUATOR = "gripper"

_HOME_QPOS = np.array([0, 0.251, 0.314, 0, 0, 0, 0.044, -0.044])
_HOME_CTRL = np.array([0, 0.251, 0.314, 0, 0, 0, 0.044])


class Arx5GymEnv(MujocoGymEnv):
    """Base class for ARX L5 robot environments with joint position control."""

    def __init__(
        self,
        xml_path: Path | None = None,
        seed: int = 0,
        control_dt: float = 0.05,
        physics_dt: float = 0.002,
        render_spec: GymRenderingSpec | None = None,
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
    ):
        if xml_path is None:
            xml_path = (Path(__file__).parent / "assets" / "arx5_block_tower_scene.xml").resolve()
        if render_spec is None:
            render_spec = GymRenderingSpec(height=480, width=640)

        super().__init__(
            xml_path=xml_path,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            render_spec=render_spec,
        )

        self.render_mode = render_mode
        self.image_obs = image_obs
        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            "render_fps": int(np.round(1.0 / self.control_dt)),
        }

        # 7 observable joints: joint1..joint6 (arm) + joint7 (gripper)
        self._obs_joint_ids = np.array(
            [self._model.joint(f"joint{i}").id for i in range(1, 8)]
        )
        # All 8 joints including coupled joint8 (for qpos reset)
        self._all_joint_ids = np.array(
            [self._model.joint(f"joint{i}").id for i in range(1, 9)]
        )
        # 7 actuators: joint1..joint6 + gripper
        self._all_ctrl_ids = np.array(
            [self._model.actuator(a).id for a in _ARM_ACTUATORS]
            + [self._model.actuator(_GRIPPER_ACTUATOR).id]
        )
        # DOF addresses for velocity observation
        self._obs_dof_adrs = np.array(
            [self._model.jnt_dofadr[jid] for jid in self._obs_joint_ids]
        )

        # Camera IDs
        cam_front = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, "front")
        cam_wrist = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
        self._camera_ids = (cam_front, cam_wrist)

        # Action space: joint position targets bounded by joint limits
        arm_low = np.array([self._model.jnt_range[self._model.joint(j).id, 0] for j in _ARM_JOINTS])
        arm_high = np.array([self._model.jnt_range[self._model.joint(j).id, 1] for j in _ARM_JOINTS])
        grip_range = self._model.jnt_range[self._model.joint(_GRIPPER_JOINT).id]
        self.action_space = spaces.Box(
            low=np.concatenate([arm_low, [grip_range[0]]]).astype(np.float32),
            high=np.concatenate([arm_high, [grip_range[1]]]).astype(np.float32),
            dtype=np.float32,
        )

        self._setup_observation_space()

        # Offscreen renderer
        self._viewer = mujoco.Renderer(self._model, height=render_spec.height, width=render_spec.width)
        self._viewer.render()

    def _setup_observation_space(self):
        obs_dict = {
            "observation.state": spaces.Box(-np.inf, np.inf, (7,), dtype=np.float32),
            "observation.velocity": spaces.Box(-np.inf, np.inf, (7,), dtype=np.float32),
            "observation.effort": spaces.Box(-np.inf, np.inf, (7,), dtype=np.float32),
            "observation.eef_6d_pose": spaces.Box(-np.inf, np.inf, (6,), dtype=np.float32),
        }
        if self.image_obs:
            h, w = self._render_specs.height, self._render_specs.width
            obs_dict["pixels"] = spaces.Dict({
                "front": spaces.Box(0, 255, (h, w, 3), dtype=np.uint8),
                "wrist": spaces.Box(0, 255, (h, w, 3), dtype=np.uint8),
            })
        self.observation_space = spaces.Dict(obs_dict)

    # ---- observation helpers ----

    def get_obs_state(self) -> np.ndarray:
        """6 arm joint positions + gripper position."""
        return self._data.qpos[self._obs_joint_ids].astype(np.float32)

    def get_obs_velocity(self) -> np.ndarray:
        """6 arm joint velocities + gripper velocity."""
        return self._data.qvel[self._obs_dof_adrs].astype(np.float32)

    def get_obs_effort(self) -> np.ndarray:
        """6 arm actuator forces + gripper actuator force."""
        return self._data.actuator_force[self._all_ctrl_ids].astype(np.float32)

    def get_obs_eef_6d_pose(self) -> np.ndarray:
        """EEF position (3) + rotation vector (3)."""
        eef_pos = self._data.sensor("eef_pos").data
        eef_quat = self._data.sensor("eef_quat").data
        rotvec = np.zeros(3)
        mujoco.mju_quat2Vel(rotvec, eef_quat, 1.0)
        return np.concatenate([eef_pos, rotvec]).astype(np.float32)

    def _compute_observation(self) -> dict:
        obs = {
            "observation.state": self.get_obs_state(),
            "observation.velocity": self.get_obs_velocity(),
            "observation.effort": self.get_obs_effort(),
            "observation.eef_6d_pose": self.get_obs_eef_6d_pose(),
        }
        if self.image_obs:
            front, wrist = self.render()
            obs["pixels"] = {"front": front, "wrist": wrist}
        return obs

    # ---- control ----

    def reset_robot(self):
        """Set robot joints to home configuration."""
        self._data.qpos[self._all_joint_ids] = _HOME_QPOS
        self._data.ctrl[self._all_ctrl_ids] = _HOME_CTRL
        mujoco.mj_forward(self._model, self._data)

    def apply_action(self, action: np.ndarray):
        """Set position targets and step the simulation."""
        self._data.ctrl[self._all_ctrl_ids] = action
        for _ in range(self._n_substeps):
            mujoco.mj_step(self._model, self._data)

    def render(self):
        rendered_frames = []
        for cam_id in self._camera_ids:
            self._viewer.update_scene(self._data, camera=cam_id)
            rendered_frames.append(self._viewer.render())
        return rendered_frames
