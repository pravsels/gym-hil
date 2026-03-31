from typing import Any, Dict, Literal, Tuple

import mujoco
import numpy as np
from gymnasium import spaces

from gym_hil.arx5_gym_env import Arx5GymEnv
from gym_hil.mujoco_gym_env import GymRenderingSpec

_BLOCK_NAMES = ("block1", "block2", "block3", "block4")
_SAMPLING_BOUNDS = np.array([[0.12, -0.12], [0.28, 0.12]])


class Arx5BlockTowerGymEnv(Arx5GymEnv):
    """Block tower stacking task for the ARX L5 robot."""

    def __init__(
        self,
        seed: int = 0,
        control_dt: float = 0.05,
        physics_dt: float = 0.002,
        render_spec: GymRenderingSpec | None = None,
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = True,
    ):
        if render_spec is None:
            render_spec = GymRenderingSpec(height=480, width=640)

        super().__init__(
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            render_spec=render_spec,
            render_mode=render_mode,
            image_obs=image_obs,
        )

        self._block_z = [self._model.geom(name).size[2] for name in _BLOCK_NAMES]

        obs_dict = dict(self.observation_space.spaces)
        n_cubes = len(_BLOCK_NAMES)
        obs_dict["cube_poses"] = spaces.Box(-np.inf, np.inf, (n_cubes * 7,), dtype=np.float32)
        self.observation_space = spaces.Dict(obs_dict)

    def reset(
        self, seed=None, **kwargs
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self._model, self._data)
        self.reset_robot()

        for i, name in enumerate(_BLOCK_NAMES):
            xy = self._random.uniform(_SAMPLING_BOUNDS[0], _SAMPLING_BOUNDS[1])
            self._data.jnt(name).qpos[:3] = (*xy, self._block_z[i])

        mujoco.mj_forward(self._model, self._data)
        return self._compute_observation(), {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        self.apply_action(action)
        obs = self._compute_observation()
        return obs, 0.0, False, False, {}

    def _compute_observation(self) -> dict:
        obs = super()._compute_observation()
        parts = []
        for name in _BLOCK_NAMES:
            parts.append(self._data.sensor(f"{name}_pos").data)
            parts.append(self._data.sensor(f"{name}_quat").data)
        obs["cube_poses"] = np.concatenate(parts).astype(np.float32)
        return obs
