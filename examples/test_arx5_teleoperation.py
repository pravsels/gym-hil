"""Test teleoperation with the ARX5 block tower environment."""
import argparse
import time

import gymnasium as gym
import numpy as np

import gym_hil  # noqa: F401


def main():
    parser = argparse.ArgumentParser(description="ARX5 teleop test")
    parser.add_argument("--use-keyboard", action="store_true")
    parser.add_argument("--render-mode", default="human")
    parser.add_argument("--reset-delay", type=float, default=2.0)
    args = parser.parse_args()

    env_id = (
        "gym_hil/Arx5BlockTowerKeyboard-v0"
        if args.use_keyboard
        else "gym_hil/Arx5BlockTowerGamepad-v0"
    )

    env = gym.make(env_id, image_obs=True)

    print(f"Environment: {env_id}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space keys: {list(env.observation_space.spaces.keys())}")

    obs, _ = env.reset()
    print(f"Observation keys: {sorted(obs.keys())}")

    # Neutral action: no movement, gripper at midpoint
    dummy_action = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    try:
        while True:
            obs, reward, terminated, truncated, info = env.step(dummy_action)

            is_intervention = info.get("is_intervention", False)
            teleop_action = info.get("teleop_action", None)

            if is_intervention:
                ta_shape = teleop_action.shape if teleop_action is not None else "N/A"
                print(f"  Intervention active | teleop_action shape: {ta_shape}")

            if terminated or truncated:
                print("Episode ended, resetting...")
                obs, _ = env.reset()

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        env.close()
        print("Session ended")


if __name__ == "__main__":
    main()
