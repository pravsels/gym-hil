"""Run the rewact policy in the ARX5 block tower sim environment.

Quickstart (with MuJoCo viewer):
    python examples/run_rewact.py --render --episodes 3
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

import gym_hil  # noqa: F401
import gymnasium as gym

EPS = 1e-8

# Features the model actually consumes (observation.state + images).
# The other input_features (velocity, effort, eef_6d_pose) are listed in
# the config but the RewACT architecture only reads OBS_STATE and image
# features.  We still normalize and include them so the batch looks
# identical to what the preprocessor would produce.
STATE_FEATURES = [
    "observation.state",
    "observation.velocity",
    "observation.effort",
    "observation.eef_6d_pose",
]
IMAGE_FEATURES = [
    "observation.images.front",
    "observation.images.wrist",
]


def _remap_checkpoint_keys(state_dict):
    """Remap checkpoint keys from old naming to current rewact naming convention.

    The checkpoint uses model.backbone.* but the current code expects
    model.vision_encoder.resnet_feature_extractor.*, and
    model.encoder_img_feat_input_proj.* → model.vision_encoder.feat_proj.*.
    """
    remap = {
        "model.backbone.": "model.vision_encoder.resnet_feature_extractor.",
        "model.encoder_img_feat_input_proj.": "model.vision_encoder.feat_proj.",
    }
    remapped = {}
    for key, val in state_dict.items():
        new_key = key
        for old_prefix, new_prefix in remap.items():
            if key.startswith(old_prefix):
                new_key = new_prefix + key[len(old_prefix):]
                break
        remapped[new_key] = val
    return remapped


def load_policy(checkpoint_path: str, device: str = "cuda"):
    import os
    from lerobot_policy_rewact.modeling_rewact import RewACTPolicy
    from safetensors.torch import load_file as _lf

    policy = RewACTPolicy.from_pretrained(checkpoint_path)

    ckpt = _lf(os.path.join(checkpoint_path, "model.safetensors"))
    ckpt = _remap_checkpoint_keys(ckpt)
    missing, unexpected = policy.load_state_dict(ckpt, strict=False)
    if missing:
        logging.warning("Still missing after remap: %s", missing)
    if unexpected:
        logging.warning("Still unexpected after remap: %s", unexpected)

    policy.to(device)
    policy.eval()
    return policy


def load_norm_stats(checkpoint_path: str, device: str = "cuda"):
    """Load MEAN_STD normalisation statistics from the checkpoint."""
    pre_path = Path(checkpoint_path) / "policy_preprocessor_step_3_normalizer_processor.safetensors"
    post_path = Path(checkpoint_path) / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"

    raw = load_file(str(pre_path))
    stats = {}
    for feat in STATE_FEATURES + IMAGE_FEATURES:
        mean = raw[f"{feat}.mean"].to(device)
        std = raw[f"{feat}.std"].to(device)
        stats[feat] = {"mean": mean, "std": std}

    raw_post = load_file(str(post_path))
    stats["action"] = {
        "mean": raw_post["action.mean"].to(device),
        "std": raw_post["action.std"].to(device),
    }
    return stats


def normalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / (std + EPS)


def unnormalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return x * (std + EPS) + mean


def obs_to_batch(obs, stats, device="cuda"):
    batch = {}

    for key in ("observation.state", "observation.velocity", "observation.effort"):
        arr = torch.from_numpy(obs[key]).float().to(device)
        arr = normalize(arr, stats[key]["mean"], stats[key]["std"])
        batch[key] = arr.unsqueeze(0)

    # eef_6d_pose: env produces (6,), policy config says (7,).
    # The 7th element is gripper width — same as observation.state[-1].
    eef = obs["observation.eef_6d_pose"]
    gripper = obs["observation.state"][-1:]
    eef_padded = np.concatenate([eef, gripper])
    eef_t = torch.from_numpy(eef_padded).float().to(device)
    eef_t = normalize(eef_t, stats["observation.eef_6d_pose"]["mean"], stats["observation.eef_6d_pose"]["std"])
    batch["observation.eef_6d_pose"] = eef_t.unsqueeze(0)

    if "pixels" in obs:
        for cam in ("front", "wrist"):
            key = f"observation.images.{cam}"
            img = obs["pixels"][cam]  # (H, W, 3) uint8
            img_t = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0  # (3, H, W)
            img_t = img_t.to(device)
            img_t = normalize(img_t, stats[key]["mean"], stats[key]["std"])
            batch[key] = img_t.unsqueeze(0)

    return batch


def main():
    parser = argparse.ArgumentParser(description="Run rewact policy in ARX5 sim")
    parser.add_argument("--checkpoint", default="checkpoints/rewact_block_tower")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    print(f"Loading rewact policy from {args.checkpoint}...")
    policy = load_policy(args.checkpoint, device=device)
    stats = load_norm_stats(args.checkpoint, device=device)
    print("Policy loaded.")

    env = gym.make("gym_hil/Arx5BlockTowerBase-v0", image_obs=True)

    if args.render:
        from gym_hil.wrappers.viewer_wrapper import PassiveViewerWrapper
        env = PassiveViewerWrapper(env)

    episode_returns = []

    for ep in range(args.episodes):
        obs, _ = env.reset()
        episode_return = 0.0
        step_count = 0

        policy.reset()

        while True:
            batch = obs_to_batch(obs, stats, device=device)
            with torch.no_grad():
                action_norm, reward_pred = policy.select_action(batch)

            action = unnormalize(
                action_norm.float(),
                stats["action"]["mean"],
                stats["action"]["std"],
            )
            action = action.squeeze(0).cpu().numpy()
            reward = reward_pred.item() if isinstance(reward_pred, torch.Tensor) else float(reward_pred)

            obs, _, terminated, truncated, info = env.step(action)
            episode_return += reward
            step_count += 1

            if terminated or truncated:
                break

        episode_returns.append(episode_return)
        print(f"Episode {ep+1}/{args.episodes}: steps={step_count}, return={episode_return:.4f}")

    env.close()

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
