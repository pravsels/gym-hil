"""Interactive camera tuner for MuJoCo scene cameras.

Usage:
  ~/micromamba/micromamba run -n gym-hil python examples/tune_front_camera.py

Move the free camera with mouse, then press Ctrl+C in terminal.
The script prints a copy-paste <camera .../> line using pos + xyaxes.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


def _format_camera_xml(
    camera_name: str,
    pos: np.ndarray,
    right: np.ndarray,
    up: np.ndarray,
    fovy: float,
) -> str:
    return (
        f'<camera name="{camera_name}" '
        f'pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}" '
        f'xyaxes="{right[0]:.6f} {right[1]:.6f} {right[2]:.6f} '
        f'{up[0]:.6f} {up[1]:.6f} {up[2]:.6f}" '
        f'fovy="{fovy:.0f}"/>'
    )


def _replace_camera_line(xml_path: Path, camera_name: str, new_line: str) -> bool:
    text = xml_path.read_text()
    pattern = rf'<camera\s+name="{re.escape(camera_name)}"[^>]*/>'
    updated, n = re.subn(pattern, new_line, text, count=1)
    if n == 0:
        return False
    xml_path.write_text(updated)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive MuJoCo camera tuner.")
    parser.add_argument(
        "--xml",
        type=Path,
        default=Path("gym_hil/assets/arx5_block_tower_scene.xml"),
        help="Scene XML path.",
    )
    parser.add_argument(
        "--camera-name",
        type=str,
        default="front",
        help="Camera name used in generated XML line.",
    )
    parser.add_argument(
        "--fovy",
        type=float,
        default=60.0,
        help="fovy to include in generated XML line.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Overwrite camera line in XML on exit.",
    )
    args = parser.parse_args()

    xml_path = args.xml.resolve()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    # Home robot pose (arm + two-finger coupled joints) for consistent framing.
    data.qpos[:8] = np.array([0.0, 0.251, 0.314, 0.0, 0.0, 0.0, 0.044, -0.044], dtype=np.float64)
    mujoco.mj_forward(model, data)

    scene = mujoco.MjvScene(model, maxgeom=10000)
    opt = mujoco.MjvOption()
    pert = mujoco.MjvPerturb()

    last_line = ""
    print("Tune camera with mouse in viewer. Press Ctrl+C in terminal to print final XML line.")
    print("Tip: rotate/pan/zoom until view looks right, then Ctrl+C.")

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            while viewer.is_running():
                mujoco.mjv_updateScene(
                    model,
                    data,
                    opt,
                    pert,
                    viewer.cam,
                    mujoco.mjtCatBit.mjCAT_ALL,
                    scene,
                )
                gl_cam = scene.camera[0]
                pos = np.array(gl_cam.pos, dtype=np.float64)
                forward = np.array(gl_cam.forward, dtype=np.float64)
                up = np.array(gl_cam.up, dtype=np.float64)
                right = np.cross(forward, up)
                right /= np.linalg.norm(right)
                up /= np.linalg.norm(up)
                last_line = _format_camera_xml(args.camera_name, pos, right, up, args.fovy)

                viewer.sync()
                time.sleep(1 / 120)
    except KeyboardInterrupt:
        pass

    if not last_line:
        print("No camera sample captured.")
        return

    print("\nCopy-paste camera line:")
    print(last_line)

    if args.apply:
        ok = _replace_camera_line(xml_path, args.camera_name, last_line)
        if ok:
            print(f"\nUpdated {xml_path} camera '{args.camera_name}'.")
        else:
            print(f"\nCould not find camera '{args.camera_name}' in {xml_path}.")


if __name__ == "__main__":
    main()

