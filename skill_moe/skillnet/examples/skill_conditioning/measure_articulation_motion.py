"""Measure end-effector motion while an articulated object is moving in LIBERO-90 demos.

For each demo, the articulated joint is taken as the non-robot state dimension with the
largest net change. Frames where that joint moves faster than 20% of its peak speed form
the manipulation segment, and only that segment is summarized.

Only frame-independent quantities are compared across scenes (displacement, path
curvature, wrist rotation, gripper width). Movement direction is printed in world
coordinates and is not comparable across scenes, because furniture orientation differs.

Usage:
  python measure_articulation_motion.py /path/to/LIBERO/libero_90
  python measure_articulation_motion.py /path/to/LIBERO/libero_90 --files KITCHEN_SCENE7_open_the_microwave_demo
Requires numpy, h5py, scipy.
"""

import argparse

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

# Skill class in SkillNet's instruct2plan_obj_90.json is given in brackets.
DEFAULT_FILES = [
    "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_demo",  # [3]
    "KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet_demo",  # [3]
    "KITCHEN_SCENE2_open_the_top_drawer_of_the_cabinet_demo",  # [3]
    "KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet_demo",  # [3]
    "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet_demo",  # [3]
    "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_demo",  # [3]
    "KITCHEN_SCENE7_open_the_microwave_demo",  # [3]
    "KITCHEN_SCENE6_close_the_microwave_demo",  # [3]
    "KITCHEN_SCENE3_turn_on_the_stove_demo",  # [4]
    "KITCHEN_SCENE8_turn_off_the_stove_demo",  # [4]
]

# states = [time, 7 arm qpos, 2 gripper qpos, objects ...]
FIRST_OBJECT_DIM = 10
SPEED_FRACTION = 0.2


def summarize_demo(demo):
    states = demo["states"][:]
    net = np.abs(states[-1, FIRST_OBJECT_DIM:] - states[0, FIRST_OBJECT_DIM:])
    joint = FIRST_OBJECT_DIM + int(np.argmax(net))
    speed = np.abs(np.diff(states[:, joint]))
    moving = np.where(speed > SPEED_FRACTION * speed.max())[0]
    start, end = moving[0], moving[-1] + 1

    pos = demo["obs/ee_pos"][:]
    ori = demo["obs/ee_ori"][:]
    grip = demo["obs/gripper_states"][:]
    disp = pos[end] - pos[start]
    dist = np.linalg.norm(disp)
    path = np.sum(np.linalg.norm(np.diff(pos[start : end + 1], axis=0), axis=1))
    relative = Rotation.from_rotvec(ori[end]) * Rotation.from_rotvec(ori[start]).inv()
    return {
        "joint": joint,
        "joint_change": float(net.max()),
        "direction": disp / max(dist, 1e-6),
        "displacement_cm": dist * 100,
        "curvature": path / max(dist, 1e-6),
        "rotation_deg": float(np.degrees(np.linalg.norm(relative.as_rotvec()))),
        "gripper_width_cm": float((grip[start:end, 0] - grip[start:end, 1]).mean() * 100),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("libero90_dir", help="directory containing the LIBERO-90 *_demo.hdf5 files")
    parser.add_argument("--files", nargs="+", default=DEFAULT_FILES, help="demo file stems without .hdf5")
    args = parser.parse_args()

    print("file | demos | joint change | displacement cm | curvature | wrist rotation deg | gripper cm | direction (world)")
    for stem in args.files:
        with h5py.File(f"{args.libero90_dir}/{stem}.hdf5", "r") as f:
            rows = [summarize_demo(f["data"][key]) for key in f["data"]]
        mean = {k: np.mean([r[k] for r in rows], axis=0) for k in rows[0] if k != "joint"}
        direction = ",".join(f"{v:+.2f}" for v in mean["direction"])
        print(
            f"{stem} | {len(rows)} | {mean['joint_change']:.2f} | {mean['displacement_cm']:.1f} | "
            f"{mean['curvature']:.2f} | {mean['rotation_deg']:.1f} | {mean['gripper_width_cm']:.1f} | ({direction})"
        )


if __name__ == "__main__":
    main()
