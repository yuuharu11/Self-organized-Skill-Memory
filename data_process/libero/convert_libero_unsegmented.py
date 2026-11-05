"""Convert LIBERO RLDS demos to LeRobot, keeping one episode per source demo.

Episode boundaries do not leak skill boundaries. Skill annotations from the slice
index are kept only as evaluation columns:
  eval_skill_class      int64[1]  per-frame skill class (pick0 place1 push2 open/close3 turn4)
  eval_skill_segment    int64[1]  index of the skill segment inside the demo
  eval_all_classes      string    task-level skill sequence, e.g. "[0, 1]"
  source_episode_index  int64[1]  RLDS order index (== slice-index episode_index)
The names deliberately differ from `class` / `all_classes`, which the SkillNet
transforms feed to the model as `skills`.

See docs/reproduce_core_path.md for the tfds overlay this needs (protobuf compat).
"""

from __future__ import annotations

import argparse
import functools
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import convert_libero_to_lerobot as conv  # noqa: E402
import tensorflow_datasets as tfds  # noqa: E402
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from tqdm import tqdm  # noqa: E402

_orig_create = LeRobotDataset.create


def _fast_create(*args, **kwargs):
    kwargs.setdefault("image_writer_threads", int(os.environ.get("CONV_IMAGE_THREADS", "16")))
    kwargs.setdefault("image_writer_processes", int(os.environ.get("CONV_IMAGE_PROCS", "8")))
    return _orig_create(*args, **kwargs)


LeRobotDataset.create = staticmethod(_fast_create)  # type: ignore[assignment]


def create_unsegmented(repo_id: str, sequence_state_action: bool) -> LeRobotDataset:
    state_feature = {"dtype": "float32", "shape": (8,)}
    action_feature = {"dtype": "float32", "shape": (7,)}
    if sequence_state_action:
        state_feature["sequence"] = True
        action_feature["sequence"] = True
    features = {
        "image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
        "wrist_image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
        "state": state_feature,
        "actions": action_feature,
        "eval_skill_class": {"dtype": "int64", "shape": (1,)},
        "eval_skill_segment": {"dtype": "int64", "shape": (1,)},
        "eval_all_classes": {"dtype": "string", "shape": (1,)},
        "source_episode_index": {"dtype": "int64", "shape": (1,)},
    }
    return LeRobotDataset.create(repo_id=repo_id, robot_type="panda", fps=10, features=features)


def convert_unsegmented(args: argparse.Namespace) -> None:
    preset = conv.PRESETS[args.preset]
    out = conv.output_path_for(args.output_repo_id)
    if out.exists():
        if not args.overwrite:
            raise FileExistsError(out)
        shutil.rmtree(out)
    slice_by_episode = conv.load_slice_index(args.slice_index)
    dataset = create_unsegmented(args.output_repo_id, preset["sequence_state_action"])
    episode_index = 0
    for raw_name in preset["raw_dataset_names"]:
        raw = tfds.load(raw_name, data_dir=str(args.data_dir), split="train", shuffle_files=False)
        for episode in tqdm(raw, desc=raw_name):
            steps = list(episode["steps"].as_numpy_iterator())
            item = slice_by_episode[episode_index]
            if int(item["length"]) != len(steps):
                raise ValueError(f"length mismatch at {episode_index}: {item['length']} != {len(steps)}")
            instr0 = steps[0]["language_instruction"].decode()
            if instr0 != item["task"]:
                raise ValueError(f"task mismatch at {episode_index}: {instr0!r} != {item['task']!r}")
            for seg_i, seg in enumerate(item["segments"]):
                for step in steps[int(seg["start"]):int(seg["end"])]:
                    dataset.add_frame({
                        "image": step["observation"]["image"],
                        "wrist_image": step["observation"]["wrist_image"],
                        "state": step["observation"]["state"],
                        "actions": step["action"],
                        "task": step["language_instruction"].decode(),
                        "eval_skill_class": conv.class_value_to_array(seg["class"]),
                        "eval_skill_segment": np.array([seg_i], dtype=np.int64),
                        "eval_all_classes": str(seg["all_classes"]),
                        "source_episode_index": np.array([episode_index], dtype=np.int64),
                    })
            dataset.save_episode()
            episode_index += 1
            if args.max_episodes and episode_index >= args.max_episodes:
                return
    if episode_index != len(slice_by_episode):
        raise ValueError(f"converted {episode_index} episodes, slice index has {len(slice_by_episode)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=["libero40", "libero90"], required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--slice-index", type=Path, required=True)
    p.add_argument("--output-repo-id", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max-episodes", type=int, default=0, help="stop early; for smoke tests only")
    convert_unsegmented(p.parse_args())


if __name__ == "__main__":
    main()
