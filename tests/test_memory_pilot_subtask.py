import pathlib
import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "skill_moe/skillnet/src"))

from openpi.models.pi0_moe_skill import subtask_generated_lengths
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.training.config_moe_skill import subtask_task_indices_for_split
from openpi.transforms import AttachLiberoSubtask, TokenizeSubtaskInference, TokenizeSubtaskTraining


def test_libero_subtask_transform_preserves_rows_and_maps_no_segment():
    transform = AttachLiberoSubtask({"move the object": ["pick the cup", "place the cup"]})
    rows = [
        {"prompt": "move the object", "eval_skill_segment": i, "row_id": i}
        for i in (-1, 0, 1)
    ]
    output = [transform(row) for row in rows]
    assert len(output) == len(rows)
    assert [row["row_id"] for row in output] == [-1, 0, 1]
    assert [row["subtask"] for row in output] == ["none", "pick the cup", "place the cup"]


def test_subtask_prefix_and_training_target_format_and_eos():
    tokenizer = PaligemmaTokenizer(max_len=96)
    prompt = "pick the black bowl"
    inference = TokenizeSubtaskInference(tokenizer)({"prompt": prompt})
    prefix_len = int(inference["tokenized_prompt_mask"].sum())
    prefix = tokenizer._tokenizer.decode(inference["tokenized_prompt"][:prefix_len].tolist())
    assert prefix.endswith("Task: pick the black bowl. Subtask: ")

    target = "pick the bowl"
    train = TokenizeSubtaskTraining(tokenizer)({"prompt": prompt, "subtask": target})
    ids = train["tokenized_prompt"]
    mask = train["tokenized_prompt_mask"]
    loss_mask = train["token_loss_mask"]
    assert int(mask.sum()) == int(loss_mask.sum()) + prefix_len
    assert ids[int(mask.sum()) - 1] == 1
    decoded = tokenizer._tokenizer.decode(ids[prefix_len : int(mask.sum()) - 1].tolist())
    assert decoded.strip() == target


def test_decoder_eos_is_included_and_absent_eos_uses_max_length():
    tokens = jnp.asarray([[7, 8, 1, 9, 0], [4, 5, 6, 0, 0]], dtype=jnp.int32)
    np.testing.assert_array_equal(np.asarray(subtask_generated_lengths(tokens)), [3, 5])
    tokenizer = PaligemmaTokenizer(max_len=96)
    encoded = tokenizer._tokenizer.encode("pick the cup")
    assert tokenizer.detokenize(encoded + [1, 17, 0]) == "pick the cup"


def test_research_split_task_indices_exclude_libero10():
    split_a = subtask_task_indices_for_split("A")
    split_b = subtask_task_indices_for_split("B")
    assert split_a == tuple(range(20, 30)) + tuple(range(30, 40))
    assert split_b == tuple(range(20, 30)) + tuple(range(10, 20))
    assert not set(range(10)).intersection(split_a)
    assert not set(range(10)).intersection(split_b)
