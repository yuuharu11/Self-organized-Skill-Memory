import numpy as np

from skill_moe.skillnet.examples.emergent.light.intermediate.task_ambiguity import (
    first_intervals_mask, retrieval_metrics, split_goal_episodes,
)


def _rows():
    episodes, tasks, frames = [], [], []
    for task in range(10, 20):
        for episode_num in range(2):
            ep = f"goal-{task}-{episode_num}"
            for frame in (20, 0, 10, 30):
                episodes.append(ep)
                tasks.append(task)
                frames.append(frame)
    for episode_num in range(2):
        for frame in (0, 10, 20):
            episodes.append(f"object-{episode_num}")
            tasks.append(20)
            frames.append(frame)
    n = len(episodes)
    return {"episode_id": np.asarray(episodes), "task_index": np.asarray(tasks),
            "frame": np.asarray(frames), "a": np.zeros((n, 25, 7), np.float32)}


def test_goal_episode_holdout_is_disjoint_and_covers_all_ten_tasks():
    train, query, heldout = split_goal_episodes(_rows(), seed=0, holdout_fraction=0.5)
    assert len(heldout) == 10
    assert set(query["task_index"]) == set(range(10, 20))
    assert not set(train["episode_id"]) & set(query["episode_id"])
    assert set(train["task_index"]) == set(range(10, 21))


def test_early_intervals_are_chosen_by_frame_order_per_episode():
    rows = _rows()
    early = first_intervals_mask(rows, count=2)
    for ep in np.unique(rows["episode_id"]):
        selected = rows["frame"][early & (rows["episode_id"] == ep)]
        assert sorted(selected.tolist()) == [0, 10]


def test_retrieval_metrics_reduce_prediction_draws_before_row_masking():
    query = {"task_index": np.asarray([10, 11]),
             "subtask_text": np.asarray(["pick", "place"])}
    memory = {"task_index": np.asarray([10, 11]),
              "subtask_text": np.asarray(["pick", "place"])}
    gold = np.zeros((2, 3), np.float32)
    memory_b = np.asarray([[0, 0, 0], [1, 0, 0]], np.float32)
    actions = np.zeros((2, 2, 1), np.float32)
    predictions = np.zeros((4, 2, 2, 1), np.float32)
    metric = retrieval_metrics(query, memory, gold, memory_b, [0, 1],
                               predictions, actions, early_mask=np.asarray([True, False]))
    assert metric["all"]["same_task_rate"] == 1.0
    assert metric["early"]["n"] == 1
    assert metric["all"]["iii_action_mse"] == 0.0
