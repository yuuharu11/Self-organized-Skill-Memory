import numpy as np
import torch

from skill_moe.skillnet.examples.memory_pilot.unseen_libero import seg_context_run as runner
from skill_moe.skillnet.examples.memory_pilot.unseen_libero.seg_context_run import build_unseen_queries
from skill_moe.skillnet.examples.memory_pilot.unseen_libero import seg_context_retrieval as sc


def test_eval_stride10_filter_and_previous_velocity(tmp_path):
    # Four overlapping windows at starts 0, 5, 10, 15 reconstruct one causal
    # trace. Official Eval retains the two original stride-10 windows only.
    n = 4
    actions = np.zeros((n, 10, 7), np.float32)
    for i in range(n):
        actions[i, :, 0] = np.arange(i * 5, i * 5 + 10, dtype=np.float32)
    arrays = {
        "tokens": np.zeros((n, 2, 4), np.float16),
        "groups": np.zeros((n, 2), np.int8),
        "task": np.zeros(n, np.int32),
        "episode": np.zeros(n, np.int32),
        "frame": np.asarray([0, 5, 10, 15], np.int32),
        "is_dev": np.zeros(n, bool),
        "actions": actions,
        "state": np.zeros((n, 8), np.float32),
    }
    for name, value in arrays.items():
        np.save(tmp_path / f"{name}.npy", value)

    _, all_rows = build_unseen_queries(tmp_path, split="eval")
    _, benchmark_rows = build_unseen_queries(tmp_path, split="eval", official_stride10=True)
    assert len(all_rows["key"]) == 4
    assert len(benchmark_rows["key"]) == 2
    assert [int(x[0]) for x in benchmark_rows["key"]] == [0, 2]
    assert np.allclose(benchmark_rows["previous_velocity"][0], 0)
    # The row at frame 10 has the immediately preceding executed delta from
    # the reconstructed trace, not a difference between consecutive actions.
    expected = sc._physical6(actions[0, 9:10, :6])[0, :3]
    assert np.allclose(benchmark_rows["previous_velocity"][1], expected)
    assert np.all(benchmark_rows["episode"] >= 1_000_000)


def test_selector_uses_memory_store_for_candidates_and_query_store_for_query_tokens(monkeypatch):
    class Store:
        def __init__(self, tokens):
            self.values = tokens
            self.calls = 0

        def tokens(self, keys):
            self.calls += 1
            ids = [int(k[0]) for k in keys]
            x = np.asarray([self.values[i] for i in ids], np.float32)
            return x, np.ones(x.shape[:2], bool)

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.logit_scale = torch.nn.Parameter(torch.tensor(0.))
            self.config = type('Cfg', (), {'continuity_weight': 0.})()

        def forward(self, tokens, mask, ha, hs, state, hmask=None):
            return torch.nn.functional.normalize(tokens[:, 0, :2], dim=-1)

    class Model:
        def __init__(self): self.encoder = Encoder()
        def eval(self): self.encoder.eval(); return self

    memory_store = Store(np.zeros((2, 3, 2), np.float32))
    query_store = Store(np.asarray([[[1., 0.], [0., 1.], [0., 0.]]], np.float32))
    seen = {}
    def encode_items(model, store, bank, inputs, device):
        seen['memory_store'] = store
        return np.asarray([[1., 0.], [0., 1.]], np.float32)
    monkeypatch.setattr(runner, '_encode_items', encode_items)
    value_actions = np.zeros((2, 10, 6), np.float32)
    value_actions[1] = 1.
    bank = {'episode': np.asarray([0, 1]), 'task': np.asarray([0, 1]),
            'value_actions10': value_actions}
    inputs = {'key': np.asarray([[0], [1]])}
    queries = {'key': np.asarray([[0]]), 'history_actions': np.zeros((1, 40, 7), np.float32),
        'history_states': np.zeros((1, 40, 8), np.float32), 'history_mask': np.zeros((1, 40), bool),
        'state': np.zeros((1, 8), np.float32), 'previous_velocity': np.zeros((1, 3), np.float32),
        'episode': np.asarray([10]), 'task': np.asarray([10]), 'actions': np.zeros((1, 10, 7), np.float32)}
    result = runner.evaluate_selector(Model(), memory_store, bank, inputs, queries, device='cpu',
                                      query_store=query_store)
    assert seen['memory_store'] is memory_store
    assert memory_store.calls == 0
    assert query_store.calls == 1
    assert result['top'].tolist() == [0]
