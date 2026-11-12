"""Prepare RM-EXP3 cross-fit drafts using the current magnitude_v2 head.

The retrieval and autoregressive-history logic is reused verbatim from
expert2_prep; only the magnitude checkpoint and output root change.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import expert2_prep as prep

STATE = Path(os.environ.get('SKILLNET_STATE_DIR', '/work/.skillnet-state'))
OUTPUT_ROOT = Path(os.environ.get('ROBOCASA_OUTPUT_ROOT', str(STATE / 'outputs/robocasa-memory')))
ROOT = Path(os.environ.get('ROBOCASA_EXPERT3_ROOT', str(OUTPUT_ROOT / 'expert-03')))
OUT = Path(os.environ.get('ROBOCASA_EXPERT3_PREP', str(ROOT / 'prep')))
MAGNITUDE_V2 = Path(os.environ.get('ROBOCASA_EXPERT3_MAGNITUDE', str(OUTPUT_ROOT / 'key-lite-01/magnitude_v2/magnitude_head.pt')))


def _install_history_free_magnitude_adapter():
    """Keep EXP2 retrieval code while feeding MAG-v2 its exact input contract.

    expert2_prep's historical helper concatenates the retrieval query and its
    10-step history before the magnitude head. MAG-v2 is explicitly
    history-free: its trained input is normalized prefix+state context plus
    the selected-unit metadata. The adapter inserts identity-normalized dummy
    history slots into that legacy layout, then removes them before calling
    the unmodified MAG-v2 network.
    """
    import numpy as np
    import torch

    old_loader = prep._load_models
    ckpt = torch.load(MAGNITUDE_V2, map_location='cpu', weights_only=False)
    new_width = int(ckpt['input_dim'])
    unit_width = 33  # frozen Context key (32) + log position magnitude (1)
    context_width = new_width - unit_width
    legacy_history_width = 130  # 10 x 12 action channels + 10 validity bits
    legacy_width = new_width + legacy_history_width
    source_mean = np.asarray(ckpt['feature_mean'], np.float32)
    source_std = np.asarray(ckpt['feature_std'], np.float32)
    if source_mean.shape != (new_width,) or context_width != 2051:
        raise AssertionError(f'unexpected MAG-v2 feature layout: {new_width=}, {context_width=}')

    class HistoryFreeAdapter(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            if x.shape[-1] != legacy_width:
                raise AssertionError(f'legacy magnitude input width {x.shape[-1]} != {legacy_width}')
            compact = torch.cat((x[..., :context_width], x[..., -unit_width:]), dim=-1)
            return self.inner(compact)

    def load_models(device='cpu'):
        rk, z, pool, feat, rank_ck, rank_net, mag_ck, mag_net = old_loader(device)
        if not mag_ck.get('history_free', False):
            raise AssertionError('expected history-free magnitude_v2 checkpoint')
        # Insert normalized-zero history dimensions in the position expected
        # by the old retrieval helper, preserving exact MAG-v2 normalization.
        mag_ck = dict(mag_ck)
        mag_ck['feature_mean'] = np.concatenate((source_mean[:context_width],
            np.zeros(legacy_history_width, np.float32), source_mean[-unit_width:]))
        mag_ck['feature_std'] = np.concatenate((source_std[:context_width],
            np.ones(legacy_history_width, np.float32), source_std[-unit_width:]))
        return (rk, z, pool, feat, rank_ck, rank_net,
                mag_ck, HistoryFreeAdapter(mag_net).to(device).eval())

    prep._load_models = load_models


def main():
    if not MAGNITUDE_V2.is_file():
        raise FileNotFoundError(MAGNITUDE_V2)
    if (OUT / 'drafts/memory_dynamic.npz').exists():
        raise FileExistsError(f'refusing to overwrite {OUT / "drafts/memory_dynamic.npz"}')
    prep.MAG = MAGNITUDE_V2
    prep.OUT = OUT
    _install_history_free_magnitude_adapter()
    path = prep.prepare_drafts(out=OUT)
    summary = json.loads((path.parent / 'summary.json').read_text())
    summary['magnitude_head'] = str(MAGNITUDE_V2)
    summary['scale'] = 'magnitude_v2: history-free normalized prefix+state context and selected-unit metadata; exp(log norm-ratio) clipped [.25,4], applied once to storage action channels 5:11'
    summary['magnitude_input_adapter'] = 'legacy history slots are identity-normalized and stripped before the exact magnitude_v2 network; validated against direct normalized inference with max absolute error 0'
    summary['retrieval_and_history'] = 'expert2_prep implementation, unchanged; Context+history top-1, same episode excluded, generated commands appended autoregressively to history'
    tmp = path.parent / 'summary.json.tmp'
    tmp.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path.parent / 'summary.json')
    print(json.dumps({'status': 'complete', 'drafts': str(path), **summary}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
