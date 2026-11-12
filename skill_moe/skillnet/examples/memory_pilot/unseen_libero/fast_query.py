"""F1: the public pi0-FAST LIBERO policy's output as a training-free motion-memory query.

pi0-FAST tokenises an action chunk by a per-axis DCT (low frequencies first) + BPE, and its VLM is trained
end to end to emit those tokens.  Its output is therefore a prediction of the shape of the next motion from
images + language.  Here that prediction is used, without any training, to pick a unit from the S1 pools.

Model / pipeline (pi0_fast_libero is not defined in this fork's training config; it is rebuilt here with the
upstream openpi settings, no openpi file is changed)
  Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180), checkpoint gs://openpi-assets/checkpoints/
  pi0_fast_libero.  Inputs: base + left-wrist image, right-wrist zeros with mask True (FAST convention), state =
  raw 8-d state z-score-normalised (mean/std) with the FAST norm stats, prompt "Task: <instruction>, State: <256-bin
  state>;\\n".  Greedy decoding (temperature 0), at most 256 new tokens.  Output: upstream ExtractFASTActions
  (text round trip, "Action: ... |"), FAST BPE -> integer DCT coefficients [10,7] / scale 10 -> idct, then
  z-score un-normalisation with the FAST stats and the upstream ``extra_delta_transform`` of this checkpoint
  (AbsoluteActions on the first 6 axes: + raw state[:6]).  The raw action is finally normalised with the
  pi05_libero stats so that it lives in the same space as every other prediction / target here.
  Normalisation: current upstream openpi would apply quantile normalisation to a PI0_FAST model, but this
  checkpoint was trained with z-score (mean/std) normalisation of state and actions: on 16 LIBERO-40 rows the
  teacher-forced NLL of the true action tokens is 0.67 nats/token with z-score + delta, against 8.3 with
  quantile + delta (and >= 8.1 for every other combination).  ``fast_query.py`` therefore uses z-score.
  The FAST action tokenizer is read as data (tokenizer.json via PreTrainedTokenizerFast); its decode is
  re-implemented in ``FastActionDecoder`` (the hub code is not executed).

Observations: the unseen rows (``unseen_paths.UNSEEN_ROWS``) and Dev (113) / Eval (619) windows as K1 / S1.
The saved fast-query-01 run read the legacy rows (oracle-01/unseen_rows), whose HDF5-derived images were
rotated 180 degrees relative to the LIBERO training data; there "asis" = rotated and "rot180" = correct.
The rows now default to the corrected build, so on a re-run "asis" is the correct orientation and "rot180"
the upside-down diagnostic (see docs/research/09_unseen_libero/image_orientation_fix.md).

Evaluation re-uses S1 (scale_only_query.py): pools W / U, S-shared type, G1 gripper, six-axis MSE, episode
bootstrap.  Ground-truth actions are read only in ``evaluate`` for scoring and for the labelled oracle rows.
Nothing is selected on Eval (the optional P5 top-k re-rank chooses k on Dev).

Stages:  python fast_query.py {infer,pi05,evaluate,report,all} [--smoke] [--out DIR]
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

import numpy as np

STATE_DIR = Path(os.environ.get('SKILLNET_STATE_DIR', Path(__file__).resolve().parents[5] / '.skillnet-state')).expanduser()
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = STATE_DIR / 'outputs/memory-pilot/unseen-libero'
OUT = ROOT / 'fast-query-01'
S1 = ROOT / 'scale-only-01' / 'production'
from unseen_paths import UNSEEN_ROWS as UNSEEN  # corrected rows (image orientation fix)
K1 = ROOT / 'key-query-01'
CKPTS = STATE_DIR / 'checkpoints/openpi-assets/checkpoints'
FAST_CKPT = Path(os.environ.get('SKILLNET_PI0FAST_CHECKPOINT', CKPTS / 'pi0_fast_libero')).expanduser()
PI05_CKPT = Path(os.environ.get('SKILLNET_PI05_CHECKPOINT', CKPTS / 'pi05_libero')).expanduser()
FAST_TOKENIZER = Path(os.environ.get('SKILLNET_FAST_TOKENIZER', OUT / 'fast_tokenizer')).expanduser()
PI05_RUN = Path(os.environ.get('SKILLNET_PI05_ROOT', STATE_DIR / 'outputs/memory-pilot/pi05-memory40-20261104')).expanduser() / 'run-01'
ACTION_DIM, HORIZON, MAX_TOKEN_LEN, MAX_DECODE = 7, 10, 180, 256
PALIGEMMA_EOS = 1
FAST_SKIP = 128
PI05_NOISE_SEED = 4101
BOOT_SEED = 4101
VIEWS = ('asis', 'rot180')


def _so():
    import scale_only_query as so
    return so


def log(out, **rec):
    rec = {'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **rec}
    print(json.dumps(rec, ensure_ascii=False), flush=True)
    with open(Path(out) / 'progress.log', 'a') as f: f.write(json.dumps(rec, ensure_ascii=False) + '\n')


# ======================================================================== normalisation
def norm_stats(ckpt):
    p = list((Path(ckpt) / 'assets').rglob('norm_stats.json'))
    if len(p) != 1: raise ValueError(f'expected one norm_stats.json under {ckpt}, got {p}')
    s = json.loads(p[0].read_text())['norm_stats']
    return {k: {q: np.asarray(s[k][q], np.float64) for q in ('mean', 'std', 'q01', 'q99')} for k in ('state', 'actions')}


def qnorm(x, st):
    d = np.asarray(x).shape[-1]
    return (np.asarray(x, np.float64) - st['q01'][:d]) / (st['q99'][:d] - st['q01'][:d] + 1e-6) * 2. - 1.


def qunnorm(x, st):
    d = np.asarray(x).shape[-1]
    return (np.asarray(x, np.float64) + 1.) / 2. * (st['q99'][:d] - st['q01'][:d] + 1e-6) + st['q01'][:d]


def znorm(x, st):
    d = np.asarray(x).shape[-1]
    return (np.asarray(x, np.float64) - st['mean'][:d]) / (st['std'][:d] + 1e-6)


def zunnorm(x, st):
    d = np.asarray(x).shape[-1]
    return np.asarray(x, np.float64) * (st['std'][:d] + 1e-6) + st['mean'][:d]


def fast_to_common(a_model, raw_state, fast_ns, pi05_ns):
    """FAST model-space chunk [...,10,7] -> raw action (z-score unnormalise, + raw state on axes 0-5) -> pi05-normalised."""
    raw = zunnorm(a_model, fast_ns['actions'])
    raw[..., :6] += np.asarray(raw_state, np.float64)[..., None, :6]          # AbsoluteActions(make_bool_mask(6, -1))
    return qnorm(raw, pi05_ns['actions']).astype(np.float32), raw.astype(np.float32)


# ======================================================================== tokenisation
class FastActionDecoder:
    """Data-only re-implementation of physical-intelligence/fast UniversalActionProcessor.decode + the
    openpi FASTTokenizer.extract_actions text round trip.  Returns the integer DCT coefficients as well."""

    def __init__(self, path=FAST_TOKENIZER, paligemma=None):
        from transformers import PreTrainedTokenizerFast
        cfg = json.loads((Path(path) / 'processor_config.json').read_text())
        self.scale = float(cfg['scale']); self.min_token = int(cfg['min_token'])
        self.bpe = PreTrainedTokenizerFast.from_pretrained(str(path))
        self.pg = paligemma or _paligemma_sp()

    def to_fast_ids(self, pg_ids):
        return self.pg.vocab_size() - 1 - FAST_SKIP - np.asarray(pg_ids, np.int64)

    def encode(self, chunk):
        """[T,D] model-space chunk -> FAST ids (for tests / round trips)."""
        from scipy.fft import dct
        c = np.around(dct(np.asarray(chunk, np.float64), axis=0, norm='ortho') * self.scale)
        s = ''.join(map(chr, np.maximum(c.flatten() - self.min_token, 0).astype(int)))
        return self.bpe(s)['input_ids']

    def decode_fast(self, fast_ids, horizon=HORIZON, dim=ACTION_DIM):
        from scipy.fft import idct
        txt = self.bpe.decode(list(map(int, fast_ids)))
        coeff = np.array(list(map(ord, txt)), np.int64) + self.min_token
        if coeff.size != horizon * dim: raise ValueError(f'decoded {coeff.size} coefficients, expected {horizon * dim}')
        coeff = coeff.reshape(horizon, dim)
        return coeff, idct(coeff / self.scale, axis=0, norm='ortho')

    def extract(self, out_tokens, horizon=HORIZON, dim=ACTION_DIM):
        """Generated PaliGemma ids (stopped at the first EOS, as a batch-1 upstream call) -> dict."""
        t = [int(x) for x in np.asarray(out_tokens).reshape(-1)]
        n_gen = t.index(PALIGEMMA_EOS) if PALIGEMMA_EOS in t else len(t)
        t = t[:n_gen]
        text = self.pg.decode(t)
        res = {'n_generated': n_gen, 'eos': n_gen < len(out_tokens), 'text': text, 'ok': False, 'reason': '',
               'coeff': np.zeros((horizon, dim), np.int64), 'actions': np.zeros((horizon, dim), np.float64), 'n_action_tokens': 0}
        if 'Action: ' not in text:
            res['reason'] = 'no "Action: " in output'; return res
        body = text.split('Action: ')[1].split('|')[0].strip()
        if '|' not in text.split('Action: ')[1]: res['reason'] = 'no terminating "|"'
        raw = np.array(self.pg.encode(body), np.int64); res['n_action_tokens'] = int(len(raw))
        try:
            res['coeff'], res['actions'] = self.decode_fast(self.to_fast_ids(raw), horizon, dim); res['ok'] = True
        except Exception as e:                                              # upstream: zeros on failure
            res['reason'] = (res['reason'] + '; ' if res['reason'] else '') + f'decode: {e}'
        return res


def _paligemma_sp():
    import sentencepiece, openpi.shared.download as download
    path = download.maybe_download(os.environ.get('SKILLNET_PALIGEMMA_TOKENIZER', 'gs://big_vision/paligemma_tokenizer.model'), gs={'token': 'anon'})
    with path.open('rb') as f: return sentencepiece.SentencePieceProcessor(model_proto=f.read())


def fast_prefix(sp, prompt, state_fast_norm, max_len=MAX_TOKEN_LEN):
    """openpi FASTTokenizer.tokenize(prompt, state, actions=None): tokens, mask, ar_mask, loss_mask."""
    cleaned = prompt.lower().strip().replace('_', ' ')
    disc = np.digitize(np.asarray(state_fast_norm), bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
    tokens = sp.encode(f"Task: {cleaned}, State: {' '.join(map(str, disc))};\n", add_bos=True)
    if len(tokens) > max_len: raise ValueError(f'prefix has {len(tokens)} tokens > {max_len}')
    n = len(tokens); pad = max_len - n
    return (np.asarray(tokens + [0] * pad, np.int32), np.asarray([True] * n + [False] * pad), np.zeros(max_len, np.int32),
            np.zeros(max_len, bool))


# ======================================================================== model
def load_fast(ckpt=FAST_CKPT):
    import jax, flax.nnx as nnx
    from openpi.models.pi0_fast import Pi0FASTConfig
    from openpi.models.model import restore_params
    cfg = Pi0FASTConfig(action_dim=ACTION_DIM, action_horizon=HORIZON, max_token_len=MAX_TOKEN_LEN)
    model = nnx.eval_shape(cfg.create, jax.random.key(0))
    state = nnx.state(model); expected = state.to_pure_dict()
    loaded = restore_params(Path(ckpt) / 'params', restore_type=np.ndarray)
    a, ta = jax.tree.flatten(expected); b, tb = jax.tree.flatten(loaded)
    if ta != tb or any(x.shape != y.shape for x, y in zip(a, b, strict=True)):
        raise ValueError('pi0_fast_libero checkpoint does not match the Pi0FAST model tree')
    state.replace_by_pure_dict(jax.tree.map(jax.numpy.asarray, loaded)); nnx.update(model, state)
    return model, cfg


def source_rows(kind):
    """'unseen' -> oracle-01 unseen rows; 'l40' -> LIBERO-40 rows (sanity check on seen tasks)."""
    if kind == 'unseen':
        d = {k: np.load(UNSEEN / f'{k}.npy', mmap_mode='r', allow_pickle=False) for k in ('image_base', 'image_wrist', 'state', 'task', 'episode', 'frame')}
        d['instruction'] = np.load(UNSEEN / 'instruction.npy'); return d
    d = {k: np.load(PI05_RUN / 'rows' / f'{k}.npy', mmap_mode='r', allow_pickle=False)
         for k in ('image_base', 'image_wrist', 'state', 'task', 'episode', 'frame', 'validation', 'prompt', 'prompt_mask')}
    tasks = {int(k): v for k, v in json.loads((PI05_RUN / 'data_manifest.json').read_text())['tasks'].items()}
    d['instruction'] = np.asarray([tasks[int(t)] for t in d['task']]); return d


def raw_state_of(rows, ids, pi05_ns):
    """Raw 8-d state: the stored rows hold the pi05-quantile-normalised state."""
    return qunnorm(np.asarray(rows['state'][ids, :8], np.float64), pi05_ns['state'])


def images_of(rows, ids, view):
    ib = np.asarray(rows['image_base'][ids]); iw = np.asarray(rows['image_wrist'][ids])
    if view == 'rot180': ib, iw = ib[:, ::-1, ::-1], iw[:, ::-1, ::-1]
    elif view != 'asis': raise ValueError(view)
    return np.ascontiguousarray(ib), np.ascontiguousarray(iw)


def fast_obs(rows, ids, view, sp, fast_ns, pi05_ns):
    import jax.numpy as jnp
    from openpi.models.model import Observation
    ib, iw = images_of(rows, ids, view)
    raw = raw_state_of(rows, ids, pi05_ns); st = znorm(raw, fast_ns['state']).astype(np.float32)
    toks = [fast_prefix(sp, str(rows['instruction'][i]), st[j]) for j, i in enumerate(ids)]
    images = {'base_0_rgb': jnp.asarray(ib, jnp.float32) / 255 * 2 - 1, 'left_wrist_0_rgb': jnp.asarray(iw, jnp.float32) / 255 * 2 - 1}
    images['right_wrist_0_rgb'] = jnp.zeros_like(images['base_0_rgb'])
    obs = Observation(images=images, image_masks={k: jnp.ones((len(ids),), bool) for k in images}, state=jnp.asarray(st),
                      tokenized_prompt=jnp.asarray(np.stack([t[0] for t in toks])), tokenized_prompt_mask=jnp.asarray(np.stack([t[1] for t in toks])),
                      token_ar_mask=jnp.asarray(np.stack([t[2] for t in toks])), token_loss_mask=jnp.asarray(np.stack([t[3] for t in toks])))
    return obs, raw, st, np.stack([t[0] for t in toks])


def infer_fast(out, *, jobs, batch=16):
    """jobs: list of (name, kind, ids, view).  Saves predictions/fast_<name>.npz (resumable per job)."""
    import jax
    from openpi.shared.nnx_utils import module_jit
    out = Path(out); D = out / 'predictions'; D.mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs if not (D / f'fast_{j[0]}.npz').exists()]
    if not todo: return
    fast_ns, pi05_ns = norm_stats(FAST_CKPT), norm_stats(PI05_CKPT)
    sp = _paligemma_sp(); dec = FastActionDecoder(paligemma=sp)
    model, _ = load_fast(); fn = module_jit(model.sample_actions, static_argnames=('max_decoding_steps', 'temperature'))
    for name, kind, ids, view in todo:
        rows = source_rows(kind); ids = np.asarray(ids, np.int64); t0 = time.time()
        acc = {k: [] for k in ('tokens', 'prefix', 'state_raw', 'state_fast', 'coeff', 'model_actions', 'actions', 'raw_actions',
                               'ok', 'n_generated', 'n_action_tokens', 'eos')}
        reasons, texts = [], []
        for st in range(0, len(ids), batch):
            q = ids[st:st + batch]
            obs, raw, sfast, prefix = fast_obs(rows, q, view, sp, fast_ns, pi05_ns)
            toks = np.asarray(fn(jax.random.key(0), obs, max_decoding_steps=MAX_DECODE, temperature=0.0)).astype(np.int64)
            for j in range(len(q)):
                r = dec.extract(toks[j])
                common, rawa = fast_to_common(r['actions'][None], raw[j:j + 1], fast_ns, pi05_ns)
                acc['tokens'].append(toks[j]); acc['prefix'].append(prefix[j]); acc['state_raw'].append(raw[j]); acc['state_fast'].append(sfast[j])
                acc['coeff'].append(r['coeff']); acc['model_actions'].append(r['actions']); acc['actions'].append(common[0]); acc['raw_actions'].append(rawa[0])
                acc['ok'].append(r['ok']); acc['n_generated'].append(r['n_generated']); acc['n_action_tokens'].append(r['n_action_tokens']); acc['eos'].append(r['eos'])
                reasons.append(r['reason']); texts.append(r['text'])
            if (st // batch) % 10 == 0: log(out, stage='infer', job=name, done=min(st + batch, len(ids)), total=len(ids), sec=round(time.time() - t0, 1))
        arr = {k: np.asarray(v) for k, v in acc.items()}
        arr.update(row=ids, reason=np.asarray(reasons, dtype='U200'), text=np.asarray(texts, dtype='U2000'))
        _so()._savez(D / f'fast_{name}.npz', **arr)
        log(out, stage='infer', job=name, rows=len(ids), view=view, decode_failures=int((~arr['ok']).sum()), sec=round(time.time() - t0, 1))


def infer_pi05(out, *, jobs, batch=4):
    """pi0.5 predictions as key_query_probe.extract_policy_feature_stage (K1 policy_features: noise key
    fold_in(key(4101), row id), 10 steps, batch 4), with the same or rotated images."""
    import jax, jax.numpy as jnp
    from openpi.shared.nnx_utils import module_jit
    import key_query_probe as kq
    from openpi.models.model import Observation
    out = Path(out); D = out / 'predictions'; D.mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs if not (D / f'pi05_{j[0]}.npz').exists()]
    if not todo: return
    base = kq._load_policy_run_module()
    model, _ = base.load_policy(PI05_CKPT); fn = module_jit(model.sample_actions, static_argnames=('num_steps',))
    for name, kind, ids, view in todo:
        src = UNSEEN if kind == 'unseen' else PI05_RUN / 'rows'; offset = 0     # K1: noise key fold_in(4101, row id within its source)
        rows = {k: np.load(src / f'{k}.npy', mmap_mode='r', allow_pickle=False) for k in ('image_base', 'image_wrist', 'state', 'prompt', 'prompt_mask')}
        ids = np.asarray(ids, np.int64); res = []; t0 = time.time()
        for st in range(0, len(ids), batch):
            q = ids[st:st + batch]; ib, iw = images_of(rows, q, view)
            images = {'base_0_rgb': jnp.asarray(ib, jnp.float32) / 255 * 2 - 1, 'left_wrist_0_rgb': jnp.asarray(iw, jnp.float32) / 255 * 2 - 1}
            images['right_wrist_0_rgb'] = jnp.zeros_like(images['base_0_rgb'])
            obs = Observation(images=images, image_masks={k: jnp.full((len(q),), k != 'right_wrist_0_rgb') for k in images},
                              state=jnp.asarray(rows['state'][q]), tokenized_prompt=jnp.asarray(rows['prompt'][q]),
                              tokenized_prompt_mask=jnp.asarray(rows['prompt_mask'][q]))
            seeds = jnp.asarray(offset + q, jnp.uint32)
            keys = jax.vmap(lambda r: jax.random.fold_in(jax.random.key(PI05_NOISE_SEED), r))(seeds)
            noise = jax.vmap(lambda k: jax.random.normal(k, (model.action_horizon, model.action_dim)))(keys)
            res.append(np.asarray(fn(jax.random.PRNGKey(0), obs, num_steps=10, noise=noise))[:, :, :7])
        _so()._savez(D / f'pi05_{name}.npz', actions=np.concatenate(res).astype(np.float32), row=ids)
        log(out, stage='pi05', job=name, rows=len(ids), view=view, sec=round(time.time() - t0, 1))


def split_ids():
    """Dev / Eval unseen row ids in the fixed K1 order (asserted against S1 prep)."""
    is_dev = np.load(UNSEEN / 'is_dev.npy')
    dev, ev = np.flatnonzero(is_dev), np.flatnonzero(~is_dev)
    for s, ids in (('dev', dev), ('eval', ev)):
        if not np.array_equal(np.load(S1 / 'prep' / f'W_inputs_{s}.npz')['row'], ids): raise AssertionError(f'{s} rows differ from S1')
    return dev, ev


def l40_sanity_ids(n=96):
    """LIBERO-40 validation rows spread over all 40 tasks (seen-task sanity check of the pipeline)."""
    r = source_rows('l40'); val = np.flatnonzero(np.asarray(r['validation'], bool))
    return val[np.unique(np.linspace(0, len(val) - 1, n).astype(np.int64))]


# ======================================================================== query geometry (no targets)
def phys6(x):
    so = _so(); _, vs, _ = so._mods(); return vs.to_physical(np.asarray(x)[..., :6])


def nn_scale_only(q_phys, pool_phys, allowed):
    """Nearest candidate by the S-shared scale-only error between the query chunk and each candidate.
    Returns (pick, distance matrix [N,K] with disallowed = inf)."""
    so = _so(); d = so.scale_only_error(q_phys, pool_phys, 'S-shared').astype(np.float64)
    d = np.where(allowed, d, np.inf); return d.argmin(1), d


def dct_type(values, phys16=None):
    """S-shared DCT-3 type: 18 physical DCT values divided by the rms speed of their 16-point decoding."""
    so = _so(); v = np.asarray(values, np.float64).reshape(len(values), -1)
    if phys16 is None: phys16 = np.stack([so.dct3_decode(x, so.L16) for x in v])
    return v / np.maximum(so.rms_speed(phys16), 1e-9)[:, None]


def nn_dct(q_values, pool_type, allowed):
    qt = dct_type(q_values)
    d = (qt ** 2).sum(1)[:, None] + (pool_type ** 2).sum(1)[None] - 2 * qt @ pool_type.T
    d = np.where(allowed, np.maximum(d, 0.), np.inf); return d.argmin(1), d


def available_chunk(chunk, n):
    """Physical frames [n,6] of a query chunk stored front-aligned in [10,7]."""
    return phys6(np.asarray(chunk)[:int(n)])


def u_query(chunks, counts):
    """U queries from the frames available at a segment start: 16-point resampling and DCT-3 values."""
    so = _so(); q16, qv = [], []
    for c, n in zip(chunks, counts):
        p = available_chunk(c, n); q16.append(so._resample(p) if len(p) > 1 else np.repeat(p, so.L16, 0)); qv.append(so.dct3_values(p))
    return np.asarray(q16, np.float32), np.asarray(qv)


def p5_scores(pool_name, seed, split, which='holdout'):
    """Full S1 P5 score matrix [N,K] (higher = better) for the S-shared condition (frozen S1 models)."""
    import torch
    so = _so(); P = S1 / 'prep'
    sel = json.loads((S1 / 'selection.json').read_text())['selected'][f'{pool_name}|S-shared|P5|{seed}'][which]
    pool = dict(np.load(P / f'{pool_name}_pool.npz')); d = dict(np.load(P / f'{pool_name}_inputs_{split}.npz'))
    X = torch.as_tensor(so.query_inputs(d, 'P5')); KI = torch.as_tensor(so.key_inputs(pool_name, pool, 'S-shared'))
    r = torch.load(S1 / 'models' / f"{pool_name}_S-shared_P5_s{seed}_f{sel['frac']:.4f}.pt", weights_only=False)
    net = so.make_net(pool_name, 'P5', X.shape[1], KI.shape[1]); net.load_state_dict(r[which]['state']); net.eval()
    with torch.no_grad(): return net(X, KI).numpy().astype(np.float64)


def rerank(scores, dist, allowed, k):
    """Among the k best candidates by `scores`, take the one nearest to the FAST query (smallest `dist`)."""
    s = np.where(allowed, scores, -np.inf); top = np.argsort(-s, 1)[:, :k]
    return top[np.arange(len(top)), np.take_along_axis(dist, top, 1).argmin(1)]


# ======================================================================== metrics
def spearman_rows(a, b, allowed):
    """Mean over rows of Spearman / Pearson between a and b on the allowed entries."""
    from scipy.stats import rankdata
    sp, pe = [], []
    for i in range(len(a)):
        m = allowed[i]; x, y = np.asarray(a[i][m], np.float64), np.asarray(b[i][m], np.float64)
        if x.std() == 0 or y.std() == 0: sp.append(0.); pe.append(0.); continue
        sp.append(np.corrcoef(rankdata(x), rankdata(y))[0, 1]); pe.append(np.corrcoef(x, y)[0, 1])
    return np.asarray(sp), np.asarray(pe)


def cosine_rows(a, b, eps=1e-6):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1); ok = (na > eps) & (nb > eps)
    c = np.full(len(a), np.nan); c[ok] = (a[ok] * b[ok]).sum(-1) / (na[ok] * nb[ok]); return c


def displacement(norm_chunk):
    """Total 3-D position displacement of a normalised chunk [N,T,>=3] (physical units)."""
    return phys6(norm_chunk)[..., :3].sum(-2)


# ======================================================================== jobs
def inference_jobs(smoke=False):
    dev, ev = split_ids(); l40 = l40_sanity_ids(96)
    if smoke: dev, ev, l40 = dev[:8], ev[:16], l40[:8]
    fast = [(f'{s}_{v}', 'unseen', ids, v) for v in VIEWS for s, ids in (('dev', dev), ('eval', ev))] + [('l40_asis', 'l40', l40, 'asis')]
    pi05 = [(f'{s}_{v}', 'unseen', ids, v) for v in VIEWS for s, ids in (('dev', dev), ('eval', ev))] + [('l40_asis', 'l40', l40, 'asis')]
    return fast, pi05



# ======================================================================== evaluation (ground truth read here only)
K_GRID = (1, 2, 3, 5, 10, 20, 50)
W_RULES = ('stored', 'hist', 'fast', 'pi05', 'fastrot', 'pi05rot', 'oracle')
U_RULES = ('stored', 'hist', 'fast', 'pi05', 'oracle')


def load_pred(path, ids):
    z = np.load(path)
    if not np.array_equal(z['row'], ids): raise AssertionError(f'{path} rows misaligned')
    return {k: z[k] for k in z.files}


def _w_generate(so, pool, pick, rule, inp, mag, y):
    if rule == 'oracle': return so.oracle_w(pool, pick, 'S-shared', y)
    if rule in ('stored', 'hist'): return so.generate_w(pool, pick, 'S-shared', rule, inp)
    return so.generate_w(pool, pick, 'S-shared', 'pi05', dict(inp, pi05=mag[rule]))      # match the rms of that chunk


def _u_generate(so, upool, pick, rule, inp, mag, tg):
    if rule == 'oracle': return so.oracle_u(upool, pick, 'S-shared', tg)
    if rule in ('stored', 'hist'): return so.generate_u(upool, pick, 'S-shared', rule, inp)
    now, n = mag[rule]
    return so.generate_u(upool, pick, 'S-shared', 'pi05', dict(inp, pi05_now=now, n_pi05=n))


def evaluate(out, *, smoke=False, seeds=None):
    so = _so(); cd, vs, kq = so._mods(); seeds = seeds or so.SEEDS
    out = Path(out); P = S1 / 'prep'; D = out / 'predictions'; meta = json.loads((P / 'prep.json').read_text())
    thr = meta['gripper_threshold']; gmean = np.asarray(meta['global_mean_action'], np.float32)
    ua = np.load(UNSEEN / 'actions.npy', mmap_mode='r', allow_pickle=False)                      # scoring only
    meta_u = {k: np.asarray(np.load(UNSEEN / f'{k}.npy', mmap_mode='r')) for k in ('task', 'episode', 'frame')}
    ustate = np.load(UNSEEN / 'state.npy', mmap_mode='r')
    pool = dict(np.load(P / 'W_pool.npz')); upool = dict(np.load(P / 'U_pool.npz'))
    u_type = dct_type(upool['values'], upool['phys16'])
    mean_tr = {'W': np.load(P / 'W_err_S-shared_train.npz')['err'].mean(0), 'U': np.load(P / 'U_err_S-shared_train.npz')['err'].mean(0)}
    rng = np.random.default_rng(BOOT_SEED); flat = {}; summary = {'decode': {}, 'checks': {}}
    for split in ('dev', 'eval'):
        inp = dict(np.load(P / f'W_inputs_{split}.npz')); ids = inp['row']
        if smoke: ids = ids[:{'dev': 8, 'eval': 16}[split]]; inp = {k: v[:len(ids)] for k, v in inp.items()}
        n = len(ids)
        F = {v: load_pred(D / f'fast_{split}_{v}.npz', ids) for v in VIEWS}
        Q = {v: load_pred(D / f'pi05_{split}_{v}.npz', ids) for v in VIEWS}
        for v in VIEWS:
            summary['decode'][f'{split}_{v}'] = {'rows': n, 'failures': int((~F[v]['ok']).sum()), 'no_eos': int((~F[v]['eos']).sum()),
                                                 'reasons': sorted(set(map(str, F[v]['reason'][~F[v]['ok']]))),
                                                 'n_action_tokens_median': float(np.median(F[v]['n_action_tokens'])),
                                                 'n_generated_max': int(F[v]['n_generated'].max())}
        rerun = Q['asis']['actions']; off = np.abs(rerun - inp['pi05'])
        summary['checks'][f'pi05_rerun_vs_official_{split}'] = {'max_abs': float(off.max()), 'mean_abs': float(off.mean())}
        chunks = {'fast': F['asis']['actions'], 'pi05': inp['pi05'], 'fastrot': F['rot180']['actions'], 'pi05rot': Q['rot180']['actions']}
        # ------------------------------------------------------------ W
        y = np.asarray(ua[ids, :, :7], np.float32); yphys = phys6(y)
        ep = inp['task'].astype(np.int64) * 100000 + inp['episode'].astype(np.int64)
        hard = inp['hard'].astype(bool) if 'hard' in inp else np.zeros(n, bool)
        A = so.allowed_matrix(inp['task'], pool['task'], pool['keep'], False)
        E = so.scale_only_error(yphys.astype(np.float32), pool['phys'], 'S-shared').astype(np.float64)
        g1 = so.gripper_g1(inp, thr, 'W'); gp = so.grip_part(g1, y)
        R = {}
        keep = np.repeat(inp['hist'][:, -1:, :], 10, 1)
        base = {'keep_prev': keep, 'zero': np.zeros_like(y), 'global_mean': np.broadcast_to(gmean, y.shape), 'pi05': inp['pi05'],
                'pi05_rerun': rerun, 'pi05rot': chunks['pi05rot'], 'fast': chunks['fast'], 'fastrot': chunks['fastrot']}
        for k, pred in base.items():
            R[f'base|{k}'] = so.six_axis(pred, y)
            R[f'base|{k}|full7'] = ((np.asarray(pred, np.float64)[..., :7] - y) ** 2).mean((1, 2))
        R['base|keep_prev|active7'] = R['base|keep_prev'] + so.grip_part(keep[:, 0, 6], y)
        picks, dist, scores = {'oracle_type': np.where(A, E, np.inf).argmin(1)}, {}, {}
        for q in ('fast', 'pi05', 'fastrot', 'pi05rot'):
            picks[f'{q}_nn'], dist[q] = nn_scale_only(phys6(chunks[q]), pool['phys'], A)
        for sd in seeds:
            sc = p5_scores('W', sd, split)[:n]; scores[sd] = sc
            picks[f'P5_s{sd}'] = np.where(A, sc, -np.inf).argmax(1)
            ref = so.score_models(S1, 'W', 'S-shared', 'P5', sd, split)['holdout'][:n]
            if not np.array_equal(ref, picks[f'P5_s{sd}']): raise AssertionError('P5 picks differ from S1 score_models')
            for k in K_GRID: picks[f'P5rr{k}_s{sd}'] = rerank(sc, dist['fast'], A, k)
        rnd = np.stack([rng.choice(np.flatnonzero(A[i]), 5) for i in range(n)])
        for key, pk in list(picks.items()) + [('random', None)]:
            R[f'type|{key}'] = E[np.arange(n), pk] if pk is not None else np.mean([E[np.arange(n), rnd[:, j]] for j in range(5)], 0)
            for rule in W_RULES:
                if pk is None: v = np.mean([so.six_axis(_w_generate(so, pool, rnd[:, j], rule, inp, chunks, y), y) for j in range(5)], 0)
                else: v = so.six_axis(_w_generate(so, pool, pk, rule, inp, chunks, y), y)
                R[f'{key}|{rule}'] = v; R[f'{key}|{rule}|active7'] = v + gp
        R['type|oracle_value'] = np.where(A, E, np.inf).min(1)
        R['type|random_value'] = np.where(A, E, 0).sum(1) / A.sum(1)
        # ranking quality: -(query distance) against -(true type error), allowed candidates only
        for q in ('fast', 'pi05', 'fastrot', 'pi05rot'):
            R[f'rank|{q}|spearman'], R[f'rank|{q}|pearson'] = spearman_rows(-np.where(A, dist[q], 0), -E, A)
        for sd in seeds:
            R[f'rank|P5_s{sd}|spearman'], R[f'rank|P5_s{sd}|pearson'] = spearman_rows(scores[sd], -E, A)
        R['rank|mean_train|spearman'], R['rank|mean_train|pearson'] = spearman_rows(np.broadcast_to(-mean_tr['W'], E.shape), -E, A)
        # direction of the total 3-D displacement
        gt = displacement(y); has = inp['has_hist'].astype(bool)
        for k, pred in (('fast', chunks['fast']), ('pi05', chunks['pi05']), ('keep_prev', keep), ('fastrot', chunks['fastrot']),
                        ('pi05rot', chunks['pi05rot'])):
            R[f'cos|{k}'] = cosine_rows(displacement(pred), gt)
        for key in ('fast_nn', 'pi05_nn', f'P5_s{seeds[0]}', 'oracle_type'):
            R[f'cos|{key}|stored'] = cosine_rows(np.asarray(pool['phys'], np.float64)[picks[key]][:, :, :3].sum(1), gt)
        R['has_hist'] = has.astype(np.float64)
        for k, v in R.items(): flat[f'W|{split}||{k}'] = np.asarray(v, np.float64)
        flat[f'W|{split}|episode'] = ep; flat[f'W|{split}|hard'] = hard
        summary.setdefault('W', {})[split] = {'rows': n, 'hard_rows': int(hard.sum()), 'G1_gripper_part': float(gp.mean()), 'rows_with_history': int(has.sum())}
        log(out, stage='evaluate', pool='W', split=split, rows=n)
        # ------------------------------------------------------------ U
        uin = dict(np.load(P / f'U_inputs_{split}.npz'))
        hard_by_row = {int(r): bool(h) for r, h in zip(ids, hard)} if split == 'eval' else None
        built = {}
        for q in ('fast', 'pi05', 'fastrot', 'pi05rot'):
            si, st_ = so.build_segments(ua, meta_u['task'], meta_u['episode'], meta_u['frame'], ustate, ids, pi05=chunks[q], hard_by_row=hard_by_row)
            built[q] = si; tgt = st_
        si = built['fast']; m = len(si['start'])
        key_of = {(int(t), int(e), int(st)): i for i, (t, e, st) in enumerate(zip(uin['task'], uin['episode'], uin['start']))}
        idx = np.asarray([key_of[(int(t), int(e), int(st))] for t, e, st in zip(si['task'], si['episode'], si['start'])])
        if not np.allclose(built['pi05']['pi05_now'], uin['pi05_now'][idx]) or (not smoke and not np.array_equal(idx, np.arange(len(uin['start'])))):
            raise AssertionError('U segments misaligned with S1')
        uinp = {k: v for k, v in si.items() if not k.startswith('pi05')}; uinp['has_hist'] = uinp['n_past'] > 0
        tg = [so._unflatten(tgt['target_flat'], tgt['target_off'], i) for i in range(m)]
        uep = uinp['task'].astype(np.int64) * 100000 + uinp['episode'].astype(np.int64); uhard = uinp['hard'].astype(bool)
        Au = so.allowed_matrix(uinp['task'], upool['task'], upool['keep'], False)
        Eu = so.scale_only_error(tgt['target16'], upool['phys16'], 'S-shared').astype(np.float64)
        ug1 = so.gripper_g1(uinp, thr, 'U')
        mag = {q: (built[q]['pi05_now'], built[q]['n_pi05']) for q in built}
        RU = {}
        spans = {q: [so._unflatten(built[q]['pi05_span_flat'], built[q]['pi05_span_off'], i) for i in range(m)] for q in built}
        kp = [np.repeat(uinp['last'][i][None], len(tg[i]), 0) for i in range(m)]
        for k, seq in (('keep_prev', kp), ('pi05', spans['pi05']), ('fast', spans['fast']), ('fastrot', spans['fastrot']), ('pi05rot', spans['pi05rot'])):
            RU[f'base|{k}'] = np.array([so.six_axis(a, t) for a, t in zip(seq, tg)])
            RU[f'base|{k}|full7'] = np.array([((a[:, :7] - t[:, :7]) ** 2).mean() for a, t in zip(seq, tg)])
        RU['base|keep_prev|active7'] = RU['base|keep_prev'] + np.array([((uinp['last'][i][6] - t[:, 6]) ** 2).mean() / 7 for i, t in enumerate(tg)])
        upicks, udist, uscores = {'oracle_type': np.where(Au, Eu, np.inf).argmin(1)}, {}, {}
        for q in built:
            q16, qv = u_query(*mag[q])
            upicks[f'{q}_nn'], udist[f'{q}_nn'] = nn_scale_only(q16, upool['phys16'], Au)
            upicks[f'{q}_dct'], udist[f'{q}_dct'] = nn_dct(qv, u_type, Au)
        for sd in seeds:
            sc = p5_scores('U', sd, split)[idx]; uscores[sd] = sc
            upicks[f'P5_s{sd}'] = np.where(Au, sc, -np.inf).argmax(1)
            if not np.array_equal(so.score_models(S1, 'U', 'S-shared', 'P5', sd, split)['holdout'][idx], upicks[f'P5_s{sd}']):
                raise AssertionError('U P5 picks differ from S1')
            for k in K_GRID: upicks[f'P5rr{k}_s{sd}'] = rerank(sc, udist['fast_nn'], Au, k)
        urnd = np.stack([rng.choice(np.flatnonzero(Au[i]), 5) for i in range(m)])
        for key, pk in list(upicks.items()) + [('random', None)]:
            RU[f'type|{key}'] = Eu[np.arange(m), pk] if pk is not None else np.mean([Eu[np.arange(m), urnd[:, j]] for j in range(5)], 0)
            for rule in U_RULES:
                draws = [urnd[:, j] for j in range(5)] if pk is None else [pk]; acc = None
                for d in draws:
                    e = so.u_errors(_u_generate(so, upool, d, rule, uinp, mag, tg), tg, ug1)
                    e['keep_overlap'] = so.u_overlap_baseline(kp, tg, e['n_overlap'])
                    acc = e if acc is None else {k2: acc[k2] + e[k2] for k2 in acc}
                e = {k2: v / len(draws) for k2, v in acc.items()}
                RU[f'{key}|{rule}'] = e['full']; RU[f'{key}|{rule}|active7'] = e['full'] + e['grip_full']
                RU[f'{key}|{rule}|overlap'] = e['overlap']; RU[f'{key}|{rule}|overlap_keep'] = e['keep_overlap']
        RU['type|oracle_value'] = np.where(Au, Eu, np.inf).min(1); RU['type|random_value'] = np.where(Au, Eu, 0).sum(1) / Au.sum(1)
        for q in built:
            for kind in ('nn', 'dct'):
                RU[f'rank|{q}_{kind}|spearman'], RU[f'rank|{q}_{kind}|pearson'] = spearman_rows(-np.where(Au, udist[f'{q}_{kind}'], 0), -Eu, Au)
        for sd in seeds:
            RU[f'rank|P5_s{sd}|spearman'], RU[f'rank|P5_s{sd}|pearson'] = spearman_rows(uscores[sd], -Eu, Au)
        RU['rank|mean_train|spearman'], RU['rank|mean_train|pearson'] = spearman_rows(np.broadcast_to(-mean_tr['U'], Eu.shape), -Eu, Au)
        for k, v in RU.items(): flat[f'U|{split}||{k}'] = np.asarray(v, np.float64)
        flat[f'U|{split}|episode'] = uep; flat[f'U|{split}|hard'] = uhard
        summary.setdefault('U', {})[split] = {'rows': m, 'hard_rows': int(uhard.sum()), 'segment_length_median': float(np.median([len(t) for t in tg]))}
        log(out, stage='evaluate', pool='U', split=split, rows=m)
    # LIBERO-40 seen-task sanity check of the FAST pipeline
    l40 = load_pred(D / 'fast_l40_asis.npz', np.load(D / 'fast_l40_asis.npz')['row']); pl = load_pred(D / 'pi05_l40_asis.npz', l40['row'])
    yl = np.asarray(np.load(PI05_RUN / 'rows' / 'actions.npy', mmap_mode='r')[l40['row'], :, :7], np.float32)
    summary['checks']['libero40_seen_rows'] = {'rows': int(len(yl)), 'fast_six': float(so.six_axis(l40['actions'], yl).mean()),
                                               'pi05_six': float(so.six_axis(pl['actions'], yl).mean()), 'zero_six': float(so.six_axis(np.zeros_like(yl), yl).mean()),
                                               'fast_decode_failures': int((~l40['ok']).sum())}
    E_dir = out / 'evaluation'; E_dir.mkdir(parents=True, exist_ok=True)
    so._savez(E_dir / 'rows.npz', **flat)
    summary.update(seeds=list(seeds), k_grid=list(K_GRID), eval_targets_used_only_for_scoring=True, gripper_threshold=thr)
    so._atomic_json(E_dir / 'summary.json', summary)



# ======================================================================== aggregate / report
REF_SPEARMAN = {'S1 P5（履歴＋状態）': .658, '直前の実行行動だけ': .651, '状態だけ': .339, 'VLM 平均だけ': .040, '学習誤差の平均順（クエリなし）': .096}
RULE_JA = {'stored': '保存のまま', 'hist': '履歴の速さ', 'fast': 'FAST の大きさ', 'pi05': 'π0.5 の大きさ', 'oracle': 'oracle s*',
           'fastrot': 'FAST(rot180) の大きさ', 'pi05rot': 'π0.5(rot180) の大きさ'}


def aggregate(out, *, seeds=None):
    so = _so(); seeds = seeds or so.SEEDS
    out = Path(out); z = np.load(out / 'evaluation' / 'rows.npz'); summ = json.loads((out / 'evaluation' / 'summary.json').read_text())
    res = {'tables': {}, 'ci': {}, 'k': {}, 'summary': summ}
    for pool_name in ('W', 'U'):
        # ---- choose k for the P5 top-k re-rank on Dev (seed-mean type error of the pick); Eval never used
        dev = lambda key: z[f'{pool_name}|dev||{key}']
        curve = {k: float(np.mean([dev(f'type|P5rr{k}_s{sd}').mean() for sd in seeds])) for k in K_GRID}
        kstar = min(K_GRID, key=lambda k: (round(curve[k], 12), k)); res['k'][pool_name] = {'dev_type_error': curve, 'chosen': kstar}
        for split in ('eval', 'dev'):
            tag = f'{pool_name}|{split}'; g = lambda key: z[f'{tag}||{key}']
            hard = z[f'{tag}|hard'].astype(bool); ep = z[f'{tag}|episode']

            def val(key):
                if key.startswith('P5|') or key.startswith('P5rr|'):
                    head, rest = key.split('|', 1); name = 'P5' if head == 'P5' else f'P5rr{kstar}'
                    return np.mean([g(f'{name}_s{sd}|{rest}') for sd in seeds], 0)
                if key.startswith('type|P5'):
                    name = 'P5' if key == 'type|P5' else f'P5rr{kstar}'
                    return np.mean([g(f'type|{name}_s{sd}') for sd in seeds], 0)
                if key.startswith('rank|P5|'):
                    return np.mean([g(f'rank|P5_s{sd}|{key.split("|")[2]}') for sd in seeds], 0)
                return g(key)

            def st(v, nanok=False):
                f = np.nanmean if nanok else np.mean
                return {'all': float(f(v)), 'hard': float(f(v[hard])) if hard.any() else None}
            T = {}
            pickers = ['fast_nn', 'pi05_nn', 'fastrot_nn', 'pi05rot_nn', 'P5', 'P5rr', 'random', 'oracle_type']
            if pool_name == 'U': pickers[2:2] = ['fast_dct', 'pi05_dct']; pickers += ['fastrot_dct', 'pi05rot_dct']
            rules = W_RULES if pool_name == 'W' else U_RULES
            for pk in pickers:
                for r in rules:
                    for sfx in ('', '|active7') + (('|overlap', '|overlap_keep') if pool_name == 'U' else ()):
                        T[f'{pk}|{r}{sfx}'] = st(val(f'{pk}|{r}{sfx}'))
                tv = val(f'type|{pk}'); orc, rnd = g('type|oracle_value'), g('type|random_value')
                T[f'type|{pk}'] = {'type_error': float(tv.mean()), 'gap_recovery': float((rnd - tv).mean() / max((rnd - orc).mean(), 1e-12)),
                                   'strict_top1': float((tv <= orc + .1 * (rnd - orc)).mean())}
            for b in ('keep_prev', 'zero', 'global_mean', 'pi05', 'pi05_rerun', 'pi05rot', 'fast', 'fastrot'):
                if f'{tag}||base|{b}' in z: T[f'base|{b}'] = st(g(f'base|{b}')); T[f'base|{b}|full7'] = st(g(f'base|{b}|full7'))
            T['base|keep_prev|active7'] = st(g('base|keep_prev|active7'))
            rank_keys = ['fast', 'pi05', 'fastrot', 'pi05rot'] if pool_name == 'W' else ['fast_nn', 'fast_dct', 'pi05_nn', 'pi05_dct', 'fastrot_nn', 'fastrot_dct', 'pi05rot_nn', 'pi05rot_dct']
            for rk in rank_keys + ['P5', 'mean_train']:
                for m in ('spearman', 'pearson'): T[f'rank|{rk}|{m}'] = st(val(f'rank|{rk}|{m}'))
            if pool_name == 'W':
                has = g('has_hist').astype(bool)
                for k in ('fast', 'pi05', 'keep_prev', 'fastrot', 'pi05rot', 'fast_nn|stored', 'pi05_nn|stored', f'P5_s{seeds[0]}|stored', 'oracle_type|stored'):
                    c = g(f'cos|{k}'); ok = ~np.isnan(c)
                    T[f'cos|{k}'] = {'all': float(np.nanmean(c)), 'hard': float(np.nanmean(c[hard])) if hard.any() else None,
                                     'with_history': float(np.nanmean(c[has])), 'positive': float((c[ok] > 0).mean()), 'valid': int(ok.sum()),
                                     'positive_hard': float((c[ok & hard] > 0).mean()) if (ok & hard).any() else None}
            res['tables'][tag] = T
            if split != 'eval': continue
            C = {}

            def ci(a, b):
                return {'all': so.boot_diff(a, b, ep), 'hard': so.boot_diff(a[hard], b[hard], ep[hard]) if hard.any() else None}
            keep = g('base|keep_prev')
            nn_keys = ['fast_nn'] + (['fast_dct'] if pool_name == 'U' else [])
            for r in ('stored', 'hist', 'fast', 'pi05'):
                p5 = val(f'P5|{r}')
                for nk in nn_keys:
                    a = val(f'{nk}|{r}')
                    C[f'{nk}|{r}-keep_prev'] = ci(a, keep); C[f'{nk}|{r}-P5'] = ci(a, p5)
                C[f'pi05_nn|{r}-keep_prev'] = ci(val(f'pi05_nn|{r}'), keep)
                C[f'P5rr|{r}-P5'] = ci(val(f'P5rr|{r}'), p5); C[f'P5rr|{r}-keep_prev'] = ci(val(f'P5rr|{r}'), keep)
                C[f'P5|{r}-keep_prev'] = ci(p5, keep)
            C['fast-pi05'] = ci(g('base|fast'), g('base|pi05')); C['fast-keep_prev'] = ci(g('base|fast'), keep)
            C['fastrot-pi05rot'] = ci(g('base|fastrot'), g('base|pi05rot')); C['fastrot-keep_prev'] = ci(g('base|fastrot'), keep)
            C['pi05-keep_prev'] = ci(g('base|pi05'), keep); C['pi05rot-keep_prev'] = ci(g('base|pi05rot'), keep)
            for r in ('stored', 'hist'):
                for q in ('fastrot_nn', 'pi05rot_nn'):
                    C[f'{q}|{r}-keep_prev'] = ci(val(f'{q}|{r}'), keep); C[f'{q}|{r}-P5'] = ci(val(f'{q}|{r}'), val(f'P5|{r}'))
            if pool_name == 'W':
                C['spearman|fast-P5'] = ci(g('rank|fast|spearman'), val('rank|P5|spearman'))
            else:
                C['spearman|fast_nn-P5'] = ci(g('rank|fast_nn|spearman'), val('rank|P5|spearman'))
            res['ci'][tag] = C
    so._atomic_json(out / 'results.json', res)
    return res


def _f(x, d=3):
    if x is None: return '—'
    return f'{x:.{d}f}'.replace('0.', '.', 1) if abs(x) < 1 else f'{x:.{d}f}'


def _c(s):
    return '—' if s is None else f"{_f(s['all'])} / {_f(s['hard'])}"


def _ci(c):
    if c is None: return '—'
    return f"{c['mean']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]"


def report(out, *, smoke=False, seeds=None):
    out = Path(out); res = aggregate(out, seeds=seeds); summ = res['summary']
    PK_JA = {'fast_nn': 'FAST 最近傍（型）', 'fast_dct': 'FAST 最近傍（DCT-3）', 'pi05_nn': 'π0.5 最近傍（型）', 'pi05_dct': 'π0.5 最近傍（DCT-3）',
             'fastrot_nn': 'FAST(rot180) 最近傍（型）', 'fastrot_dct': 'FAST(rot180) 最近傍（DCT-3）', 'pi05rot_nn': 'π0.5(rot180) 最近傍（型）',
             'pi05rot_dct': 'π0.5(rot180) 最近傍（DCT-3）', 'P5': 'S1 P5（学習クエリ、3 seed 平均）', 'P5rr': 'P5 top-k を FAST で並べ替え',
             'random': 'random 型（5回平均）', 'oracle_type': 'oracle 型'}
    L = ['# F1：π0-FAST の出力を運動メモリのクエリに使う' + ('（smoke）' if smoke else ''), '',
         '各セルは「全体 / 難しい行」の 6 軸 MSE（正規化行動、S1 と同じ定義）。型は S-shared（6 軸で 1 つの大きさ、向きは残す）。'
         'Eval は選択に使わない（P5 の top-k の k は Dev で選択）。', '']
    for pool_name in ('W', 'U'):
        T = res['tables'][f'{pool_name}|eval']; C = res['ci'][f'{pool_name}|eval']; k = res['k'][pool_name]['chosen']
        rules = ['stored', 'hist', 'fast', 'pi05', 'oracle']
        L += [f"## {pool_name}：6 軸 MSE（Eval {summ[pool_name]['eval']['rows']} 行、難しい行 {summ[pool_name]['eval']['hard_rows']}）", '',
              '| 型の選び方 | ' + ' | '.join(RULE_JA[r] for r in rules) + ' |', '|---|' + '---:|' * len(rules)]
        pk_main = ['fast_nn'] + (['fast_dct'] if pool_name == 'U' else []) + ['pi05_nn'] + (['pi05_dct'] if pool_name == 'U' else []) + ['P5', 'P5rr', 'random', 'oracle_type']
        for pk in pk_main:
            lab = PK_JA[pk] + (f'（k={k}）' if pk == 'P5rr' else '')
            L.append(f'| {lab} | ' + ' | '.join(_c(T[f'{pk}|{r}']) for r in rules) + ' |')
        L += ['', '| 基準 | 6 軸 MSE | 7 軸（グリッパー込み） |', '|---|---:|---:|']
        for b, lab in (('keep_prev', '直前の行動を保つ'), ('zero', 'ゼロ'), ('global_mean', '全体平均'), ('pi05', 'π0.5（保存済みの予測）'),
                       ('pi05_rerun', 'π0.5（この実行で再計算）'), ('fast', 'π0-FAST をそのまま')):
            if f'base|{b}' in T: L.append(f"| {lab} | {_c(T[f'base|{b}'])} | {_c(T.get(f'base|{b}|full7'))} |")
        L += ['', f'### {pool_name}：差の 95% 区間（episode bootstrap、全体 / 難しい行）', '', '| 比較 | 大きさ | 全体 | 難しい行 |', '|---|---|---:|---:|']
        for key, c in C.items():
            if '|' in key and not key.startswith('spearman'):
                a, r = key.split('|', 1); r, cmp_ = r.split('-', 1)
                L.append(f"| {PK_JA.get(a, a)} − {'直前保持' if cmp_ == 'keep_prev' else 'P5'} | {RULE_JA[r]} | {_ci(c['all'])} | {_ci(c['hard'])} |")
            else:
                L.append(f"| {key} | — | {_ci(c['all'])} | {_ci(c['hard'])} |")
        L += ['', f'### {pool_name}：G1 込み active7（副指標）', '', '| 型の選び方 | ' + ' | '.join(RULE_JA[r] for r in rules) + ' |', '|---|' + '---:|' * len(rules)]
        for pk in pk_main: L.append(f'| {PK_JA[pk]} | ' + ' | '.join(_c(T[f'{pk}|{r}|active7']) for r in rules) + ' |')
        L.append(f"| 直前保持（active7） | {_c(T['base|keep_prev|active7'])} | | | | |")
        if pool_name == 'U':
            L += ['', '### U：重なったフレームだけで採点（直前保持も同じフレーム）', '', '| 型の選び方 | 大きさ | 方法 | 直前保持 |', '|---|---|---:|---:|']
            for pk in ('fast_nn', 'fast_dct', 'P5'):
                for r in ('stored', 'fast', 'pi05'):
                    L.append(f"| {PK_JA[pk]} | {RULE_JA[r]} | {_c(T[f'{pk}|{r}|overlap'])} | {_c(T[f'{pk}|{r}|overlap_keep'])} |")
        L += ['', f'### {pool_name}：候補の並びの良さ（クエリごとの順位相関の平均、全体 / 難しい行）', '', '| クエリ | Spearman | Pearson |', '|---|---:|---:|']
        rk = ['fast', 'pi05', 'fastrot', 'pi05rot'] if pool_name == 'W' else ['fast_nn', 'fast_dct', 'pi05_nn', 'pi05_dct', 'fastrot_nn', 'pi05rot_nn']
        for q in rk + ['P5', 'mean_train']:
            L.append(f"| {q} | {_c(T[f'rank|{q}|spearman'])} | {_c(T[f'rank|{q}|pearson'])} |")
        L += ['', f'### {pool_name}：型選択の参考（S-shared の型誤差）', '', '| 型の選び方 | 型誤差 | 差の回収率 | 厳しい top1 |', '|---|---:|---:|---:|']
        for pk in pk_main + ['fastrot_nn', 'pi05rot_nn']:
            t = T[f'type|{pk}']; L.append(f"| {PK_JA[pk]} | {t['type_error']:.4f} | {100 * t['gap_recovery']:.0f}% | {100 * t['strict_top1']:.0f}% |")
        L += ['', f"P5 の top-k の k（Dev の型誤差で選択）：{k}。Dev の曲線：" + ', '.join(f'k={a}: {b:.4f}' for a, b in res['k'][pool_name]['dev_type_error'].items())]
        if pool_name == 'W':
            L += ['', '### W：合計 3 次元変位の向き（正解とのコサイン）', '', '| 予測 | 平均（全体） | 平均（難しい行） | 履歴がある行 | cos>0 の割合（全体 / 難しい行） | 有効行 |', '|---|---:|---:|---:|---:|---:|']
            for kk, lab in (('fast', 'π0-FAST'), ('pi05', 'π0.5'), ('keep_prev', '直前の行動を保つ'), ('fast_nn|stored', 'FAST 最近傍の単位（保存のまま）'),
                            ('pi05_nn|stored', 'π0.5 最近傍の単位'), (f'P5_s{_so().SEEDS[0]}|stored', 'P5 の単位（seed 4101）'), ('oracle_type|stored', 'oracle 型の単位'),
                            ('fastrot', 'π0-FAST（rot180）'), ('pi05rot', 'π0.5（rot180）')):
                c = T[f'cos|{kk}']
                L.append(f"| {lab} | {_f(c['all'])} | {_f(c['hard'])} | {_f(c['with_history'])} | {100 * c['positive']:.0f}% / {100 * c['positive_hard']:.0f}% | {c['valid']} |")
        L += ['', f'### {pool_name}：画像を 180 度戻した診断（rot180）', '', '| 予測 | 6 軸 MSE |', '|---|---:|']
        L.append(f"| π0-FAST（rot180）をそのまま | {_c(T['base|fastrot'])} |"); L.append(f"| π0.5（rot180）をそのまま | {_c(T['base|pi05rot'])} |")
        for q, lab in (('fastrot_nn', 'FAST(rot180)'), ('pi05rot_nn', 'π0.5(rot180)')):
            for r in ('stored', 'hist') + (('fastrot', 'pi05rot') if pool_name == 'W' else ()):
                L.append(f"| {lab} 最近傍の型、{RULE_JA[r]} | {_c(T[f'{q}|{r}'])} |")
        L.append(f"| 直前の行動を保つ（参考） | {_c(T['base|keep_prev'])} |")
        L += ['', f"FAST(rot180) − π0.5(rot180)：{_ci(C['fastrot-pi05rot']['all'])}、FAST(rot180) − 直前保持：{_ci(C['fastrot-keep_prev']['all'])}", '']
    L += ['## デコードと確認', '']
    for k, v in summ['decode'].items(): L.append(f"- {k}：{v['rows']} 行中 失敗 {v['failures']}、EOS なし {v['no_eos']}、行動トークン数の中央値 {v['n_action_tokens_median']:.0f}、理由 {v['reasons']}")
    for k, v in summ['checks'].items(): L.append(f'- {k}：{json.dumps(v, ensure_ascii=False)}')
    L.append('- 参照：' + '、'.join(f'{a} {b:.3f}' for a, b in REF_SPEARMAN.items()) + '（S1 の W、Spearman）')
    (out / 'report.md').write_text('\n'.join(L) + '\n')
    return res


# ======================================================================== entry
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=('infer', 'pi05', 'evaluate', 'report', 'all'))
    ap.add_argument('--smoke', action='store_true'); ap.add_argument('--out', default=None)
    ap.add_argument('--batch', type=int, default=16)
    a = ap.parse_args()
    out = Path(a.out) if a.out else OUT / ('smoke' if a.smoke else 'production'); out.mkdir(parents=True, exist_ok=True)
    fast_jobs, pi05_jobs = inference_jobs(a.smoke)
    settings = {'checkpoint': str(FAST_CKPT), 'config': {'action_dim': ACTION_DIM, 'action_horizon': HORIZON, 'max_token_len': MAX_TOKEN_LEN},
                'decoding': {'temperature': 0.0, 'max_decoding_steps': MAX_DECODE, 'batch': a.batch, 'stop': 'first EOS per row'},
                'normalisation': 'z-score (mean/std) of the pi0_fast_libero norm stats; extra delta on axes 0-5',
                'output_space': 'pi05_libero quantile-normalised actions (same as targets / pi0.5 predictions)',
                'fast_tokenizer': str(FAST_TOKENIZER), 'pi05': {'checkpoint': str(PI05_CKPT), 'noise_seed': PI05_NOISE_SEED, 'num_steps': 10}}
    (out / 'settings.json').write_text(json.dumps(settings, indent=1))
    if a.stage in ('infer', 'all'): infer_fast(out, jobs=fast_jobs, batch=a.batch)
    if a.stage in ('pi05', 'all'): infer_pi05(out, jobs=pi05_jobs)
    if a.stage in ('evaluate', 'all'): evaluate(out, smoke=a.smoke)
    if a.stage in ('report', 'all'): report(out, smoke=a.smoke); log(out, stage='report', done=True)


if __name__ == '__main__':
    main()
