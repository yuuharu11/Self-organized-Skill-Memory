"""Unseen LIBERO task split and image orientation of the offline rows."""
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pytest

from tests.support import (
    MEMORY_PILOT,
    UNSEEN,
    load_script,
)

tasks = load_script(UNSEEN / "tasks.py")
UL = UNSEEN
RUN_CASE = MEMORY_PILOT / "anchor_flow/run_case.py"


# ── Task split ────────────────────────────────────────────────────────────────────────────────────

def test_long_scene_matching_does_not_confuse_scene1_and_scene10():
    assert tasks.scene('KITCHEN_SCENE10_close_it')=='KITCHEN_SCENE10'
    assert 'KITCHEN_SCENE10' not in tasks.LONG_SCENES
    assert tasks.scene('LIVING_ROOM_SCENE1_put_it') in tasks.LONG_SCENES

def test_all_exclusions_apply_independently():
    rows=[{'task_id':0,'name':'KITCHEN_SCENE10_close_it','language':'close it'},
          {'task_id':1,'name':'KITCHEN_SCENE4_novel','language':'novel'},
          {'task_id':2,'name':'KITCHEN_SCENE9_turn_on_the_stove','language':'turn on the stove'},
          {'task_id':3,'name':'STUDY_SCENE3_move_book','language':' PICK_up  the BOOK '},
          {'task_id':4,'name':'STUDY_SCENE2_new','language':'new instruction'}]
    assert [x['task_id'] for x in tasks.eligible(rows,['pick up the book'])]==[0,4]

def test_scene_strata_deterministic_and_input_order_independent():
    rows=[{'task_id':10*s+i,'name':f'KITCHEN_SCENE{s+10}_move_{i}','language':f'move {s} {i}'} for s in range(11) for i in range(3)]
    a=tasks.select_tasks(rows,[]);b=tasks.select_tasks(rows[::-1],[])
    assert a==b and len(a)==10 and len({tasks.scene(x['name']) for x in a})==10
    assert not {tasks.scene(x['name']) for x in a}&tasks.LONG_SCENES

def test_actual_benchmark_has_no_training_or_long_overlap():
    # Require the real submodule: with examples/ on sys.path, bare 'libero'
    # resolves to an empty namespace stub and must not count as available.
    pytest.importorskip('libero.libero.benchmark')
    from libero.libero import benchmark
    rows=[];training=[]
    for name in ['libero_spatial','libero_object','libero_goal','libero_10']:
        suite=benchmark.get_benchmark_dict()[name]();training += [suite.get_task(i).language for i in range(suite.n_tasks)]
    suite=benchmark.get_benchmark_dict()['libero_90']()
    for i in range(suite.n_tasks):
        t=suite.get_task(i);rows.append({'task_id':i,'name':t.name,'language':t.language})
    overlap={x['name'] for x in rows if tasks.normalize_instruction(x['language']) in set(map(tasks.normalize_instruction,training))}
    assert overlap==tasks.EXPLICIT_OVERLAPS
    selected=tasks.select_tasks(rows,training)
    assert len(selected)==10
    assert all(tasks.normalize_instruction(x['language']) not in set(map(tasks.normalize_instruction,training)) for x in selected)
    assert not {tasks.scene(x['name']) for x in selected}&tasks.LONG_SCENES


# ── Image orientation (docs/research/09_unseen_libero/image_orientation_fix.md) — Image orientation of the offline unseen LIBERO-90 rows (see docs/research/09_unseen_libero/image_orientation_fix.md). ───

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod); return mod


P = _load('unseen_paths_test', UL / 'unseen_paths.py')


def closed_loop_transform(frame):
    """The policy-input transform of anchor_flow/run_case.py (and worker.py's recording)."""
    return np.ascontiguousarray(frame[::-1, ::-1])


def test_orient_matches_closed_loop_transform():
    x = np.random.default_rng(0).integers(0, 256, (128, 128, 3), dtype=np.uint8)
    np.testing.assert_array_equal(P.orient_libero_frame(x), closed_loop_transform(x))
    np.testing.assert_array_equal(P.orient_libero_frame(x)[0, 0], x[-1, -1])      # 180 degrees, not a mirror
    assert not np.array_equal(P.orient_libero_frame(x), x)


def test_closed_loop_still_rotates_both_cameras():
    src = RUN_CASE.read_text()
    for cam in ('agentview_image', 'robot0_eye_in_hand_image'):
        assert re.search(r'obs\["%s"\]\[::-1, ::-1\]' % cam, src), cam
    w = (UL / 'worker.py').read_text()
    assert "obs['agentview_image'][::-1,::-1]" in w and "obs['robot0_eye_in_hand_image'][::-1,::-1]" in w


def test_prepare_rotates_both_hdf5_cameras_and_writes_a_new_directory():
    src = (UL / 'prepare_oracle.py').read_text()
    assert "'observation/image':orient_libero_frame(base[start])" in src
    assert "'observation/wrist_image':orient_libero_frame(wrist[start])" in src
    assert 'never overwritten' in src
    assert P.UNSEEN_ROWS != P.LEGACY_UNSEEN_ROWS


def test_no_consumer_reads_the_legacy_rows():
    hits = []
    for f in sorted(UL.glob('*.py')):
        if f.name == 'unseen_paths.py': continue
        code = '\n'.join(l.split('#')[0] for l in f.read_text().splitlines())
        code = re.sub(r'"""[\s\S]*?"""', '', code)
        if re.search(r"""/\s*['"]unseen_rows['"]|['"]oracle-01['"]\s*\)?\s*/\s*['"]unseen_rows""", code): hits.append(f.name)
    assert not hits, f'still joining the legacy oracle-01/unseen_rows path: {hits}'


def _rows_ready(root):
    return (root / 'image_base.npy').exists() and (root / 'manifest.json').exists()


@pytest.mark.skipif(not _rows_ready(P.UNSEEN_ROWS), reason='corrected unseen rows not built')
def test_corrected_rows_equal_closed_loop_transform_of_raw_hdf5():
    h5py = pytest.importorskip('h5py')
    image_tools = pytest.importorskip('openpi.shared.image_tools')
    man = json.loads((P.UNSEEN_ROWS / 'manifest.json').read_text())
    ib = np.load(P.UNSEEN_ROWS / 'image_base.npy', mmap_mode='r'); iw = np.load(P.UNSEEN_ROWS / 'image_wrist.npy', mmap_mode='r')
    frame = np.load(P.UNSEEN_ROWS / 'frame.npy')
    starts = np.r_[0, np.cumsum([r['rows'] for r in man['source_records']])]
    checked = 0
    for k in (0, len(man['source_records']) // 2, len(man['source_records']) - 1):
        rec = man['source_records'][k]
        if not Path(rec['source_hdf5']).exists(): pytest.skip('LIBERO HDF5 not available')
        with h5py.File(rec['source_hdf5'], 'r') as f:
            obs = f['data'][rec['source_demo']]['obs']
            for row in (starts[k], starts[k + 1] - 1):
                t = int(frame[row]) * 2                                   # rows subsample every other HDF5 frame
                for arr, cam in ((ib, 'agentview_rgb'), (iw, 'eye_in_hand_rgb')):
                    raw = np.asarray(obs[cam][t], np.uint8)
                    want = np.asarray(image_tools.resize_with_pad(closed_loop_transform(raw), 224, 224)).astype(int)
                    wrong = np.asarray(image_tools.resize_with_pad(raw, 224, 224)).astype(int)
                    got = np.asarray(arr[row]).astype(int)
                    # the openpi input path (LiberoInputs + ResizeImages) may round differently by one level
                    assert np.abs(got - want).max() <= 1
                    assert np.abs(got - wrong).mean() > 10 * max(np.abs(got - want).mean(), .1)
                    checked += 1
    assert checked == 12


@pytest.mark.skipif(not (_rows_ready(P.UNSEEN_ROWS) and _rows_ready(P.LEGACY_UNSEEN_ROWS)), reason='rows not available')
def test_legacy_rows_are_the_upside_down_copy_and_nothing_else_changed():
    for k in ('image_base', 'image_wrist'):
        new = np.load(P.UNSEEN_ROWS / f'{k}.npy', mmap_mode='r'); old = np.load(P.LEGACY_UNSEEN_ROWS / f'{k}.npy', mmap_mode='r')
        np.testing.assert_array_equal(np.asarray(old[:8])[:, ::-1, ::-1], new[:8])
    for k in ('actions', 'state', 'prompt', 'prompt_mask', 'task', 'episode', 'frame', 'is_dev'):
        np.testing.assert_array_equal(np.load(P.UNSEEN_ROWS / f'{k}.npy'), np.load(P.LEGACY_UNSEEN_ROWS / f'{k}.npy'))
