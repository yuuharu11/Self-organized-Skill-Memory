#!/usr/bin/env python3
"""Cut one LIBERO demo into variable-length motion segments and render a video.

Rule, applied from the start of the demo: always cut at a gripper switch; otherwise
extend the segment while its low-frequency DCT reconstruction stays within epsilon.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import dct, idct

Q01 = np.array([-.747375, -.796125, -.9375, -.115803, -.169430, -.194502, -1.], np.float32)
Q99 = np.array([.937125, .8595, .937125, .140226, .181035, .311546, .9996], np.float32)
COLORS = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#6250d6')
AXES = ('x', 'y', 'z')
FRAME_STRIDE = 2  # the demos are 20 Hz; the rows used elsewhere are 10 Hz
MIN_LENGTH = 5


def normalize(a):
    return (np.asarray(a, np.float32) - Q01) / (Q99 - Q01 + 1e-6) * 2 - 1


def low_freq(x, k):
    """Reconstruct a [L,D] segment from its k lowest temporal DCT coefficients."""
    c = dct(np.asarray(x, np.float64), type=2, norm='ortho', axis=0)
    c[min(k, len(x)):] = 0.
    return idct(c, type=2, norm='ortho', axis=0)


def cut(actions, *, epsilon, k=3, min_length=MIN_LENGTH):
    """Return [start,end) segments. Motion error excludes the gripper, which forces a cut."""
    motion, grip = actions[:, :6], np.sign(actions[:, 6])
    switches = set((np.flatnonzero(grip[1:] != grip[:-1]) + 1).tolist())
    segments, start = [], 0
    n = len(actions)
    while start < n:
        forced = [c for c in switches if start < c < start + min_length]
        end = min(forced) if forced else min(start + min_length, n)
        reason = 'gripper' if forced else 'end'
        while not forced and end < n:
            if end in switches:
                reason = 'gripper'
                break
            err = np.mean((low_freq(motion[start:end + 1], k) - motion[start:end + 1]) ** 2)
            if err > epsilon:
                reason = 'error'
                break
            end += 1
        segments.append({'start': start, 'end': end, 'length': end - start, 'reason': reason,
                         'error': float(np.mean((low_freq(motion[start:end], k) - motion[start:end]) ** 2))})
        start = end
    # A stub at the very end carries no motion; give it to the previous segment.
    if len(segments) > 1 and segments[-1]['length'] < min_length and segments[-2]['reason'] != 'gripper':
        stub = segments.pop()
        segments[-1].update(end=stub['end'], length=stub['end'] - segments[-1]['start'], reason='end')
    return segments


def reconstruct(actions, segments, k=3):
    out = np.empty_like(actions[:, :6], dtype=np.float64)
    for s in segments:
        out[s['start']:s['end']] = low_freq(actions[s['start']:s['end'], :6], k)
    return out


def draw_track(ax, segments, n, *, cursor=None):
    for i, s in enumerate(segments):
        ax.barh(0, s['length'], left=s['start'], color=COLORS[i % len(COLORS)], height=.8, edgecolor='#fcfcfb', linewidth=1.5)
    if cursor is not None:
        ax.axvline(cursor, color='#0b0b0b', linewidth=2)
    ax.set_xlim(0, n); ax.set_yticks([]); ax.set_xticks([])
    for side in ax.spines.values():
        side.set_visible(False)


def render(path, images, wrist, actions, segments, title, fps=10):
    n, rec = len(actions), reconstruct(actions, segments)
    writer = imageio.get_writer(path, fps=fps, codec='libx264', quality=7, macro_block_size=None, pixelformat='yuv420p')
    fig = plt.figure(figsize=(10, 6), dpi=100, facecolor='#fcfcfb')
    grid = fig.add_gridspec(3, 2, height_ratios=[3.4, .35, 2.2], hspace=.28, wspace=.05, left=.07, right=.98, top=.92, bottom=.07)
    for t in range(n):
        fig.clf()
        fig.suptitle(title, fontsize=11, color='#0b0b0b')
        current = next(i for i, s in enumerate(segments) if s['start'] <= t < s['end'])
        for j, (frames, name) in enumerate(((images, 'front'), (wrist, 'wrist'))):
            ax = fig.add_subplot(grid[0, j]); ax.imshow(frames[t][::-1]); ax.axis('off')
            ax.set_title(name, fontsize=9, color='#52514e')
        ax = fig.add_subplot(grid[1, :]); draw_track(ax, segments, n, cursor=t)
        s = segments[current]
        ax.set_title(f"segment {current + 1}/{len(segments)}  frames {s['start']}-{s['end']}  cut by: {s['reason']}", fontsize=9, color='#52514e')
        ax = fig.add_subplot(grid[2, :])
        for d, name in enumerate(AXES):
            ax.plot(actions[:, d], color=COLORS[d], linewidth=2, label=f'{name} demo')
            ax.plot(rec[:, d], color=COLORS[d], linewidth=1.5, linestyle='--', label=f'{name} DCT 3')
        ax.plot(actions[:, 6], color='#52514e', linewidth=1, label='gripper')
        for b in segments[1:]:
            ax.axvline(b['start'], color='#d8d7d1', linewidth=1)
        ax.axvline(t, color='#0b0b0b', linewidth=2)
        ax.set_xlim(0, n); ax.set_ylabel('normalized action', fontsize=8); ax.set_xlabel('frame (10 Hz)', fontsize=8)
        ax.tick_params(labelsize=7); ax.legend(ncol=7, fontsize=7, loc='upper right', frameon=False)
        fig.canvas.draw()
        writer.append_data(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])
    writer.close()
    plt.close(fig)


def contact_sheet(path, images, actions, table, title):
    n = len(actions)
    fig, axes = plt.subplots(len(table) + 1, 1, figsize=(10, 1 + .55 * len(table) + 2), height_ratios=[.5] * len(table) + [3], facecolor='#fcfcfb')
    for ax, (eps, segments) in zip(axes, table.items()):
        draw_track(ax, segments, n)
        ax.set_ylabel(f'eps {eps}', rotation=0, ha='right', va='center', fontsize=9, color='#52514e')
        ax.set_title(f'{len(segments)} segments, median length {int(np.median([s["length"] for s in segments]))}', fontsize=8, loc='right', color='#52514e')
    ax = axes[-1]
    for d, name in enumerate(AXES):
        ax.plot(actions[:, d], color=COLORS[d], linewidth=2, label=name)
    ax.plot(actions[:, 6], color='#52514e', linewidth=1, label='gripper')
    ax.set_xlim(0, n); ax.set_xlabel('frame (10 Hz)', fontsize=8); ax.set_ylabel('normalized action', fontsize=8)
    ax.tick_params(labelsize=7); ax.legend(ncol=4, fontsize=7, frameon=False)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, default=Path('/work/data/LIBERO'))
    ap.add_argument('--output', type=Path, default=Path('/work/docs/research/09_unseen_libero/segmentation_demo'))
    ap.add_argument('--epsilons', type=float, nargs='+', default=[.005, .01, .02, .04])
    ap.add_argument('--video-epsilon', type=float, default=.02)
    args = ap.parse_args()
    demos = {'long_moka_pot': ('libero_10', 'KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5'),
             'unseen_bowl_on_cabinet': ('libero_90', 'KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet_demo.hdf5'),
             'unseen_red_mug': ('libero_90', 'STUDY_SCENE3_pick_up_the_red_mug_and_place_it_to_the_right_of_the_caddy_demo.hdf5')}
    summary = {}
    for name, (suite, fname) in demos.items():
        with h5py.File(args.data / suite / fname) as h:
            g = h['data']['demo_0']
            actions = normalize(g['actions'][::FRAME_STRIDE])
            images, wrist = g['obs']['agentview_rgb'][::FRAME_STRIDE], g['obs']['eye_in_hand_rgb'][::FRAME_STRIDE]
        table = {e: cut(actions, epsilon=e) for e in args.epsilons}
        title = fname.replace('_demo.hdf5', '').replace('_', ' ')
        contact_sheet(args.output / f'{name}.png', images, actions, table, title)
        render(args.output / f'{name}.mp4', images, wrist, actions, table[args.video_epsilon], f'{title}  (eps {args.video_epsilon})')
        summary[name] = {'file': fname, 'frames': len(actions), 'segments': {str(e): v for e, v in table.items()}}
    (args.output / 'segments.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
