#!/usr/bin/env python3
"""Inspect RoboCasa LeRobot metadata; fall back to pyarrow for version skew."""
import argparse, json, pathlib
import numpy as np
def main():
    p=argparse.ArgumentParser(); p.add_argument('path',nargs='?',default=None); a=p.parse_args()
    root=pathlib.Path(a.path or pathlib.Path(__import__('os').environ.get('SKILLNET_STATE_DIR','.'))/'data/robocasa')
    found=list(root.glob('**/meta/info.json'))
    if not found: raise SystemExit(f'No LeRobot meta/info.json under {root}; run data_process/robocasa/download.py --preset smoke')
    info=json.loads(found[0].read_text()); print('dataset:',found[0].parent.parent); print('fps:',info.get('fps'),'episodes:',info.get('total_episodes')); print('features:')
    for name,spec in info.get('features',{}).items(): print(f"  {name}: shape={spec.get('shape')} dtype={spec.get('dtype')}")
    print('action modality (verified from robocasa.utils.env_utils.convert_action):')
    for k,v in {'eef_position':[0,3],'eef_rotation':[3,6],'gripper_close':[6,7],'base_motion':[7,11],'control_mode':[11,12]}.items(): print(f'  {k}: action[{v[0]}:{v[1]}]')
    dataset_root=found[0].parent.parent
    try:
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
        except ImportError:
            from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        metadata=LeRobotDatasetMetadata(str(dataset_root),root=dataset_root)
        print('LeRobot metadata API: available; task count:',len(getattr(metadata,'tasks',{})))
    except Exception as exc:
        print('LeRobot API unavailable/incompatible; using parquet fallback:',repr(exc))
        import pyarrow.parquet as pq
        parquet=next(dataset_root.glob('data/**/*.parquet'))
        table=pq.read_table(parquet)
        print('parquet sample:',parquet.name,'columns:',table.column_names,'rows:',table.num_rows)
        for name in ('action','actions','observation.state'):
            if name in table.column_names:
                value=table[name][0].as_py(); print(f'  {name}: shape=({len(value)},) dtype={np.asarray(value).dtype}')
if __name__=='__main__': main()
