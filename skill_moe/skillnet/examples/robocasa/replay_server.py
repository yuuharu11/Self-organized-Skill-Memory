"""Websocket policy that replays one saved LeRobot action sequence verbatim."""
import argparse, asyncio, json, pathlib
import numpy as np
import websockets.asyncio.server
from websockets.exceptions import ConnectionClosed
from skillnet_client import msgpack_numpy

class Replay:
    def __init__(self,actions,replan): self.actions=actions; self.replan=replan; self.i=0
    def infer(self,observation):
        del observation
        out=self.actions[self.i:self.i+self.replan]; self.i+=len(out)
        if len(out)==0: out=self.actions[-1:]
        if len(out)<10: out=np.concatenate([out,np.repeat(out[-1:],10-len(out),axis=0)])
        return {'actions':dataset_to_env_actions(out).astype(np.float32)}

def dataset_to_env_actions(actions):
    """Convert PandaOmron LeRobot order to ``env_utils.convert_action`` order."""
    actions=np.asarray(actions)
    if actions.shape[-1] != 12: raise ValueError(f'expected 12 action values, got {actions.shape}')
    return np.concatenate([actions[...,5:8],actions[...,8:11],actions[...,11:12],actions[...,0:5]],axis=-1)

async def serve(args):
    import pyarrow.compute as pc, pyarrow.parquet as pq
    root=pathlib.Path(args.dataset); tables=[pq.read_table(p) for p in sorted((root/'data').glob('**/*.parquet'))]
    table=__import__('pyarrow').concat_tables(tables); table=table.filter(pc.equal(table['episode_index'],args.episode))
    actions=np.stack([np.asarray(x) for x in table['action'].to_pylist()]).astype(np.float32)
    policy=Replay(actions,args.replan_steps); packer=msgpack_numpy.Packer()
    async def handler(ws):
        try:
            await ws.send(packer.pack({'policy':'demo-replay','action_dim':12,'steps':len(actions)}))
            async for msg in ws:
                obs=msgpack_numpy.unpackb(msg); await ws.send(packer.pack(policy.infer(obs)))
        except ConnectionClosed:
            # The evaluation client may close its websocket when the single
            # rollout ends without a close frame; this is normal for smoke.
            return
    async with websockets.asyncio.server.serve(handler,args.host,args.port,compression=None,max_size=None):
        print(f'replay server listening {args.host}:{args.port}, {len(actions)} actions',flush=True)
        await asyncio.Future()
def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset',required=True); p.add_argument('--episode',type=int,default=0); p.add_argument('--host',default='0.0.0.0'); p.add_argument('--port',type=int,default=8000); p.add_argument('--replan-steps',type=int,default=5); asyncio.run(serve(p.parse_args()))
if __name__=='__main__': main()
