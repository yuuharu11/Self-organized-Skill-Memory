"""Official dense pi05 normal path, without any Expert overlay or memory."""
from pathlib import Path
import sys,os,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pi05_memory'))
from serve import MemoryPolicy

class NormalPolicy(MemoryPolicy):
    def infer(self, observation):
        assert observation["s2_condition"]=="normal"
        result=super().infer(observation)
        result["s2_debug"].update(query_seed=int(observation["s2_rng_seed"]),num_steps=10)
        return result

def main():
    import torch
    torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--port',type=int,required=True)
    a=p.parse_args();policy=NormalPolicy(a.checkpoint,a.checkpoint,normal_only=True)
    # This normal-only constructor never reads a bridge checkpoint or memory.
    WebsocketPolicyServer(policy,host='127.0.0.1',port=a.port,metadata={'policy':'public_pi05_libero_normal','num_steps':10}).serve_forever()
if __name__=='__main__':main()
