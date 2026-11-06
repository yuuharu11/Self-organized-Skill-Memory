"""Reuse the paired harness exactly; add passive video/final-frame recording."""
from pathlib import Path
import os,sys,json
import numpy as np

def main():
    import torch
    torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))
    here=Path(__file__).resolve().parent
    sys.path.insert(0,str(here.parent/'anchor_flow'))
    import run_case
    # No task-level annotation is needed by the normal policy's whitelist.
    run_case._load_annotation=lambda *args:{'all_classes':[],'objects':[]}
    job=json.loads(Path(sys.argv[1]).read_text());result_path=Path(sys.argv[2]);artifacts={}
    if job.get('record'):
        import libero.libero.envs as envs
        import imageio.v2 as imageio
        original=envs.OffScreenRenderEnv
        class RecordedEnv(original):
            def __init__(self,*args,**kwargs):
                self.frames=[];self.last_observation=None;self.record_steps=0;super().__init__(*args,**kwargs)
            def frame(self,obs):
                from PIL import Image
                a=np.asarray(Image.fromarray(np.ascontiguousarray(obs['agentview_image'][::-1,::-1])).resize((128,128)))
                b=np.asarray(Image.fromarray(np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1,::-1])).resize((128,128)))
                return np.concatenate([a,b],axis=1)
            def set_init_state(self,*args,**kwargs):
                obs=super().set_init_state(*args,**kwargs);self.last_observation=obs;imageio.imwrite(result_path.parent/'initial.png',self.frame(obs));artifacts['initial_frame']=str(result_path.parent/'initial.png');return obs
            def step(self,*args,**kwargs):
                r=super().step(*args,**kwargs);self.last_observation=r[0];self.record_steps+=1
                if self.record_steps%5==0:self.frames.append(self.frame(r[0]))
                return r
            def close(self):
                try:
                    if self.last_observation is not None:
                        p=result_path.parent/'final.png';imageio.imwrite(p,self.frame(self.last_observation));artifacts['final_frame']=str(p)
                    if self.frames:
                        p=result_path.parent/'two_views.mp4';imageio.mimwrite(p,self.frames,fps=4);artifacts['video']=str(p)
                finally:super().close()
        envs.OffScreenRenderEnv=RecordedEnv
    result=run_case._run_trial(job);result['artifacts']=artifacts
    result['import_audit']={'run_case':run_case.__file__,'torch_threads':torch.get_num_threads(),'affinity':sorted(os.sched_getaffinity(0))}
    result_path.write_text(json.dumps(result,ensure_ascii=False)+'\n')
if __name__=='__main__':main()
