"""KC2 contract tests: inputs, output shapes, Dev selection, paired-count labels."""
from __future__ import annotations
import inspect,sys,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

HERE=Path(__file__).resolve().parents[2]/'skill_moe/skillnet/examples/memory_pilot/unseen_libero'
sys.path.insert(0,str(HERE))
import kc2_run as K


class KC2ContractTests(unittest.TestCase):
    def test_joint_model_keeps_q_256_and_projects_only_for_search(self):
        torch.set_num_threads(1)
        model=K.KC2Net('i','dot').eval()
        fake=SimpleNamespace(window=lambda ix:(torch.zeros((len(ix),10,16)),torch.ones((len(ix),10),dtype=torch.bool)))
        with torch.no_grad():
            q=model(fake,torch.tensor([0,1]));g=model.grip_logits(q);k=torch.zeros((7,18));s=model.score(q,k)
        self.assertEqual(tuple(q.shape),(2,256))
        self.assertEqual(tuple(g.shape),(2,10))
        self.assertEqual(tuple(model.project(q).shape),(2,18))
        self.assertEqual(tuple(s.shape),(2,7))
        self.assertEqual(list(inspect.signature(model.forward).parameters),['gs','ix'])

    def test_distance_and_dot_scores_are_the_specified_formulas(self):
        torch.set_num_threads(1)
        m=K.KC2Net('i','dot').eval();q=torch.zeros((2,256));keys=torch.ones((3,18))
        with torch.no_grad():
            m.project.weight.zero_();m.project.bias.fill_(1.)
            expected=m.logit_scale.exp().clamp(max=100.)*(m.project(q)@keys.T)
            torch.testing.assert_close(m.score(q,keys),expected)
            m.score_mode='distance'
            expected=-torch.cdist(m.project(q),keys).square()*m.logit_scale.exp().clamp(max=100.)
            torch.testing.assert_close(m.score(q,keys),expected)

    def test_dev_selection_enforces_kc1_plus_five_percent_then_minimizes_switches(self):
        limit=K.DEV_LIMIT
        rows=[{'name':'too-good-gripper-but-fails-search','lambda':.1,'dev_raw_coeff_sq':limit+1,
               'dev_gripper_errors':0,'dev_gripper_exact_accuracy':1.},
              {'name':'eligible-2','lambda':.3,'dev_raw_coeff_sq':limit,'dev_gripper_errors':12,
               'dev_gripper_exact_accuracy':.8},
              {'name':'eligible-1','lambda':1.,'dev_raw_coeff_sq':limit-.2,'dev_gripper_errors':9,
               'dev_gripper_exact_accuracy':.75}]
        self.assertEqual(K.select_joint_candidate(rows)['name'],'eligible-1')
        self.assertFalse(K.within_retrieval_constraint(limit+1e-4))

    def test_paired_success_counts_have_correct_winner_orientation(self):
        kc={('a',0):1,('a',1):1,('a',2):0,('a',3):0}
        ref={('a',0):1,('a',1):0,('a',2):1,('a',3):0}
        got=K.paired_counts(kc,ref)
        self.assertEqual(got['kc2_only'],1)
        self.assertEqual(got['reference_only'],1)
        self.assertTrue(got['success_difference_check'])
        kc[('a',4)]=1;ref[('a',4)]=0
        got=K.paired_counts(kc,ref)
        self.assertEqual((got['kc2_only'],got['reference_only']),(2,1))
        self.assertTrue(got['success_difference_check'])

    def test_kc1_report_table_winner_columns_were_corrected(self):
        import os
        p=Path(os.environ.get('SKILLNET_STATE_DIR',Path(__file__).resolve().parents[2]/'.skillnet-state'))/'outputs/memory-pilot/unseen-libero/kc1-01/report.md'
        if not p.exists(): self.skipTest('KC1 report not present')
        text=p.read_text()
        self.assertIn('| libero90 | T5 AtopkGdec | 7 | 4 |',text)
        self.assertIn('| libero10 | π0.5 | 3 | 25 |',text)

    def test_gripper_decoder_starts_from_last_sign_and_switches_once(self):
        import kq2cl_common as C
        prob=np.asarray([[.99,.99,.01,.01,.99,.01,.99,.01,.99,.01]],np.float32)
        got=C.decode_gripper(prob,np.asarray([1.]),.5)
        np.testing.assert_array_equal(got,np.asarray([[1.,1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.]],np.float32))

    def test_dense_dev_threshold_selection_normalizes_open_close_labels(self):
        from kc2_eval import choose_threshold
        probs=np.full((2,10),.05,np.float32);probs[1,4:]=.95
        target=np.zeros((2,10),np.float32);target[1,4:]=1.
        th,metric,_=choose_threshold(probs,target,np.asarray([-1.,-1.]),np.asarray([True,True]))
        self.assertGreater(th,0.)
        self.assertEqual(metric['errors'],0)
        self.assertEqual(metric['exact_step_accuracy'],1.)


if __name__=='__main__':unittest.main()
