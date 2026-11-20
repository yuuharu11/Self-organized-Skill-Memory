"""XF3 contract: label direction, timing position, aligned 20Hz inputs, inference contract."""
from __future__ import annotations
import inspect,json,sys,unittest
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parents[2]/'skill_moe/skillnet/examples/memory_pilot/unseen_libero'
sys.path.insert(0,str(HERE))
import xf3_run as X


class XF3Tests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_close_sign_and_first_switch_position(self):
        a=np.zeros((2,10,7),np.float32)
        a[0,4:,6]=1.
        a[1,:,6]=1.
        signs=X.close_sign(a)
        np.testing.assert_array_equal(signs[0,:],[-1,-1,-1,-1,1,1,1,1,1,1])
        np.testing.assert_array_equal(signs[1,:],np.ones(10))
        np.testing.assert_array_equal(X.switch_class(signs,np.asarray([-1.,1.])),[4,10])

    def test_hazard_decoder_switches_once_at_class_and_direction(self):
        p=np.eye(11,dtype=np.float32)[[4,10]]
        decoded,cls=X.hazard_decode(p,np.asarray([-1.,1.]),0.)
        np.testing.assert_array_equal(cls,[4,10])
        np.testing.assert_array_equal(decoded[0],[-1,-1,-1,-1,1,1,1,1,1,1])
        np.testing.assert_array_equal(decoded[1],np.ones(10))

    def test_all_input_modes_have_expected_output_shapes(self):
        torch.manual_seed(1);q=torch.randn(2,256);gx=torch.randn(2,4);v=torch.randn(2,34,2048)
        for mode in X.INPUTS:
            m=X.XF3Decoder(mode,'bce')
            args=(q,gx,None) if mode=='G-q' else (q,gx,v) if mode=='G-qv' else (None,gx,v)
            self.assertEqual(tuple(m(*args).shape),(2,10))
            h=X.XF3Decoder(mode,'hazard')
            self.assertEqual(tuple(h(*args).shape),(2,11))

    def test_inference_signature_excludes_future_labels(self):
        names=list(inspect.signature(X.XF3Decoder.forward).parameters)
        self.assertEqual(names,['self','q','grip_extra','vlm'])
        self.assertFalse({'target','actions','task_id','pi05','future'}&set(names))

    def test_switch_rows_are_oversampled_exactly_fourfold(self):
        idx=np.arange(5);classes=np.asarray([10,4,10,0,10])
        got=X._weighted_switch_rows(idx,classes)
        self.assertEqual(len(got),11)
        self.assertEqual(int((got==1).sum()),4)
        self.assertEqual(int((got==3).sum()),4)

    def test_vlm_tokens_and_20hz_rows_are_aligned(self):
        root=X.KC2_OUT/'data'
        for split in ('dev2','unseen20'):
            d=root/split
            self.assertTrue((d/'VLM_DONE').is_file())
            keys=np.load(d/'key.npy');tokens=np.load(d/'vlm.npy',mmap_mode='r')
            self.assertEqual(tokens.shape,(len(keys),34,2048))
            self.assertEqual(tokens.dtype,np.float16)
        info=json.loads((root/'unseen20'/'check.json').read_text())
        self.assertGreater(info['rows'],0)
        self.assertTrue((root/'unseen20'/'VLM_DONE').exists())
        import vq_geometry,dev_v2
        np.testing.assert_array_equal(vq_geometry.load_rate_matched('unseen20')['key'],
                                      np.load(root/'unseen20'/'key.npy'))
        np.testing.assert_array_equal(dev_v2.load_dev_v2_20()['key'],np.load(root/'dev2'/'key.npy'))


if __name__=='__main__':unittest.main()
