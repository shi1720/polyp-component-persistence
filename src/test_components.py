import itertools, unittest
import numpy as np
from scipy import ndimage as ndi
from components import component_masks,crc_index,suffix_envelope

class Tests(unittest.TestCase):
    def test_switch_counterexample(self):
        s=np.zeros((5,9)); s[1,1:3]=.9; s[3,5:8]=.8
        paths=list(component_masks(s,[1,.9,.8,0]))
        y=np.zeros_like(s,bool); y[1,1:3]=True
        loss=[1-(z['lcc']&y).sum()/y.sum() for z in paths]
        self.assertEqual(loss,[1,0,1,0])
        self.assertTrue(np.all(paths[2]['prefix'][y]))
    def test_exhaustive_three_level_images(self):
        # Every 2x3 image over three score levels, for both connectivities.
        for flat in itertools.product([0,.5,1],repeat=6):
            s=np.array(flat).reshape(2,3)
            for conn in [1,2]:
                path=list(component_masks(s,[1,.5,0],conn,min_area=2))
                for j in range(3):
                    union=np.logical_or.reduce([z['lcc'] for z in path[:j+1]])
                    self.assertTrue(np.array_equal(union,path[j]['prefix']))
                    for key in ['raw','prefix','anchor','opening']:
                        self.assertFalse(np.any(path[j][key]&~path[j]['raw']))
                        if j: self.assertFalse(np.any(path[j-1][key]&~path[j][key]))
    def test_suffix_envelope(self):
        x=np.array([[1,0,1,0],[1,.5,.7,0]])
        e=suffix_envelope(x)
        self.assertTrue(np.all(e>=x)); self.assertTrue(np.all(np.diff(e,axis=-1)<=0))
        for j in range(x.shape[1]):self.assertTrue(np.array_equal(e[:,j],x[:,j:].max(axis=-1)))
    def test_finite_population_crc(self):
        # Exact leave-one-out expectation over deterministic finite populations.
        rng=np.random.default_rng(1720)
        for _ in range(100):
            x=np.sort(rng.uniform(size=(10,9)),axis=1)[:,::-1]; x=np.c_[x,np.zeros(10)]
            for alpha in [.2,.4,.7]:
                risks=[x[i,crc_index(np.delete(x,i,axis=0),alpha)] for i in range(10)]
                self.assertLessEqual(np.mean(risks),alpha+1e-12)
if __name__=='__main__':unittest.main()
