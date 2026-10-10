import sys,unittest
from pathlib import Path
base=Path(__file__).parent;sys.path[:0]=[str(base),str(base/'source/core')]
from fractions import Fraction as F
import numpy as np
import cupy as cp
import mpmath as mp
from scipy.sparse import diags,csr_array
from metric_prototype import compile_metric
from fused_dd import prepare_fused,evaluate_fused
from test_metric_prototype import operator,coefficients
from check_high_precision import oracle,metrics

class FusedTests(unittest.TestCase):
    def test_nearest_float_floor_and_derivative(self):
        x=np.random.default_rng(3).normal(size=(16,4));a=operator(16);bs=coefficients(4,24)
        for derivative in (False,True):
            e=diags(np.where(np.arange(16)%2,-1.,1.),format='csr') if derivative else None
            plan=compile_metric(a,[bs],0,direction=e,derivative_tolerance=0 if derivative else None,parameter_radius=F(1,64) if derivative else 0)
            prepared=prepare_fused(plan);tensor=cp.asarray(x)
            for theta in (-F(1,64),0,F(1,64)) if derivative else (0,):
                reference=oracle(a,bs,x,derivative,theta,384)
                actual=evaluate_fused(prepared,tensor,theta=theta);actual=actual if isinstance(actual,tuple) else (actual,)
                rounded=tuple(np.array([[float(v) for v in row] for row in target]) for target in reference)
                quality=metrics(tuple(v.get() for v in actual),reference,x);floor=metrics(rounded,reference,x)
                for error,minimum in zip(quality['error_per_input_norm'],floor['error_per_input_norm']):self.assertLessEqual(error,minimum*(1+1e-10)+1e-30)
    def test_boundaries_and_owned_output(self):
        prepared=prepare_fused(compile_metric(operator(8),[coefficients(2,6)],0));x=cp.ones((8,2))
        first=evaluate_fused(prepared,x);copy=first.copy();evaluate_fused(prepared,x*2)
        np.testing.assert_array_equal(first.get(),copy.get())
        for value in (x.astype(cp.float32),x*cp.nan,cp.ones((7,2)),cp.ones((8,4))[:,::2]):
            with self.assertRaises(ValueError):evaluate_fused(prepared,value)
        with self.assertRaises(ValueError):evaluate_fused(prepared,x,theta=1)
        with self.assertRaises(ValueError):prepare_fused(compile_metric(operator(8),[coefficients(2,6)],0,direction=csr_array(np.ones((8,8))),derivative_tolerance=0))

if __name__=='__main__':unittest.main(verbosity=2)
