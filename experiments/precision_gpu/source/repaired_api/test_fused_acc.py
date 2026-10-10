import sys,unittest
from pathlib import Path
base=Path(__file__).parent;sys.path[:0]=[str(base),str(base/'source/core')]
from fractions import Fraction as F
import numpy as np,cupy as cp
from scipy.sparse import diags,csr_array
from metric_prototype import compile_metric,evaluate_metric
from fused_acc import prepare_acc,evaluate_acc
from test_metric_prototype import operator,coefficients

class AccumulatorTests(unittest.TestCase):
    def test_modes_parameters_and_capture(self):
        x=np.random.default_rng(4).normal(size=(32,4));direction=diags(np.where(np.arange(32)%2,-1.,1.),format='csr')
        plan=compile_metric(operator(32),[coefficients(4,24)],0,direction=direction,derivative_tolerance=0,parameter_radius=F(1,64));tensor=cp.asarray(x)
        for compensated,derivative_compensated,projected in ((False,False,False),(True,True,False),(False,True,False),(False,True,True)):
            prepared=prepare_acc(plan,compensated=compensated,derivative_compensated=derivative_compensated,projected=projected)
            for theta in (-F(1,64),0,F(1,64)):
                actual=evaluate_acc(prepared,tensor,theta=theta);expected=evaluate_metric(plan,x,theta=theta)
                for value,target in zip(actual,expected):np.testing.assert_allclose(value.get(),target,rtol=1e-13,atol=1e-14)
            stream=cp.cuda.Stream(non_blocking=True)
            with stream:
                for _ in range(3):evaluate_acc(prepared,tensor,checked=False)
            stream.synchronize()
            with stream:
                stream.begin_capture();result=evaluate_acc(prepared,tensor,checked=False);graph=stream.end_capture()
            graph.launch();cp.cuda.Device().synchronize()
            for value,target in zip(result,evaluate_metric(plan,x)):np.testing.assert_allclose(value.get(),target,rtol=1e-13,atol=1e-14)
    def test_zero_and_wide_channel_guards(self):
        plan=compile_metric(csr_array((8,8)),[[np.eye(4)]],0);x=cp.ones((8,4))
        prepared=prepare_acc(plan,compensated=False,projected=True)
        np.testing.assert_array_equal(evaluate_acc(prepared,x).get(),x.get())
        with self.assertRaises(ValueError):prepare_acc(compile_metric(operator(8),[[np.eye(17)]],0))
        from types import SimpleNamespace
        with self.assertRaises(ValueError):prepare_acc({'plan':{'a':SimpleNamespace(shape=(2**30,2**30)),'float_coefficients':[np.eye(4)],'degree':1}},projected=True)
    def test_final_compensated_high_projection(self):
        x=np.random.default_rng(5).normal(size=(32,16));plan=compile_metric(operator(32),[coefficients(16,20)],0)
        prepared=prepare_acc(plan,compensated=False,projected=True,coefficient_correction=False,final_compensated=True)
        value=evaluate_acc(prepared,cp.asarray(x));expected=evaluate_metric(plan,x)
        np.testing.assert_allclose(value.get(),expected,rtol=1e-13,atol=1e-14)

if __name__=='__main__':unittest.main(verbosity=2)
