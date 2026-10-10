import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'k-graphfuse/src'))
from fractions import Fraction as F
import unittest
import numpy as np
from scipy.sparse import csr_array,diags
from c2fuse import fuse
from sparse_c2fuse import compile_sparse,evaluate_sparse
from sensitivity_c2fuse import compile_sensitivity,evaluate_sensitivity
from metric_prototype import compile_full,compile_metric,evaluate_metric,prepare_metric_cuda,evaluate_metric_cuda

def operator(n):
    rows=np.arange(n);even=rows[::2];odd=rows[1::2]
    return csr_array((np.r_[np.tile([-.2,-.1],n//2),np.ones(n//2),np.full(n//2,1/64)],
        (np.r_[rows,even,odd],np.r_[rows,odd,(odd+1)%n])),shape=(n,n))

def coefficients(q,degree=48,seed=20261010):
    rng=np.random.default_rng(seed)
    return [np.eye(q,dtype=int)]+[np.array([[F(int(v),10*3**k) for v in row]
        for row in rng.integers(-2,3,(q,q))],dtype=object) for k in range(1,degree+1)]

class MetricTests(unittest.TestCase):
    def test_ordered_full_fusion(self):
        a=operator(6);x=np.arange(12,dtype=float).reshape(6,2)
        stages=[[[[1,1],[0,1]],[[0,2],[1,0]]],[[[2,0],[1,1]],[[1,0],[0,1]]]]
        expected=evaluate_sparse(compile_sparse(a,stages,0,bounds='fast'),x)
        np.testing.assert_allclose(evaluate_sparse(compile_full(a,stages),x),expected,rtol=2e-14,atol=1e-14)
    def test_zero_operator(self):
        a=csr_array((4,4));stages=[coefficients(2,degree=4)];x=np.ones((4,2))
        plan=compile_full(a,stages);self.assertEqual(plan['degree'],0)
        np.testing.assert_array_equal(evaluate_sparse(plan,x),x)
    def test_metric_error_budget(self):
        a=operator(60);bs=coefficients(4);x=np.random.default_rng(1).normal(size=(60,4))
        base=compile_sparse(a,[bs],F(1,10**8),bounds='fast');new=compile_metric(a,[bs],F(1,10**8))
        reference=evaluate_sparse(compile_full(a,[bs]),x)
        self.assertLess(new['plan']['degree'],base['degree'])
        self.assertLessEqual(new['original_error_bound'],F(1,10**8))
        self.assertLessEqual(np.linalg.norm(evaluate_metric(new,x)-reference)/np.linalg.norm(x),1e-8)
        self.assertEqual(new['condition_penalty'],8)
    def test_affine_value_and_derivative(self):
        a=operator(40);direction=diags(np.where(np.arange(40)%2,-1.,1.),format='csr');bs=coefficients(4,24)
        x=np.random.default_rng(2).normal(size=(40,4));budget=F(1,10**8)
        new=compile_metric(a,[bs],budget,direction=direction,derivative_tolerance=budget,parameter_radius=F(1,64))
        full=compile_sensitivity(a,direction,[bs],0,0,parameter_radius=F(1,64))
        self.assertLessEqual(new['original_derivative_error_bound'],budget)
        for theta in (-1/64,0,1/64):
            for actual,expected in zip(evaluate_metric(new,x,theta=theta),evaluate_sensitivity(full,x,theta=theta)):
                self.assertLessEqual(np.linalg.norm(actual-expected)/np.linalg.norm(x),1e-8)
    def test_invalid_inputs(self):
        for a in (csr_array([[np.inf]]),csr_array([[1j]]),csr_array((2,3))):
            with self.assertRaises(ValueError):compile_full(a,[[[[1]]]])
        new=compile_metric(operator(4),[coefficients(1,4)],1e-8)
        with self.assertRaises(ValueError):evaluate_metric(new,np.ones((4,1))*np.nan)
        with self.assertRaises(ValueError):compile_metric(operator(4),[coefficients(1,4)],-1)
    def test_checked_cuda_api(self):
        import torch
        if not torch.cuda.is_available():self.skipTest('CUDA GPU required')
        a=operator(40);bs=coefficients(4,24);x=np.random.default_rng(2).normal(size=(40,4))
        direction=diags(np.where(np.arange(40)%2,-1.,1.),format='csr')
        for derivative in (False,True):
            compiled=compile_metric(a,[bs],F(1,10**8),direction=direction if derivative else None,
                derivative_tolerance=F(1,10**8) if derivative else None,parameter_radius=F(1,64) if derivative else 0)
            prepared=prepare_metric_cuda(compiled,packed=True);tensor=torch.tensor(x,device='cuda',dtype=torch.float64)
            for theta in (-1/64,0,1/64) if derivative else (0,):
                actual=evaluate_metric_cuda(prepared,tensor,theta=theta);expected=evaluate_metric(compiled,x,theta=theta)
                for value,target in zip(actual if isinstance(actual,tuple) else (actual,),expected if isinstance(expected,tuple) else (expected,)):
                    np.testing.assert_allclose(value.cpu().numpy(),target,rtol=1e-13,atol=1e-14)
            with self.assertRaises(ValueError):evaluate_metric_cuda(prepared,tensor.float())
            with self.assertRaises(ValueError):evaluate_metric_cuda(prepared,tensor*torch.nan)
    def test_general_similarity_and_balanced_control(self):
        rng=np.random.default_rng(7);dense=rng.integers(-3,4,(12,12))/16
        dense*=rng.random((12,12))<.25
        stages=[coefficients(3,6,7),coefficients(3,4,8)];x=rng.normal(size=(12,3))
        for value in (dense,(dense+dense.T)/2):
            a=csr_array(value);compiled=compile_metric(a,stages,F(1,10**10))
            expected=evaluate_sparse(compile_full(a,stages),x)
            self.assertLessEqual(np.linalg.norm(evaluate_metric(compiled,x)-expected)/np.linalg.norm(x),1e-10)
            if np.array_equal(value,value.T):self.assertEqual(compiled['condition_penalty'],1)

if __name__=='__main__':unittest.main(verbosity=2)
