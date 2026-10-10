import sys,unittest
from pathlib import Path
base=Path(__file__).parent;sys.path[:0]=[str(base),str(base/'source/core')]
from fractions import Fraction as F
import numpy as np
from scipy.sparse import diags
from metric_prototype import compile_metric
from coefficient_certificate import encoding_certificate
from test_metric_prototype import operator,coefficients

class EncodingTests(unittest.TestCase):
    def test_encoding_and_total_budgets(self):
        for derivative in (False,True):
            plan=compile_metric(operator(32),[coefficients(4)],F(1,10**16),direction=diags(np.where(np.arange(32)%2,-1.,1.),format='csr') if derivative else None,derivative_tolerance=F(1,10**16) if derivative else None,parameter_radius=F(1,64) if derivative else 0)
            result=encoding_certificate(plan)
            self.assertGreater(result['coefficient_error_bound'],0)
            self.assertLess(result['coefficient_error_bound'],F(1,10**30))
            self.assertLessEqual(result['value_tail_and_encoding_bound'],F(1,10**16))
            if derivative:self.assertLessEqual(result['derivative_tail_and_encoding_bound'],F(1,10**16))
    def test_exact_dyadic_encoding(self):
        result=encoding_certificate(compile_metric(operator(4),[[[[F(1,2)]],[[F(1,4)]]]],0))
        self.assertEqual(result['coefficient_error_bound'],0)

if __name__=='__main__':unittest.main(verbosity=2)
