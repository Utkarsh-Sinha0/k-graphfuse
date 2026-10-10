import subprocess
import sys
import unittest

import numpy as np
from scipy.sparse import csr_array
import k_graphfuse as kgf


class PackageChecks(unittest.TestCase):
    def test_namespace_executes_ordered_polynomial(self):
        a = csr_array([[0., 0.5], [0.25, 0.]])
        x = np.array([[1.], [2.]])
        plan = kgf.compile_sparse(a, [[[[1]], [[1]]], [[[2]], [[1]]]])
        np.testing.assert_allclose(kgf.evaluate_sparse(plan, x),
                                   (2*np.eye(2)+a.toarray()) @ (np.eye(2)+a.toarray()) @ x)
        self.assertEqual(kgf.__version__, "0.1.0")

    def test_cpu_import_does_not_import_torch(self):
        subprocess.run([sys.executable, "-c",
                        "import sys,k_graphfuse; assert 'torch' not in sys.modules"], check=True)


if __name__ == "__main__":
    unittest.main()
