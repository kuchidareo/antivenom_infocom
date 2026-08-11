import unittest
import numpy as np
from generate_inputs import generate_chain,outcomes_to_windows
from import_feature_map import nchw_to_windows
from markov import *
class Tests(unittest.TestCase):
 def test_all_record_patterns_and_initial_comparison(self):
  r=np.array([[(v>>2)&1,(v>>1)&1,v&1] for v in range(8)],np.uint8).reshape(-1);x=outcomes_to_windows(r,np.random.default_rng(1));np.testing.assert_array_equal(record_outcomes(x),r);b=comparison_outcomes(x).reshape(-1,4);np.testing.assert_array_equal(b[:,0],1);np.testing.assert_array_equal(b[:,1:].reshape(-1),r)
 def test_estimator(self):
  b=generate_chain(500000,.2,.7,np.random.default_rng(2));m=analyze_outcomes(b);self.assertAlmostEqual(m.p_1_given_0,.2,delta=.005);self.assertAlmostEqual(m.p_0_given_1,.7,delta=.005)
 def test_two_bit(self):
  b=generate_chain(1000000,.35,.8,np.random.default_rng(3));self.assertAlmostEqual(analyze_outcomes(b).two_bit_counter_error_rate,empirical_two_bit_error(b),delta=.003)
 def test_nchw_order(self):
  x=np.arange(16,dtype=np.float32).reshape(1,1,4,4);expected=np.array([[0,1,4,5],[2,3,6,7],[8,9,12,13],[10,11,14,15]],np.float32);np.testing.assert_array_equal(nchw_to_windows(x),expected)
if __name__=='__main__':unittest.main()
