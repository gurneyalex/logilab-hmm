"""
These test suite compare the 3 implementations of hmm (py,C,fortran)
"""

import autopath

from logilab.hmm.hmm import HMM
from logilab.hmm.hmmc import HMM_C
from logilab.hmm.hmmf import HMM_F

from numpy import array

def test1_analyze( h, chain ):
    """simple test"""
    print "Chain      : ", chain
    print "analyse    : ", h.analyze(chain)
    print "analyse_log: ", h.analyze_log(chain)
    
def test1(HMM):
    """Simple test, that will check if the viterbi algorithm
    correctly determine the hidden states of a HMM."""
    test = HMM(['a', 'b'], ['s1', 's2', 's3'],
               array([[.3, .7], [.5, .5]]),
               array([[.5, 0], [.5, .5], [0, .5]]),
               array([.9, .1]))
    test.dump()
    test1_analyze(test, ['s1'] * 3)
    test1_analyze(test, ['s1'] * 3 + ['s3'] * 3)
    test1_analyze(test, ['s1', 's2', 's3'] * 3)

    test.setRandomProba()
    test.dump()
    test1_analyze(test, ['s1'] * 3)
    test1_analyze(test, ['s1'] * 3 + ['s3'] * 3)
    test1_analyze(test, ['s1', 's2', 's3'] * 3)



test1(HMM)
test1(HMM_C)
test1(HMM_F)
