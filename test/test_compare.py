"""
These test suite compare the 3 implementations of hmm (py,C,fortran)
"""

import autopath
from support import timecall

from logilab.hmm.hmm import HMM
from logilab.hmm.hmmc import HMM_C
from logilab.hmm.hmmf import HMM_F

from numpy import array, take, alltrue, allclose

RTOL = 1e-7
ATOL = 1e-10

def _test_simu(HMM, sample, A0, B0, pi0):
    test = HMM(['a', 'b'], ['s1', 's2', 's3'], A0, B0, pi0)
    test.learn(sample, None, 3000)
    print 'trained values for type', HMM
    print 'A =', test.A
    print 'B =', test.B
    print 'pi =', test.pi

def test_simu():
    """Train a model over some simulated values from an initial
    model"""
    test = HMM(['a', 'b'], ['s1', 's2', 's3'])
    test.setRandomProba()
    print 'Original'
    print 'A =', test.A
    print 'B =', test.B
    print 'pi =', test.pi
    print
    print 'Generating sample data...'
    sample =  test.simulate(500)
    print 'Randomizing model...'
    test2 = HMM(['a', 'b'], ['s1', 's2', 's3'])
    test2.setRandomProba()
    for _HMM in (HMM, HMM_C, HMM_F):
        _test_simu( _HMM, sample, test2.A, test2.B, test2.pi )

def test_alpha_scaled():
    A0 = array([[.3, .7], [.5, .5]])
    B0 = array([[.5, 0], [.5, .5], [0, .5]])
    pi0 = array([.9, .1])
    test1 = HMM(['a', 'b'], ['s1', 's2', 's3'], A0, B0, pi0)
    test2 = HMM_F(['a', 'b'], ['s1', 's2', 's3'], A0, B0, pi0)
    test3 = HMM_C(['a', 'b'], ['s1', 's2', 's3'], A0, B0, pi0)

    obs = array([0,0,1,0,0,0,1,2,2,2,1,0])
    bo = take( B0, obs, 0)
    a1, s1 = test1.AlphaScaled( A0, bo, pi0 )
    a2, s2 = test2.AlphaScaled( A0, bo, pi0 )
    a3, s3 = test3.AlphaScaled( A0, bo, pi0 )

    assert allclose(a1,a2,RTOL,ATOL)
    assert allclose(a1,a3,RTOL,ATOL)
    assert allclose(s1,s2,RTOL,ATOL)
    assert allclose(s1,s3,RTOL,ATOL)
    print "AlphaScaled ok"

def test_beta_scaled():
    A0 = array([[.3, .7], [.5, .5]])
    B0 = array([[.5, 0], [.5, .5], [0, .5]])
    pi0 = array([.9, .1])
    test1 = HMM(['a', 'b'], ['s1', 's2', 's3'], A0, B0, pi0)
    test2 = HMM_F(['a', 'b'], ['s1', 's2', 's3'], A0, B0, pi0)
    test3 = HMM_C(['a', 'b'], ['s1', 's2', 's3'], A0, B0, pi0)

    obs = array([0,0,1,0,0,0,1,2,2,2,1,0])
    bo = take( B0, obs, 0)
    a1, s1 = test1.AlphaScaled( A0, bo, pi0 )

    beta1 = test1.BetaScaled( A0, bo, s1 )
    beta2 = test2.BetaScaled( A0, bo, s1 )
    beta3 = test3.BetaScaled( A0, bo, s1 )

    assert allclose( beta1, beta2, RTOL, ATOL )
    assert allclose( beta1, beta3, RTOL, ATOL )
    print "BetaScaled ok"

test_alpha_scaled()
test_beta_scaled()
