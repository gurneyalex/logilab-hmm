import autopath
from support import timecall

from logilab.hmm.hmm import HMM
from logilab.hmm.hmmc import HMM_C
from logilab.hmm.hmmf import HMM_F

from numpy import array, take, alltrue


def test_time_alpha():
    S = range(20)
    O = range(40)
    OBS = range(40)*4
    test = HMM( S, O )
    test.setRandomProba()
    test1 = HMM( S, O, test.A, test.B, test.pi )
    test2 = HMM_F( S, O, test.A, test.B, test.pi )
    test3 = HMM_C( S, O, test.A, test.B, test.pi )
    bo = take( test.B, OBS, 0 )
    timecall( "HMM.AlphaScaled  ", test1.AlphaScaled, test.A, bo, test.pi )
    timecall( "HMM_F.AlphaScaled", test2.AlphaScaled, test.A, bo, test.pi )
    timecall( "HMM_C.AlphaScaled", test3.AlphaScaled, test.A, bo, test.pi )

def test_time_beta():
    S = range(40)
    O = range(80)
    OBS = range(80)*4
    test = HMM( S, O )
    test.setRandomProba()
    test1 = HMM( S, O, test.A, test.B, test.pi )
    test2 = HMM_F( S, O, test.A, test.B, test.pi )
    test3 = HMM_C( S, O, test.A, test.B, test.pi )
    bo = take( test.B, OBS, 0 )
    a1, s1 = test1.AlphaScaled( test.A, bo, test.pi )
    print "A.shape  = ", test.A.shape
    print "bo.shape = ", bo.shape
    print "s1.shape = ", s1.shape
    timecall( "HMM.BetaScaled   ", test1.BetaScaled, test.A, bo, s1 )
    timecall( "HMM_F.BetaScaled ", test2.BetaScaled, test.A, bo, s1 )
    timecall( "HMM_C.BetaScaled ", test3.BetaScaled, test.A, bo, s1 )

#test_time_alpha()
test_time_beta()
