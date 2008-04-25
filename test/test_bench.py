from support import timecall
from logilab.hmm.hmm import HMM
from logilab.hmm.hmmc import HMM_C
from logilab.hmm.hmmf import HMM_F
from numpy import take


def test_time_alpha():
    S = range(20)
    O = range(40)
    OBS = range(40)*4
    test = HMM( S, O )
    test.set_random_proba()
    test1 = HMM( S, O, test.A, test.B, test.pi )
    test2 = HMM_F( S, O, test.A, test.B, test.pi )
    test3 = HMM_C( S, O, test.A, test.B, test.pi )
    bo = take( test.B, OBS, 0 )
    timecall( "HMM.alpha_scaled  ", test1.alpha_scaled, test.A, bo, test.pi )
    timecall( "HMM_F.alpha_scaled", test2.alpha_scaled, test.A, bo, test.pi )
    timecall( "HMM_C.alpha_scaled", test3.alpha_scaled, test.A, bo, test.pi )

def test_time_beta():
    S = range(40)
    O = range(80)
    OBS = range(80)*4
    test = HMM( S, O )
    test.set_random_proba()
    test1 = HMM( S, O, test.A, test.B, test.pi )
    test2 = HMM_F( S, O, test.A, test.B, test.pi )
    test3 = HMM_C( S, O, test.A, test.B, test.pi )
    bo = take( test.B, OBS, 0 )
    a1, s1 = test1.alpha_scaled( test.A, bo, test.pi )
    print "A.shape  = ", test.A.shape
    print "bo.shape = ", bo.shape
    print "s1.shape = ", s1.shape
    timecall( "HMM.beta_scaled   ", test1.beta_scaled, test.A, bo, s1 )
    timecall( "HMM_F.beta_scaled ", test2.beta_scaled, test.A, bo, s1 )
    timecall( "HMM_C.beta_scaled ", test3.beta_scaled, test.A, bo, s1 )

#test_time_alpha()
test_time_beta()
