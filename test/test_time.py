import unittest
import os
import autopath
from numpy import array, alltrue, allclose, zeros, ones, take, isfortran
import logilab.hmm.hmm as hmm
import logilab.hmm.hmmc as hmmc
import logilab.hmm.hmmf as hmmf
import logilab.hmm.hmmS as hmmS
from support import timecall

def setProbaEqui(h1, h2, h3):
    h1.A = ones( (h1.N, h1.N), float) / h1.N
    h1.B = ones( (h1.M, h1.N), float) / h1.M
    h1.pi = ones( (h1.N,), float ) / h1.N
    h2.A = h1.A
    h3.A = h1.A
    h2.B = h1.B
    h3.B = h1.B
    h2.pi = h1.pi
    h3.pi = h1.pi

def setProba(h, h1):
    h.A = h1.A
    h.B = h1.B
    h.pi = h1.pi
    
def test_time_alpha(h1, h2, h3, observation):
    bo = take( h1.B, observation, 0 )
    timecall( "HMM.AlphaScaled    ", h1.AlphaScaled, h1.A, bo, h1.pi )
    timecall( "HMM_F.AlphaScaled  ", h2.AlphaScaled, h2.A, bo, h2.pi )
    timecall( "HMM_C.AlphaScaled  ", h3.AlphaScaled, h3.A, bo, h3.pi )

def test_time_beta(h1, h2, h3, obs):
    Bo = take( h1.B, observation, 0 )
    alpha, scale_factors = h3.AlphaScaled( h3.A, Bo, h3.pi )
    timecall( "HMM.BetaScaled     ", h1.BetaScaled, h1.A, Bo, scale_factors )
    timecall( "HMM_F.BetaScaled   ", h2.BetaScaled, h2.A, Bo, scale_factors )
    timecall( "HMM_C.BetaScaled   ", h3.BetaScaled, h3.A, Bo, scale_factors )

def test_time_ksi(h1, h2, h3, obs):
    Bo = take( h1.B, observation, 0 )
    alpha, scale_factors = h3.AlphaScaled( h3.A, Bo, h3.pi )
    beta = h3.BetaScaled(h3.A, Bo, scale_factors)
    timecall( "HMM.Ksi            ", h1.Ksi, h1.A, Bo, alpha, beta )
    timecall( "HMM_F.Ksi          ", h2.Ksi, h2.A, Bo, alpha, beta )
    timecall( "HMM_C.Ksi          ", h3.Ksi, h3.A, Bo, alpha, beta )

def test_time_UpdateIterB(h1, h2, h3, obs):
    Bo = take( h1.B, observation, 0 )
    obsInd = h3._getObservationIndices(observation)
    alpha, scale_factors = h3.AlphaScaled( h3.A, Bo, h3.pi )
    beta = h3.BetaScaled(h3.A, Bo, scale_factors)
    gamma = h3._gamma(alpha, beta, scale_factors)
    B1 = zeros( (h1.M, h1.N), float, order=h1.ORDER )
    B2 = zeros( (h2.M, h2.N), float, order=h2.ORDER )
    B3 = zeros( (h3.M, h3.N), float, order=h3.ORDER )
    timecall( "HMM.UpdateIterB    ", h1.UpdateIterB, gamma, obsInd, B1 )
    timecall( "HMM_F.UpdateIterB  ", h2.UpdateIterB, gamma, obsInd, B2)
    timecall( "HMM_C.UpdateIterB  ", h3.UpdateIterB, gamma, obsInd, B3)

def test_time_CorrectM(h1, h2, h3):
    M = zeros((h1.N, h1.N))
    k = 1
    p = 1./h1.N
    timecall( "HMM.CorrectM       ", h1.CorrectM, M, k, p )
    timecall( "HMM_F.CorrectM     ", h2.CorrectM, M, k, p)
    timecall( "HMM_C.CorretM      ", h3.CorrectM, M, k, p)

def test_time_NormalizeB(h1, h2, h3):
    SGB =  2 * ones(h1.N)
    B_bar = ones( (h1.M, h1.N))
    timecall( "HMM.NormalizeB     ", h1.NormalizeB, B_bar, SGB)
    timecall( "HMM_F.NormalizeB   ", h2.NormalizeB, B_bar, SGB)
    timecall( "HMM_C.NormalizeB   ", h3.NormalizeB, B_bar, SGB)

def test_time_analyze(h1, h2, h3, obs):
    timecall( "HMM.analyze        ", h1.analyze, obs)
    timecall( "HMM_F.analyze      ", h2.analyze, obs)
    timecall( "HMM_C.analyze      ", h3.analyze, obs)

def test_time_analyze_log(h1, h2, h3, obs):
    timecall( "HMM.analyze_log    ", h1.analyze_log, obs)
    timecall( "HMM_F.analyze_log  ", h2.analyze_log, obs)
    timecall( "HMM_C.analyze_log  ", h3.analyze_log, obs)

def test_time_learn(h1, h2, h3, obs):
    timecall( "HMM.learn          ", h1.learn, obs, 1000, 0)
    timecall( "HMM_F.learn        ", h2.learn, obs, 1000, 0)
    timecall( "HMM_C.learn        ", h3.learn, obs, 1000, 0)

def test_time_multiple_learn(h1, h2, h3, chains):
    timecall( "HMM.multiple_learn  ", h1.multiple_learn, chains, 1000, 0)
    timecall( "HMM_F.multiple_learn", h2.multiple_learn, chains, 1000, 0)
    timecall( "HMM_C.multiple_learn", h3.multiple_learn, chains, 1000, 0)

def test_time_ensAveragePall(h1, h2, h3, chains):
    timecall( "HMM.ensembleAveraging", h1.ensemble_averaging, chains, "Pall", 1000, 0)
    timecall( "HMM_F.ensembleAveraging",h2.ensemble_averaging,chains, "Pall", 1000, 0)
    timecall( "HMM_C.ensembleAveraging",h3.ensemble_averaging,chains, "Pall", 1000, 0)

def test_time_ensAveragePk(h1, h2, h3, chains):
    timecall( "HMM.ensembleAveraging", h1.ensemble_averaging, chains, "Pk", 1000, 0)
    timecall( "HMM_F.ensembleAveraging",h2.ensemble_averaging,chains, "Pk", 1000, 0)
    timecall( "HMM_C.ensembleAveraging",h3.ensemble_averaging,chains, "Pk", 1000, 0)

def test_time_ensAverageUnit(h1, h2, h3, chains):
    timecall( "HMM.ensembleAveraging", h1.ensemble_averaging, chains, "unit", 1000, 0)
    timecall( "HMM_F.ensembleAveraging",h2.ensemble_averaging,chains, "unit", 1000, 0)
    timecall( "HMM_C.ensembleAveraging",h3.ensemble_averaging,chains, "unit", 1000, 0)

def 

def test_time_trainingStates(h1, initial, chains, observation, states, setStates):
    setProba(initial, h1)
    timecall( "ensembleAveragingPall",h1.ensemble_averaging, chains,setStates, "Pall",    1000, 0)
    setProba(initial, h1)
    timecall( "ensembleAveragingPk  ",h1.ensemble_averaging,chains, setStates,                                                 "Pk", 1000, 0)
    setProba(initial, h1)
    timecall( "ensembleAveragingUnit",h1.ensemble_averaging, chains, setStates,"unit", 1000, 0)
    setProba(initial, h1)
#    timecall( "learn                ",h1.learn, observation, states, 1000, 0)
    setProba(initial, h1)
#    timecall( "multiple_learn       ",h1.multiple_learn,chains, setStates,1000, 0)


if __name__ == "__main__":
    
    S = range(10)
    O = range(10)
    test = hmm.HMM( S, O )
    chains = []

    simul = test.simulate(200,1)
    observation = []
    state = []
    for couple in simul:
        observation.append(couple[1])
        state.append(couple[0])

    setObservation = []
    setState = []
    o = []
    s = []
    for i in xrange(20):
        chains.append(test.simulate(20, 1))
        for couple in chains[i]:
            o.append(couple[1])
            s.append(couple[0])
        setObservation.append(o)
        setState.append(s)
        o = []
        s = []

    initial = hmm.HMM( S, O)
    initial.setRandomProba()
    test1 = hmm.HMM( S, O)
    test2 = hmmf.HMM_F( S, O)
    test3 = hmmc.HMM_C( S, O)
    hmmS = hmmS.HMMS(S, O)
    setProba(initial, test1)
    setProba(initial, test2)
    setProba(initial, test3)
    print "\n     ----------------- AlphaScaled -----------------------"
#    test_time_alpha(test1, test2, test3, observation)
    print "\n     ----------------- BetaScaled  -----------------------"
#    test_time_beta(test1, test2, test3, observation)
    print "\n     -----------------     Ksi     -----------------------"
#    test_time_ksi(test1, test2, test3, observation)
    print "\n     ----------------- UpdateIterB -----------------------"
#    test_time_UpdateIterB(test1, test2, test3, observation)
    print "\n     -----------------  CorrectM   -----------------------"
#    test_time_CorrectM(test1, test2, test3)
    print "\n     ----------------- NormalizeB  -----------------------"
#    test_time_NormalizeB(test1, test2, test3)
    print "\n     -----------------   analyze   -----------------------"
#    test_time_analyze(test1, test2, test3, observation)
    print "\n     ----------------- analyze_log -----------------------"
#    test_time_analyze_log(test1, test2, test3, observation)
    print "\n     -----------------    learn    -----------------------"
    setProba(initial, test1)
    setProba(initial, test2)
    setProba(initial, test3)
#    test_time_learn(test1, test2, test3, observation)
    print "\n     -----------------multiple_learn-----------------------"
    setProba(initial, test1)
    setProba(initial, test2)
    setProba(initial, test3)
#    test_time_multiple_learn(test1, test2, test3, setObservation)
    print "\n     -------------EnsembleAveragingPall--------------------"
    setProba(initial, test1)
    setProba(initial, test2)
    setProba(initial, test3)
#    test_time_ensAveragePall(test1, test2, test3, setObservation)
    print "\n     -------------EnsembleAveragingPk----------------------"
    setProba(initial, test1)
    setProba(initial, test2)
    setProba(initial, test3)
#    test_time_ensAveragePk(test1, test2, test3, setObservation)
    print "\n     -------------EnsembleAveragingUnit--------------------"
    setProba(initial, test1)
    setProba(initial, test2)
    setProba(initial, test3)
#    test_time_ensAverageUnit(test1, test2, test3, setObservation)
    print "\n     -----------------------class HMMS-------------------------"
#    test_time_trainingStates(hmmS, initial, setObservation, observation, state, setState)


