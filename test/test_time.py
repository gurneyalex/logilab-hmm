from numpy import zeros, ones, take
import logilab.hmm.hmm as hmm
import logilab.hmm.hmmc as hmmc
import logilab.hmm.hmmf as hmmf
import logilab.hmm.hmmS as hmmS
from support import timecall
from support import timecall_one

def setProba(hmm, hmm1):
    hmm.A = hmm1.A
    hmm.B = hmm1.B
    hmm.pi = hmm1.pi

def test_time_alpha(hmm1, hmm2, hmm3, observation):
    bo = take( hmm1.B, observation, 0 )
    timecall( "HMM.alpha_scaled    ", hmm1.alpha_scaled, hmm1.A, bo, hmm1.pi )
    timecall( "HMM_F.alpha_scaled  ", hmm2.alpha_scaled, hmm2.A, bo, hmm2.pi )
    timecall( "HMM_C.alpha_scaled  ", hmm3.alpha_scaled, hmm3.A, bo, hmm3.pi )

def test_time_beta(hmm1, hmm2, hmm3, observation):
    Bo = take( hmm1.B, observation, 0 )
    alpha, scale_factors = hmm3.alpha_scaled( hmm3.A, Bo, hmm3.pi )
    timecall( "HMM.beta_scaled     ", hmm1.beta_scaled, hmm1.A, Bo, scale_factors )
    timecall( "HMM_F.beta_scaled   ", hmm2.beta_scaled, hmm2.A, Bo, scale_factors )
    timecall( "HMM_C.beta_scaled   ", hmm3.beta_scaled, hmm3.A, Bo, scale_factors )

def test_time_ksi(hmm1, hmm2, hmm3, observation):
    Bo = take( hmm1.B, observation, 0 )
    alpha, scale_factors = hmm3.alpha_scaled( hmm3.A, Bo, hmm3.pi )
    beta = hmm3.beta_scaled(hmm3.A, Bo, scale_factors)
    timecall( "HMM.ksi            ", hmm1.ksi, hmm1.A, Bo, alpha, beta )
    timecall( "HMM_F.ksi          ", hmm2.ksi, hmm2.A, Bo, alpha, beta )
    timecall( "HMM_C.ksi          ", hmm3.ksi, hmm3.A, Bo, alpha, beta )

def test_time_update_iter_B(hmm1, hmm2, hmm3, observation):
    Bo = take( hmm1.B, observation, 0 )
    obs_ind = hmm3._get_observationIndices(observation)
    alpha, scale_factors = hmm3.alpha_scaled( hmm3.A, Bo, hmm3.pi )
    beta = hmm3.beta_scaled(hmm3.A, Bo, scale_factors)
    gamma = hmm3._gamma(alpha, beta, scale_factors)
    B1 = zeros( (hmm1.M, hmm1.N), float, order=hmm1.ORDER )
    B2 = zeros( (hmm2.M, hmm2.N), float, order=hmm2.ORDER )
    B3 = zeros( (hmm3.M, hmm3.N), float, order=hmm3.ORDER )
    timecall( "HMM.update_iter_B    ", hmm1.update_iter_B, gamma, obs_ind, B1)
    timecall( "HMM_F.update_iter_B  ", hmm2.update_iter_B, gamma, obs_ind, B2)
    timecall( "HMM_C.update_iter_B  ", hmm3.update_iter_B, gamma, obs_ind, B3)

def test_time_correct_M(hmm1, hmm2, hmm3):
    M = zeros((hmm1.N, hmm1.N))
    k = 1
    p = 1./hmm1.N
    timecall( "HMM.correct_M       ", hmm1.correct_M, M, k, p )
    timecall( "HMM_F.correct_M     ", hmm2.correct_M, M, k, p)
    timecall( "HMM_C.CorretM      ", hmm3.correct_M, M, k, p)

def test_time_normalize_B(hmm1, hmm2, hmm3):
    SGB =  2 * ones(hmm1.N)
    B_bar = ones( (hmm1.M, hmm1.N))
    timecall( "HMM.normalize_B     ", hmm1.normalize_B, B_bar, SGB)
    timecall( "HMM_F.normalize_B   ", hmm2.normalize_B, B_bar, SGB)
    timecall( "HMM_C.normalize_B   ", hmm3.normalize_B, B_bar, SGB)

def test_time_analyze(hmm1, hmm2, hmm3, obs):
    timecall( "HMM.analyze        ", hmm1.analyze, obs)
    timecall( "HMM_F.analyze      ", hmm2.analyze, obs)
    timecall( "HMM_C.analyze      ", hmm3.analyze, obs)

def test_time_analyze_log(hmm1, hmm2, hmm3, obs):
    timecall( "HMM.analyze_log    ", hmm1.analyze_log, obs)
    timecall( "HMM_F.analyze_log  ", hmm2.analyze_log, obs)
    timecall( "HMM_C.analyze_log  ", hmm3.analyze_log, obs)

def test_time_learn(hmm1, hmm2, hmm3, obs):
    timecall( "HMM.learn          ", hmm1.learn, obs, 1000, 0)
    timecall( "HMM_F.learn        ", hmm2.learn, obs, 1000, 0)
    timecall( "HMM_C.learn        ", hmm3.learn, obs, 1000, 0)

def test_time_multiple_learn(hmm1, hmm2, hmm3, chains):
    timecall( "HMM.multiple_learn  ", hmm1.multiple_learn, 
                                chains, 1000, 0)
    timecall( "HMM_F.multiple_learn", hmm2.multiple_learn, 
                                chains, 1000, 0)
    timecall( "HMM_C.multiple_learn", hmm3.multiple_learn,
                                chains, 1000, 0)

def test_time_ensAveragePall(hmm1, hmm2, hmm3, chains):
    timecall_one( "HMM.ensembleAveraging", hmm1.ensemble_averaging,
                                chains, "Pall", 1000, 0)
    timecall_one( "HMM_F.ensembleAveraging", hmm2.ensemble_averaging,
                                chains, "Pall", 1000, 0)
    timecall_one( "HMM_C.ensembleAveraging", hmm3.ensemble_averaging,
                                chains, "Pall", 1000, 0)

def test_time_ensAveragePk(hmm1, hmm2, hmm3, chains):
    timecall_one( "HMM.ensembleAveraging", hmm1.ensemble_averaging, 
                                chains, "Pk", 1000, 0)
    timecall_one( "HMM_F.ensembleAveraging", hmm2.ensemble_averaging,
                                chains, "Pk", 1000, 0)
    timecall_one( "HMM_C.ensembleAveraging", hmm3.ensemble_averaging,
                                chains, "Pk", 1000, 0)

def test_time_ensAverageUnit(hmm1, hmm2, hmm3, chains):
    timecall_one( "HMM.ensembleAveraging", hmm1.ensemble_averaging, 
                                chains, "unit", 1000, 0)
    timecall_one( "HMM_F.ensembleAveraging", hmm2.ensemble_averaging,
                                chains, "unit", 1000, 0)
    timecall_one( "HMM_C.ensembleAveraging", hmm3.ensemble_averaging,
                                chains, "unit", 1000, 0)

def test_one_run_unit_states(hmm1, initial, chains, observation, states, setStates):
    timecall_one( "ensembleAveragingUnit", hmm1.ensemble_averaging, 
                                chains, setStates,"unit", 120, 0)
def test_one_run_Pk_states(hmm1, initial, chains, observation, states, setStates):
    timecall_one( "ensembleAveragingPk  ", hmm1.ensemble_averaging,
                                chains, setStates,"Pk", 1000, 0)

def test_one_run_Pall_states(hmm1, initial, chains, observation,
                                states, setStates):
    timecall_one( "ensembleAveragingPall", hmm1.ensemble_averaging, 
                                chains, setStates, "Pall", 1000, 0)

def test_one_run_mult_learn_states(hmm1, initial, chains, observation, 
                                states, setStates):
    timecall_one( "multiple_learn       ", hmm1.multiple_learn, 
                                chains, setStates, 1000, 0)

if __name__ == "__main__":
    
    S = range(10)
    O = range(20)
    test = hmm.HMM( S, O )
    chains = []

    simul = test.simulate(100, 1)
    observation = []
    state = []
    for couple in simul:
        observation.append(couple[1])
        state.append(couple[0])

    setObservation = []
    setState = []
    o = []
    s = []
    for i in xrange(10):
        chains.append(test.simulate(10, 1))
        for couple in chains[i]:
            o.append(couple[1])
            s.append(couple[0])
        setObservation.append(o)
        setState.append(s)
        o = []
        s = []

    initial = hmm.HMM( S, O)
    initial.set_random_proba()
    test1 = hmm.HMM( S, O)
    test2 = hmmf.HMM_F( S, O)
    test3 = hmmc.HMM_C( S, O)
    hmmSC = hmmS.HMMS_C(S, O)
    hmms = hmmS.HMMS(S, O)
    setProba(test1, initial)
    setProba(test2, initial)
    setProba(test3, initial)
    print "\n     ----------------- alpha_scaled -----------------------"
#    test_time_alpha(test1, test2, test3, observation)
    print "\n     ----------------- beta_scaled  -----------------------"
#    test_time_beta(test1, test2, test3, observation)
    print "\n     -----------------     ksi     -----------------------"
#    test_time_ksi(test1, test2, test3, observation)
    print "\n     ----------------- update_iter_B -----------------------"
#    test_time_update_iter_B(test1, test2, test3, observation)
    print "\n     -----------------  correct_M   -----------------------"
#    test_time_correct_M(test1, test2, test3)
    print "\n     ----------------- normalize_B  -----------------------"
#    test_time_normalize_B(test1, test2, test3)
    print "\n     -----------------   analyze   -----------------------"
#    test_time_analyze(test1, test2, test3, observation)
    print "\n     ----------------- analyze_log -----------------------"
#    test_time_analyze_log(test1, test2, test3, observation)
    print "\n     -----------------    learn    -----------------------"
    setProba(test1, initial)
    setProba(test2, initial)
    setProba(test3, initial)
#    test_time_learn(test1, test2, test3, observation)
    print "\n     -----------------multiple_learn-----------------------"
    setProba(test1, initial)
    setProba(test2, initial)
    setProba(test3, initial)
#    test_time_multiple_learn(test1, test2, test3, setObservation)
    print "\n     -------------EnsembleAveragingPall--------------------"
    setProba(test1, initial)
    setProba(test2, initial)
    setProba(test3, initial)
#    test_time_ensAveragePall(test1, test2, test3, setObservation)
    print "\n     -------------EnsembleAveragingPk----------------------"
    setProba(test1, initial)
    setProba(test2, initial)
    setProba(test3, initial)
#    test_time_ensAveragePk(test1, test2, test3, setObservation)
    print "\n     -------------EnsembleAveragingUnit--------------------"
    setProba(test1, initial)
    setProba(test2, initial)
    setProba(test3, initial)
#    test_time_ensAverageUnit(test1, test2, test3, setObservation)
    print "\n     -----------------class HMMS_C-------------------------"
    setProba(hmmSC, test)
    test_one_run_Pk_states(hmmSC, initial, setObservation, 
                        observation, state, setState)
    setProba(hmmSC, test)
    test_one_run_unit_states(hmmSC, initial, setObservation, 
                        observation, state, setState)
    setProba(hmmSC, test)
    test_one_run_Pall_states(hmmSC, initial, setObservation, 
                        observation, state, setState)
    setProba(hmmSC, test)
    test_one_run_mult_learn_states(hmmSC, initial, setObservation, 
                        observation, state, setState)
    setProba(hmmSC, initial)
    test_one_run_unit_states(hmmSC, initial, setObservation, 
                        observation, state, setState)
