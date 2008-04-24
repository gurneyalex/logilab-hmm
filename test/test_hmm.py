import unittest
import os
from numpy import array, allclose, zeros, take
import logilab.hmm.hmm as hmm
import logilab.hmm.hmmc as hmmc
import logilab.hmm.hmmf as hmmf
import logilab.hmm.hmmS as hmmS

class TestInitProba(unittest.TestCase):
    """ test random functions"""
    hmmKlass = hmm.HMM

    def setUp(self):
        self.hmm = self.hmmKlass(['a', 'b'], ['1', '2'], None,
                                None, None)
    def test_random_1_1(self):
        self.hmm.set_random_transition_proba()
        self.hmm.checkHMM()

    def test_random_1_2(self):
        self.hmm.set_random_observation_proba()
        self.hmm.checkHMM()

    def test_random_1_3(self):
        self.hmm.set_random_initial_proba()
        self.hmm.checkHMM()


    def test_random_1(self):
        self.hmm.set_random_proba()
        self.hmm.checkHMM()

    def testr_andom_2(self):
        self.hmm.reset_transition_proba()
        self.failUnless(allclose(self.hmm.A, zeros((2, 2), float)))
        self.hmm.set_transition_proba('a', 'b', 1 )
        self.hmm.set_transition_proba('b', 'a', 1 )
        self.hmm.checkHMM()

    def test_random_3(self):
        self.hmm.reset_observation_proba()
        self.failUnless(allclose(self.hmm.B, zeros((2, 2), float)))
        self.hmm.set_observation_proba('a', '1', 1)
        self.hmm.set_observation_proba('b', '1', 1)
        self.hmm.checkHMM()

    def test_random_4(self):
        self.hmm.reset_initial_proba()
        self.failUnless(allclose(self.hmm.pi, zeros(2, float)))
        self.hmm.set_initial_proba('a', 0.5)
        self.hmm.set_initial_proba('b' , 0.5)
        self.hmm.checkHMM()

class TestInitProbaC(TestInitProba):
    hmmKlass = hmmc.HMM_C

class TestInitProbaF(TestInitProba):
    hmmKlass = hmmf.HMM_F

class TestFunctions(unittest.TestCase):
    
    hmmKlass = hmm.HMM

    def setUp(self):
        self.observations = ['1', '2', '3']
        self.hmm1 = self.hmmKlass(['a', 'b'], ['1', '2'],
                        array([[0.2, 0.8],
                               [0.3, 0.7]]),
                        array( [[1., 0.2],
                                [0., 0.8]] ),
                        array([0.3, 0.7]))
        self.hmm2 = self.hmmKlass(['a', 'b'], ['1', '2', '3'],
                        array([[1.0, 0.0],
                               [0.0, 1.0]]),
                        array([[0.5, 0.0],
                               [ .5,  .5],
                               [0.0, 0.5]]),
                        array([0.5, 0.5]))
        self.hmm3 = self.hmmKlass(['a', 'b'], ['1', '2'],
                        array([[0., 1.],
                               [0., 1.]]),
                        array( [[1., 0.],
                                [0., 1.]] ),
                        array([0.5, 0.5]))
        self.hmm4 = self.hmmKlass(['a', 'b'], ['1', '2', '3'])
    
    def test_simulate(self):
        chain = self.hmm1.simulate(3)
        self.assertEquals(len(chain), 3)
        ens = set(chain)
        while ens != set():
            elt = ens.pop()
            self.failUnless( elt in self.hmm1.omega_O)

    def test_get_observation_indices(self):
        result = array([0, 1, 2], int)
        obs_ind = self.hmm2. _get_observationIndices(self.observations)
        self.failUnless(allclose(obs_ind, result))
    
    def test_normalize_1(self):
        res_a = array([[0.7, 0.3], [0.8, 0.2]])
        res_b = array([[0.2, 1.], [0.8, 0.]])
        res_pi = array([0.7, 0.3])
        A, B = zeros( (self.hmm1.N, self.hmm1.N), float)
        PI = zeros( self.hmm1.N)
        A, B, PI = self.hmm1.normalize()
        self.failUnless(allclose(A, res_a))
        self.failUnless(allclose(B, res_b))
        self.failUnless(allclose(PI, res_pi))
        
    def test_normalize_2(self):
        res_a = array([[0.7, 0.3], [0.8, 0.2]])
        res_b = array([[0.2, 1.], [0.8, 0.]])
        res_pi = array([0.7, 0.3])
        A, B = zeros( (self.hmm1.N, self.hmm1.N), float)
        PI = zeros( self.hmm1.N)
        P = array([1, 0])
        A, B, PI = self.hmm1.normalize(P)
        self.failUnless(allclose(A, res_a))
        self.failUnless(allclose(B, res_b))
        self.failUnless(allclose(PI, res_pi))

    def test_correctm_1( self):
        M = array([[1., 0.], [0., 0.]])
        k = 1
        p = 0.5
        result = array([[1., 0.], [0.5, 0.5]])
        MM = self.hmm3.correct_M(M, k, p)
        self.failUnless( allclose(result, MM))

    def test_correctm_2( self):
        M = array([[0., 0.], [0., 0.]])
        k = 1
        p = 0.5
        result = array([[0.5, 0.5], [0.5, 0.5]])
        MM = self.hmm3.correct_M(M, k, p)
        self.failUnless( allclose(result, MM))

    def test_alpha_scaled_1(self):
        obs_indices = [0, 1]
        res_alpha = array([[1, 0], [0, 1]])
        res_fact = array([2., 1.])
        Bo = take(self.hmm3.B, obs_indices, 0)
        alpha, fact = self.hmm3.alpha_scaled(self.hmm3.A, Bo, self.hmm3.pi )
        self.failUnless( allclose(alpha, res_alpha))
        self.failUnless( allclose(fact, res_fact))

    def test_alpha_scaled_2(self):
        obs_indices = [0, 1, 2]
        res_alpha = array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        res_fact = array([3., 3., 3.])
        Bo = take(self.hmm4.B, obs_indices, 0)
        alpha, fact = self.hmm4.alpha_scaled(self.hmm4.A, Bo, self.hmm4.pi )
        self.failUnless( allclose(alpha, res_alpha))
        self.failUnless( allclose(fact, res_fact))

    def test_beta_scaled_1(self):
        obs_indices = [0, 1, 1]
        fact = array([2., 1., 1.])
        res_beta = array([[2., 2.], [1., 1.], [1., 1.]])
        Bo = take(self.hmm3.B, obs_indices, 0)
        beta = self.hmm3.beta_scaled(self.hmm3.A, Bo, fact )
        self.failUnless( allclose(beta, res_beta))

    def test_beta_scaled_2(self):
        obs_indices = [0, 1, 2]
        fact = array([3., 3., 3.])
        res_beta = array([[3., 3.], [3., 3.], [3., 3.]])
        Bo = take(self.hmm4.B, obs_indices, 0)
        beta = self.hmm4.beta_scaled(self.hmm4.A, Bo, fact )
        self.failUnless( allclose(beta, res_beta))

    def test_gamma_1(self):
        res = array([[1, 0], [0, 1], [0, 1]], float)
        obs_indices = [0, 1, 1]
        A = self.hmm3.A
        B = self.hmm3.B
        PI = self.hmm3.pi
        Bo = take(B, obs_indices, 0)
        alpha, scale_factors = self.hmm3.alpha_scaled(A, Bo, PI )
        beta = self.hmm3.beta_scaled( A, Bo, scale_factors )
        gamma = self.hmm3._gamma(alpha, beta, scale_factors)
        self.failUnless(allclose(gamma, res))

    def test_gamma_2(self):
        res = array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        obs_indices = [0, 1, 2]
        A = self.hmm4.A
        B = self.hmm4.B
        PI = self.hmm4.pi
        Bo = take(B, obs_indices, 0)
        alpha, scale_factors = self.hmm4.alpha_scaled(A, Bo, PI )
        beta = self.hmm4.beta_scaled( A, Bo, scale_factors )
        gamma = self.hmm4._gamma(alpha, beta, scale_factors)
        self.failUnless(allclose(gamma, res))

    def test_ksi_1(self):
        obs_indices = [0, 1, 1]
        Bo = take(self.hmm3.B, obs_indices, 0)
        alpha, scale_factors = self.hmm3.alpha_scaled(self.hmm3.A, 
                                                Bo, self.hmm3.pi)
        beta = self.hmm3.beta_scaled( self.hmm3.A, Bo, scale_factors )
        ksy = self.hmm3.ksi( self.hmm3.A, Bo, alpha, beta )
        res_ksi = array([ [[0., 1.], [0., 0.]], [[0., 0.], [0., 1.]]])
        self.failUnless( allclose(ksy, res_ksi))

    def test_ksi_2(self):
        obs_indices = [0, 1, 2]
        Bo = take(self.hmm4.B, obs_indices, 0)
        alpha, scale_factors = self.hmm4.alpha_scaled(self.hmm4.A, 
                                                Bo, self.hmm4.pi)
        beta = self.hmm4.beta_scaled( self.hmm4.A, Bo, scale_factors )
        ksy = self.hmm4.ksi( self.hmm4.A, Bo, alpha, beta )
        res_ksi = array([ [[0.25, 0.25], [0.25, 0.25]], 
                          [[0.25, 0.25], [0.25, 0.25]]])
        self.failUnless( allclose(ksy, res_ksi))

    def test_update_iter_gamma_1(self):
        gamma = array([[1., 0.], [0., 1.], [0., 1.]])
        sigma_gamma_A = zeros(2)
        sigma_gamma_B = zeros(2)
        sga = array([1., 1.])
        sgb = array([1., 2.])   
        self.hmm3._update_iter_gamma( gamma, sigma_gamma_A, sigma_gamma_B )
        self.failUnless( allclose(sga, sigma_gamma_A))
        self.failUnless( allclose(sgb, sigma_gamma_B))

    def test_update_iter_gamma_2(self):
        gamma = array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        sigma_gamma_A = zeros(2)
        sigma_gamma_B = zeros(2)
        sga = array([1., 1.])
        sgb = array([1.5, 1.5])   
        self.hmm4._update_iter_gamma( gamma, sigma_gamma_A, sigma_gamma_B )
        self.failUnless( allclose(sga, sigma_gamma_A))
        self.failUnless( allclose(sgb, sigma_gamma_B))

    def test_update_iterb_1(self):
        B = self.hmm3.B
        B_bar = B
        gamma = array([[1., 0.], [0., 1.], [0., 1.]])
        obs_indices = [0, 1, 1]
        self.hmm3.update_iter_B( gamma, obs_indices, B_bar )
        self.failUnless( allclose(B, B_bar))

    def test_update_iterb_2(self):
        B_bar = self.hmm4.B
        gamma = array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        obs_indices = [0, 1, 2]
        result = array([[5./6, 5./6], [5./6, 5./6], [5./6, 5./6]])       
        self.hmm4.update_iter_B( gamma, obs_indices, B_bar )
        self.failUnless( allclose(result, B_bar))

    def test_update_itera_1(self):
        ksi = array([ [[0., 1.], [0., 0.]], [[0., 0.], [0., 1.]] ])
        A_bar = zeros((2, 2))
        resA = array([[0., 1.], [0., 1.]])
        self.hmm3._update_iter_A( ksi, A_bar )
        self.failUnless( allclose(resA, A_bar))

    def test_update_itera_2(self):
        ksi = array([ [[0.25, 0.25], [0.25, 0.25]], [[0.25, 0.25], [0.25, 0.25]]])
        A_bar = zeros((2, 2))
        resA = array([[0.5, 0.5], [0.5, 0.5]])
        self.hmm4._update_iter_A( ksi, A_bar )
        self.failUnless( allclose(resA, A_bar))

    def test_normalize_itera_1(self):
        A_bar = array([[0., 1.], [0., 1.]])
        sga = array([1., 1.])
        result = A_bar
        self.hmm3._normalize_iter_A(A_bar, sga )
        self.failUnless( allclose(A_bar, result))
    
    def test_normalize_itera_2(self):
        A_bar = array([[0.5, 0.5], [0.5, 0.5]])
        sga = array([1., 1.])
        result = A_bar
        self.hmm4._normalize_iter_A(A_bar, sga )
        self.failUnless( allclose(A_bar, result))

    def test_normalizeb_1( self):
        B_bar = array([[1., 0.], [0., 2.]])
        sgb = array([1., 2.])
        result = array([[1., 0.], [0., 1.]])
        self.hmm4.normalize_B(B_bar, sgb)
        self.failUnless(allclose(B_bar, result))

    def test_normalizeb_2( self):
        B_bar = array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        sgb = array([1.5, 1.5])
        result = array([[1./3, 1./3], [1./3, 1./3], [1./3, 1./3]])
        self.hmm4.normalize_B(B_bar, sgb)
        self.failUnless(allclose(B_bar, result))

    def test_stop(self):
        self.hmm3._stop_condition(self.hmm3.A, self.hmm3.pi, self.hmm3.B)
     
    def test_final_step(self):
        obs_indices = [0, 1, 1]
        gamma = array([[1., 0.], [0., 1.], [0., 1]])
        ksi = array([[[0., 1.], [0., 0.]], [[0., 0.], [0., 1.]] ])
        Abar = array([[0., 1.], [0., 1.]])
        Bbar = array([[1., 0.], [0., 1.]])
        pibar = array([1., 0.])
        A , B, PI = self.hmm3._final_step( gamma, ksi, obs_indices )
        self.failUnless( allclose(Abar, A))
        self.failUnless( allclose(Bbar, B))
        self.failUnless( allclose(pibar, PI))

class TestFunctionsC (TestFunctions):
    hmmKlass = hmmc.HMM_C

class TestFunctionsF(TestFunctions):
    hmmKlass = hmmf.HMM_F

class TestWeightingFactor(unittest.TestCase):

    hmmKlass = hmm.HMM
    def setUp(self):
        self.hmm1 = self.hmmKlass(['a', 'b'], ['1', '2', '3'],
                        array([[0., 1.],
                               [1., 0.]]),
                        array([[0.5, 0.0],
                               [ .5,  .5],
                               [0.0, 0.5]]),
                        array([0.5, 0.5]))
        self.hmm2 = self.hmmKlass(['a', 'b'], ['1', '2'],
                        array([[0., 1.],
                               [0., 1.]]),
                        array( [[1., 0.],
                                [0., 1.]] ),
                        array([0.5, 0.5]))

    def test_Weighting_factor_Pall_1(self):
        set_obs = [['1', '2'], ['2', '2']]
        resP = 1./4
        P = self.hmm2._weighting_factor_Pall(set_obs)
        self.failUnless(P == resP)

    def test_Weighting_factor_Pall_2(self):
        set_obs = [['1', '3'], ['1', '2'], ['2', '2']]
        resP = 1./256
        P = self.hmm1._weighting_factor_Pall(set_obs)
        self.failUnless(P == resP)

    def test_Weighting_factor_Pk_1(self):
        obs = ['1', '2']
        resP = 1./2
        P = self.hmm2._weighting_factor_Pk(obs)
        self.failUnless(P == resP)

    def test_Weighting_factor_Pk_2(self):
        obs = ['1', '3']
        resP = 1./8
        P = self.hmm1._weighting_factor_Pk(obs)
        self.failUnless(P == resP)

class TestWeightingFactorC(TestWeightingFactor):
    hmmKlass = hmmc.HMM_C

class TestWeightingFactorF(TestWeightingFactor):
    hmmKlass = hmmf.HMM_F

class TestStates(unittest.TestCase):
    hmmKlass = hmmS.HMMS

    def setUp(self):
        self.hmm1 = hmmS.HMMS(['a', 'b', 'c'], ['1', '2', '3'])
        self.hmm2 = hmmS.HMMS(['a', 'b'], ['1', '2'])
        self.aHMM_1 = hmmS.HMMS( ['a', 'b'], ['1', '2', '3'], 
                            array([[0.7, 0.3],[0.2, 0.8]]),
                            array([[0.2, 0.4], [0.6, 0.2], [0.2, 0.4]]),
                            array([0.2, 0.8]))

    def test_learn_A_1(self):
        states = ['a','a','b','a','c','b','c','a','b','a','c','b','a']
        result = array([[0.2, 0.4, 0.4], [0.75, 0, 0.25], [1./3, 2./3, 0]])
        self.hmm1._learn_A(states)
        self.failUnless( allclose(result, self.hmm1.A))

    def test_learnA_2(self):
        states = ['a', 'b', 'a', 'a', 'a', 'b', 'a']
        result = array([[0.5, 0.5, 0.], [1., 0., 0.], [0., 0., 1.]])
        self.hmm1._learn_A(states)
        self.failUnless( allclose(result, self.hmm1.A))
    
    def test_multiple_learnA(self):
        states = [['a', 'b'] * 3, ['b', 'a'] * 2]
        result = array([[0., 1.], [1., 0.]])
        self.hmm2._multiple_learn_A(states)
        self.failUnless( allclose(result, self.hmm2.A))

    def test_baumwelch(self):
        chain = ['1', '1'] * 4
        states = ['b', 'a'] + ['a', 'a'] * 3
        resA = array([[1., 0.], [1., 0.]])
        resB = array([[1., 1.], [0., 0.]])
        resPI = array([0.5, 0.5])
        self.hmm2.learn(chain, states)
        self.hmm2.checkHMM()
        self.failUnless( allclose(resA, self.hmm2.A))
        self.failUnless( allclose(resB, self.hmm2.B))
        self.failUnless( allclose(resPI, self.hmm2.pi))

    def test_multiple_learn_1(self):
        chains = [ ['1','2','2','2','2'], ['1','2','2','2','2','2','2'],
                    ['2','2','2','2','2','2','2']]
        states = [['a'] + ['b'] * 4, ['a'] + ['b'] * 4, ['b'] * 5]
        self.hmm2.multiple_learn(chains, states)
        self.hmm2.checkHMM()

    def test_ens_average_1(self):
        chains = [ ['1'] + ['2'] * 4, ['1'] + ['2'] * 4, ['2'] * 5]
        states = [ ['a'] + ['b'] * 4, ['a'] + ['b'] * 4, ['b'] * 5 ]
        self.hmm2.ensemble_averaging(chains, states, "unit", 1000, 0)
        self.hmm2.checkHMM()

    def test_ens_average_2(self):
        chains = [['1', '1'] * 4] * 5
        states = [['b', 'a'] + ['a', 'a'] * 3] * 5
        resA = array([[1., 0.], [1., 0.]])
        resB = array([[1., 1.], [0., 0.]])
        resPI = array([0.5, 0.5])
        self.hmm2.ensemble_averaging(chains, states, "unit" , 1000, 0)
        self.hmm2.checkHMM()
        self.failUnless( allclose(resA, self.hmm2.A))
        self.failUnless( allclose(resB, self.hmm2.B))
        self.failUnless( allclose(resPI, self.hmm2.pi))

class TestStatesC(TestStates):
    hmmKlass = hmmS.HMMS_C

class TestStatesF(TestStates):
    hmmKlass = hmmS.HMMS_F

class TestDeterministic(unittest.TestCase):
    """Test the viterbi algorithm on a deterministic chain"""

    hmmKlass = hmm.HMM

    def setUp(self):
        self.hmm = self.hmmKlass(['a', 'b'], ['1', '2', '3'],
                        array([[1.0, 0.0],
                               [0.0, 1.0]]),
                        array([[0.5, 0.0],
                               [ .5,  .5],
                               [0.0, 0.5]]),
                        array([0.5, 0.5]))

    def _analyze_chain(self, mc, chain, result=None):
        an1 = mc.analyze(chain)
        an2 = mc.analyze_log(chain)
        self.assertEqual(an1, an2)
        if result:
            self.assertEqual(an1, result)

    def test_viterbi_1(self):
        """Test the chain (1,1,1) [determ]"""
        chain = ['1'] * 3
        result = ['a', 'a', 'a']
        self._analyze_chain(self.hmm, chain, result)
		
    def test_viterbi_2(self):
        """test the chain (2,1,1,1,1,1) [determ]"""
        result = ['a', 'a', 'a', 'a', 'a', 'a']
        chain = ['2'] + ['1'] * 5
        self._analyze_chain( self.hmm, chain, result)

    def test_viterbi_3(self):
        """test the chain (3,2,2,2,2,2) [determ]"""
        chain = ['3'] + ['2'] * 5
        result = ['b', 'b', 'b', 'b', 'b', 'b']
        self._analyze_chain( self.hmm, chain, result)

    def test_viterbi_4(self):
        """test the chain (2,2,3,3,3,2) [determ]"""
        chain = ['2', '2', '3', '3', '3', '2']
        result = ['b', 'b', 'b', 'b', 'b', 'b']
        self._analyze_chain( self.hmm, chain, result)

    def test_viterbi_5(self):
        """test the chain (2,2,2,2,2,3) [determ]"""
        chain = ['2', '2', '2', '2', '2', '3']
        result = ['b', 'b', 'b', 'b', 'b', 'b']
        self._analyze_chain( self.hmm, chain, result)


class TestDeterministicHmmC(TestDeterministic):
    hmmKlass = hmmc.HMM_C

class TestDeterministicHmmF(TestDeterministic):
    hmmKlass = hmmf.HMM_F

class test_baumwelch(unittest.TestCase):
    """Test the Baumwelch algorithm"""

    def setUp(self):
        self.aHMM = hmm.HMM( ['a', 'b'], ['1', '2', '3'])
        self.aHMMC = hmmc.HMM_C( ['a', 'b'], ['1', '2', '3'])
        self.aHMMF = hmmf.HMM_F( ['a', 'b'], ['1', '2', '3'])
        
        self.aHMM_1 = hmm.HMM( ['a', 'b'], ['1', '2', '3'], 
                            array([[0.7, 0.3], [0.2, 0.8]]),
                            array([[0.2, 0.4], [0.6, 0.2], [0.2, 0.4]]),
                            array([0.2, 0.8]))
        self.aHMM_C = hmmc.HMM_C( ['a', 'b'], ['1', '2', '3'], 
                            array([[0.7, 0.3], [0.2, 0.8]]),
                            array([[0.2, 0.4], [0.6, 0.2], [0.2, 0.4]]),
                            array([0.2, 0.8]))
        self.aHMM_F = hmmf.HMM_F( ['a', 'b'], ['1', '2', '3'], 
                            array([[0.7, 0.3], [0.2, 0.8]]),
                            array([[0.2, 0.4], [0.6, 0.2], [0.2, 0.4]]),
                            array([0.2, 0.8]))
        self.det = hmm.HMM(['a'], ['1', '2'])
        self.test = hmm.HMM( range(5), range(5) )
        self.det2 = hmm.HMM(['a', 'b'], ['1', '2'] )

    def _learn_compare(self, chain):      
        self.aHMM.learn(chain)
        self.aHMMC.learn(chain)
        self.aHMMF.learn(chain)
        
        self.failUnless(allclose(self.aHMMC.A, self.aHMM.A))
        self.failUnless(allclose(self.aHMMF.A, self.aHMM.A))
        self.failUnless(allclose(self.aHMMC.B, self.aHMM.B))
        self.failUnless(allclose(self.aHMMF.B, self.aHMM.B))
        self.failUnless(allclose(self.aHMMC.pi, self.aHMM.pi))
        self.failUnless(allclose(self.aHMMF.pi, self.aHMM.pi))

    def test_update_iterb(self):
        B_bar = array([[ 0., 0.], [ 0., 0.], [ 0., 0.]])
        B_barF = array([[ 0., 0.], [ 0., 0.], [ 0., 0.]])
        B_barC = array([[ 0., 0.], [ 0., 0.], [ 0., 0.]])
        gamma = array([[0.17584567, 0.82415433], [0.43775031, 0.56224969],
                       [0.43195352, 0.56804648], [0.44859571, 0.55140429],
                       [0.43240921, 0.56759079], [0.44861501, 0.55138499],
                       [0.43241002, 0.56758998], [0.448615, 0.551385  ],
                       [0.43240908, 0.56759092], [0.44859262, 0.55140738],
                       [0.43188047, 0.56811953], [0.43601172, 0.56398828],
                       [0.13479001, 0.86520999], [0.13445915, 0.86554085],
                       [0.41918731, 0.58081269], [0.44750776, 0.55249224],
                       [0.41943579, 0.58056421], [0.14038371, 0.85961629],
                       [0.41931846, 0.58068154], [0.44469141, 0.55530859]])

        obs_indices = [0,1,0,1,0,1,2,1,2,1,0,1,0,1,0,1,2,1,0,2]       
        self.aHMM.update_iter_B( gamma, obs_indices, B_bar )
        self.aHMMC.update_iter_B( gamma, obs_indices, B_barC )
        self.aHMMF.update_iter_B( gamma, obs_indices, B_barF )
        self.failUnless( allclose(B_bar, B_barC))
        self.failUnless( allclose(B_bar, B_barF))
   
    def test_baumwelch_1(self):
        """test the observations (1,2,1,2,1,2,1,2,1,2) """
        chain = ['1', '2'] * 5 
        self._learn_compare(chain)

    def test_baumwelch_2(self):
        """test the observations (1,1,1,1,1,2,2,2,2,2) """
        chain =  ['1'] * 5 + ['2'] * 5
        self._learn_compare(chain)

    def test_baumwelch_3(self):
        """test the observations (3,3,3,3,3,3,3,3,3,1) """
        chain = ['3'] * 9 + ['1']
        self._learn_compare(chain)

    def test_baumwelch_4(self):
        """test the observations (1,2,1,2,1,2,1,2,1,2) """
        chain = ['1', '2'] * 5 
        self._learn_compare(chain)

    def test_baumwelch_6(self):
        chain = ['2'] * 2
        resA = self.det.A
        resB = array([[0.], [1.]])
        respi = self.det.pi
        self.det.learn(chain)
        self.failUnless( allclose(resA, self.det.A))
        self.failUnless( allclose(resB, self.det.B))
        self.failUnless( allclose(respi, self.det.pi))

    def test_baumwelch_7(self):
        observation = self.test.simulate(10)
        self.test.set_random_proba()
        self.test.learn(observation)
        self.test.checkHMM()

    def test_multiple_learn_1(self):
        chains = []
        for i in xrange(10):
            chains.append(self.aHMM.simulate(10))
        A = self.aHMM.A
        B = self.aHMM.B
        PI = self.aHMM.pi
        self.aHMM.multiple_learn(chains)
        self.aHMM.checkHMM()
        self.failUnless( allclose(self.aHMM.A, A))
        self.failUnless( allclose(self.aHMM.B, B))
        self.failUnless( allclose(self.aHMM.pi, PI))

    def test_multiple_learn_2(self):
        chains = [ ['1','2','2','2','2'], ['1','2','2','2','2','2','2'],
                    ['2','2','2','2','2','2','2']]
        self.aHMM_1.multiple_learn(chains)
        self.aHMM_1.checkHMM()
        self.aHMM_C.multiple_learn(chains)
        self.aHMM_C.checkHMM()
        self.aHMM_F.multiple_learn(chains)
        self.aHMM_F.checkHMM()
        self.failUnless( allclose(self.aHMM_1.A, self.aHMM_C.A))
        self.failUnless( allclose(self.aHMM_1.A, self.aHMM_F.A))
        self.failUnless( allclose(self.aHMM_1.B, self.aHMM_C.B))
        self.failUnless( allclose(self.aHMM_1.B, self.aHMM_F.B))
        self.failUnless( allclose(self.aHMM_1.pi, self.aHMM_C.pi))
        self.failUnless( allclose(self.aHMM_1.pi, self.aHMM_F.pi))

    def test_multiple_learn_3(self):
        chains = [ ['1','2','2','2','2'], ['1','2','2','2','2','2','2'],
                    ['2','2','2','2','2','2','2']]
        self.aHMM_1.multiple_learn(chains)
        self.aHMM_1.checkHMM()
        self.aHMM_C.multiple_learn(chains)
        self.aHMM_C.checkHMM()
        self.aHMM_F.multiple_learn(chains)
        self.aHMM_F.checkHMM()
        self.failUnless( allclose(self.aHMM_1.A, self.aHMM_C.A))
        self.failUnless( allclose(self.aHMM_1.A, self.aHMM_F.A))
        self.failUnless( allclose(self.aHMM_1.B, self.aHMM_C.B))
        self.failUnless( allclose(self.aHMM_1.B, self.aHMM_F.B))
        self.failUnless( allclose(self.aHMM_1.pi, self.aHMM_C.pi))
        self.failUnless( allclose(self.aHMM_1.pi, self.aHMM_F.pi))

    def test_multiple_learn_4(self):
        chains = [ ['2'] * 2, ['2'] * 3, ['2'] * 4]
        resA = self.det.A
        resB = array([[0.], [1.]])
        respi = self.det.pi
        self.det.multiple_learn(chains)
        self.failUnless( allclose(resA, self.det.A))
        self.failUnless( allclose(resB, self.det.B))
        self.failUnless( allclose(respi, self.det.pi))

    def test_multiple_learn_5(self):
        chains = []
        for i in xrange(10):
            chains.append(self.test.simulate(20))
        self.test.set_random_proba()
        self.test.multiple_learn(chains)
        self.test.checkHMM()

class TestEnsembleAveraging(unittest.TestCase):
    def setUp(self):
        self.det = hmm.HMM(['a'], ['1', '2'])
        self.test = hmm.HMM( ['a', 'b'], ['1', '2'] )
        self.gen = hmm.HMM( ['a', 'b'], ['1', '2'],
                            array([[0.7, 0.3], [0.2, 0.8]]),
                            array([[0.2, 0.6], [0.8, 0.4]]),
                            array([0.5, 0.5]))
        self.aHMM = hmm.HMM(['a', 'b'], ['1', '2'])

    def test_ens_average_1(self):
        set_observations = [ ['2'] * 2, ['2'] * 3, ['2'] * 4]
        resA = self.det.A
        resB = array([[0.], [1.]])
        respi = self.det.pi
        self.det.ensemble_averaging(set_observations, "unit", 1000, 0)
        self.failUnless( allclose(resA, self.det.A))
        self.failUnless( allclose(resB, self.det.B))
        self.failUnless( allclose(respi, self.det.pi))

    def test_ens_average_2(self):
        chains = []
        for i in xrange(10):
            chains.append(self.gen.simulate(10))
        self.test.ensemble_averaging(chains, "unit", 1000, 1)
        self.test.checkHMM()

    def test_ens_average_3(self):
        chains = [ ['1', '2', '2', '2', '2'], ['1', '2', '2', '2', '2'],
                        ['2','2','2','2','2','2','2']]
        self.aHMM.ensemble_averaging(chains, "unit", 1000, 0)
        self.aHMM.checkHMM()

    def test_ens_average_4(self):
        set_observations = [ ['2'] * 2, ['2'] * 3, ['2'] * 4]
        resA = self.det.A
        resB = array([[0.], [1.]])
        respi = self.det.pi
        self.det.ensemble_averaging(set_observations, "Pall", 1000, 0)
        self.failUnless( allclose(resA, self.det.A))
        self.failUnless( allclose(resB, self.det.B))
        self.failUnless( allclose(respi, self.det.pi))

    def test_ens_average_5(self):
        set_observations = [ ['2'] * 2, ['2'] * 3, ['2'] * 4]
        resA = self.det.A
        resB = array([[0.], [1.]])
        respi = self.det.pi
        self.det.ensemble_averaging(set_observations, "Pk", 1000, 0)
        self.failUnless( allclose(resA, self.det.A))
        self.failUnless( allclose(resB, self.det.B))
        self.failUnless( allclose(respi, self.det.pi))

class TestPickle(unittest.TestCase):
    """ test the pickle implementation """
    
    hmmKlass = hmm.HMM

    def setUp(self):
        self.hmm1 = self.hmmKlass( ['a', 'b'], ['1', '2', '3'])
        self.hmm2 = self.hmmKlass( ['a', 'b'], ['1', '2', '3'])
        self.hmm2.set_random_proba()
        self.hmm3 = self.hmmKlass( ['a'], ['1'])
        self.hmm4 = self.hmmKlass( [], [])

    def _compare(self, h1, h2, SaveState=None):
        self.failUnless( allclose(h1.A, h2.A))
        self.failUnless( allclose(h1.B, h2.B))
        self.failUnless( allclose(h1.pi, h2.pi))
        self.failUnless( allclose(h1.N, h2.N))
        self.failUnless( allclose(h1.M, h2.M))
        if SaveState:
            self.failUnless( h1.X_index == h2.X_index)  
            self.failUnless( h1.omega_X == h2.omega_X)  
            self.failUnless( h1.omega_O == h2.omega_O)  
            self.failUnless( h1.O_index == h2.O_index)      

    def test_pickle_1(self):
        f = open('save.data', 'w')
        self.hmm1.saveHMM(f)
        f.close()
        f = open('save.data', 'r')
        self.hmm2.loadHMM(f)
        f.close()
        self.hmm2.checkHMM()
        self._compare(self.hmm1, self.hmm2)
        os.remove('save.data')


    def test_pickle_2(self):
        f = open('save.data', 'w')
        self.hmm1.saveHMM(f, 1)
        f.close()
        f = open('save.data', 'r')
        self.hmm3.loadHMM(f)
        f.close()
        self.hmm3.checkHMM()
        self._compare(self.hmm1, self.hmm3)
        os.remove('save.data')

    def test_pickle_3(self):
        f = open('save.data', 'w')
        self.hmm1.saveHMM(f, 1)
        f.close()
        f = open('save.data', 'r')
        self.hmm4.loadHMM(f)
        f.close()
        self.hmm4.checkHMM()
        self._compare(self.hmm1, self.hmm4)
        os.remove('save.data')

class TestPickleHMMC(TestPickle):
    hmmKlass = hmmc.HMM_C

class TestPickleHMMF(TestPickle):
    hmmKlass = hmmf.HMM_F


#suite= unittest.makeSuite(TestLikelihood,'test')



if __name__ == '__main__':
    unittest.main()
