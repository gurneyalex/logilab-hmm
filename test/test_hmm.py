import unittest
import os
import autopath
from numpy import array, alltrue, allclose, zeros, take, isfortran
import logilab.hmm.hmm as hmm
import logilab.hmm.hmmc as hmmc
import logilab.hmm.hmmf as hmmf

#verbose=1

#def show_analysis(h,chain):
#    if verbose:
#        print "Chain      : ", chain
#     print "analyse    : ", h.analyze(chain)
#     print "analyse_log: ", h.analyze_log(chain)

class TestInitProba(unittest.TestCase):
    
    hmmKlass = hmm.HMM

    def setUp(self):
		self.hmm = self.hmmKlass(['a', 'b'], ['1', '2'], None,
                                None, None)
    def testRandom_1_1(self):
        self.hmm.setRandomTransitionProba()
        self.hmm.checkHMM()

    def testRandom_1_2(self):
        self.hmm.setRandomObservationProba()
        self.hmm.checkHMM()

    def testRandom_1_3(self):
        self.hmm.setRandomInitialProba()
        self.hmm.checkHMM()


    def testRandom_1(self):
        self.hmm.setRandomProba()
        self.hmm.checkHMM()

    def testRandom_2(self):
        self.hmm.resetTransitionProba()
        self.failUnless(allclose(self.hmm.A, zeros((2,2), float)))
        self.hmm.setTransitionProba('a', 'b', 1 )
        self.hmm.setTransitionProba('b', 'a', 1 )
        self.hmm.checkHMM()

    def testRandom_3(self):
        self.hmm.resetObservationProba()
        self.failUnless(allclose(self.hmm.B, zeros((2,2), float)))
        self.hmm.setObservationProba('a', '1', 1)
        self.hmm.setObservationProba('b', '1', 1)
        self.hmm.checkHMM()

    def testRandom_4(self):
        self.hmm.resetInitialProba()
        self.failUnless(allclose(self.hmm.pi, zeros(2, float)))
        self.hmm.setInitialProba('a', 0.5)
        self.hmm.setInitialProba('b',0.5)
        self.hmm.checkHMM()

class TestInitProbaC(TestInitProba):
    hmmKlass = hmmc.HMM_C

class TestInitProbaF(TestInitProba):
   hmmKlass = hmmf.HMM_F

class TestFunctions(unittest.TestCase):
    
    hmmKlass = hmm.HMM

    def setUp(self):
        self.observations = ['1','2','3']
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
        self.hmm4 = self.hmmKlass(['a', 'b'], ['1', '2','3'],None, None, None)
    
    def testSimulate(self):
        chain = self.hmm1.simulate(3)
        self.assertEquals(len(chain), 3)
        s = set(chain)
        while s != set():
            e = s.pop()
            self.failUnless( e in self.hmm1.omega_O)

    def testGetObservationIndices(self):
        result = array([0,1,2], int)
        obsInd = self.hmm2. _getObservationIndices(self.observations)
        self.failUnless(allclose(obsInd, result))
    
    def testNormalize_1(self):
        resA = array([[0.7,0.3],[0.8,0.2]])
        resB = array([[0.2, 1.],[0.8, 0.]])
        resPI = array([0.7, 0.3])
        A, B = zeros( (self.hmm1.N, self.hmm1.N), float)
        PI = zeros( self.hmm1.N)
        A, B, PI = self.hmm1.normalize()
        self.failUnless(allclose(A, resA))
        self.failUnless(allclose(B, resB))
        self.failUnless(allclose(PI, resPI))
        
    def testNormalize_2(self):
        resA = array([[0.7,0.3],[0.8,0.2]])
        resB = array([[0.2, 1.],[0.8, 0.]])
        resPI = array([0.7, 0.3])
        A, B = zeros( (self.hmm1.N, self.hmm1.N), float)
        PI = zeros( self.hmm1.N)
        P = array([1,0])
        A, B, PI = self.hmm1.normalize(P)
        self.failUnless(allclose(A, resA))
        self.failUnless(allclose(B, resB))
        self.failUnless(allclose(PI, resPI))

    def testCorrectM_1( self):
        M = array([[1., 0.],[0., 0.]])
        k = 1
        p = 0.5
        result = array([[1., 0.],[0.5, 0.5]])
        MM = self.hmm3.CorrectM(M,k,p)
        self.failUnless( allclose(result, MM))

    def testCorrectM_2( self):
        M = array([[0., 0.],[0., 0.]])
        k = 1
        p = 0.5
        result = array([[0.5, 0.5],[0.5, 0.5]])
        MM = self.hmm3.CorrectM(M,k,p)
        self.failUnless( allclose(result, MM))

    def testAlphaScaled_1(self):
        obsIndices = [0, 1]
        resAlpha = array([[1,0],[0,1]])
        resFact = array([2.,1.])
        Bo = take(self.hmm3.B, obsIndices, 0)
        alpha, fact = self.hmm3.AlphaScaled(self.hmm3.A, Bo, self.hmm3.pi )
        self.failUnless( allclose(alpha, resAlpha))
        self.failUnless( allclose(fact, resFact))

    def testAlphaScaled_2(self):
        obsIndices = [0, 1, 2]
        resAlpha = array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        resFact = array([3., 3., 3.])
        Bo = take(self.hmm4.B, obsIndices, 0)
        alpha, fact = self.hmm4.AlphaScaled(self.hmm4.A, Bo, self.hmm4.pi )
        self.failUnless( allclose(alpha, resAlpha))
        self.failUnless( allclose(fact, resFact))

    def testBetaScaled_1(self):
        obsIndices = [0, 1, 1]
        Alpha = array([[1.,0.],[0.,1.], [0., 1.]])
        fact = array([2.,1.,1.])
        resBeta = array([[2., 2.],[1., 1.], [1., 1.]])
        Bo = take(self.hmm3.B, obsIndices, 0)
        beta = self.hmm3.BetaScaled(self.hmm3.A, Bo, fact )
        self.failUnless( allclose(beta,resBeta))

    def testBetaScaled_2(self):
        obsIndices = [0, 1, 2]
        Alpha = array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        fact = array([3.,3.,3.])
        resBeta = array([[3., 3.],[3., 3.], [3., 3.]])
        Bo = take(self.hmm4.B, obsIndices, 0)
        beta = self.hmm4.BetaScaled(self.hmm4.A, Bo, fact )
        self.failUnless( allclose(beta,resBeta))

    def testGamma_1(self):
        res = array([[1,0],[0,1],[0,1]], float)
        obsIndices = [0, 1, 1]
        A = self.hmm3.A
        B = self.hmm3.B
        PI = self.hmm3.pi
        Bo = take(B, obsIndices, 0)
        alpha, scale_factors = self.hmm3.AlphaScaled(A, Bo, PI )
        beta = self.hmm3.BetaScaled( A, Bo, scale_factors )
        gamma = self.hmm3._gamma(alpha, beta, scale_factors)
        self.failUnless(allclose(gamma, res))

    def testGamma_2(self):
        res = array([[0.5, 0.5],[0.5, 0.5], [0.5, 0.5]])
        obsIndices = [0, 1, 2]
        A = self.hmm4.A
        B = self.hmm4.B
        PI = self.hmm4.pi
        Bo = take(B, obsIndices, 0)
        alpha, scale_factors = self.hmm4.AlphaScaled(A, Bo, PI )
        beta = self.hmm4.BetaScaled( A, Bo, scale_factors )
        gamma = self.hmm4._gamma(alpha, beta, scale_factors)
        self.failUnless(allclose(gamma, res))

    def testKsi_1(self):
        obsIndices = [0, 1, 1]
        Bo = take(self.hmm3.B, obsIndices, 0)
        alpha, scale_factors = self.hmm3.AlphaScaled(self.hmm3.A, Bo, self.hmm3.pi)
        beta = self.hmm3.BetaScaled( self.hmm3.A, Bo, scale_factors )
        ksi = self.hmm3.Ksi( self.hmm3.A, Bo, alpha, beta )
        resKsi = array([ [[0., 1.],[0., 0.]], [[0., 0.],[0., 1.]]])
        self.failUnless( allclose(ksi, resKsi))

    def testKsi_2(self):
        obsIndices = [0, 1, 2]
        Bo = take(self.hmm4.B, obsIndices, 0)
        alpha, scale_factors = self.hmm4.AlphaScaled(self.hmm4.A, Bo, self.hmm4.pi)
        beta = self.hmm4.BetaScaled( self.hmm4.A, Bo, scale_factors )
        ksi = self.hmm4.Ksi( self.hmm4.A, Bo, alpha, beta )
        resKsi = array([ [[0.25, 0.25],[0.25, 0.25]], [[0.25, 0.25],[0.25, 0.25]]])
        self.failUnless( allclose(ksi, resKsi))

    def testUpdateIterGamma_1(self):
        gamma = array([[1., 0.],[0., 1.],[0., 1.]])
        sigma_gamma_A = zeros(2)
        sigma_gamma_B = zeros(2)
        SGA = array([1., 1.])
        SGB = array([1., 2.])   
        self.hmm3._update_iter_gamma( gamma, sigma_gamma_A, sigma_gamma_B )
        self.failUnless( allclose(SGA, sigma_gamma_A))
        self.failUnless( allclose(SGB, sigma_gamma_B))

    def testUpdateIterGamma_2(self):
        gamma = array([[0.5, 0.5],[0.5, 0.5],[0.5, 0.5]])
        sigma_gamma_A = zeros(2)
        sigma_gamma_B = zeros(2)
        SGA = array([1., 1.])
        SGB = array([1.5, 1.5])   
        self.hmm4._update_iter_gamma( gamma, sigma_gamma_A, sigma_gamma_B )
        self.failUnless( allclose(SGA, sigma_gamma_A))
        self.failUnless( allclose(SGB, sigma_gamma_B))

    def testUpdateIterB_1(self):
        B = self.hmm3.B
        B_bar = B
        gamma = array([[1., 0.],[0., 1.],[0., 1.]])
        obsIndices = [0, 1, 1]
        self.hmm3.UpdateIterB( gamma, obsIndices, B_bar )
        self.failUnless( allclose(B, B_bar))

    def testUpdateIterB_2(self):
        B_bar = self.hmm4.B
        gamma = array([[0.5, 0.5],[0.5, 0.5],[0.5, 0.5]])
        obsIndices = [0, 1, 2]
        result = array([[5./6, 5./6], [5./6, 5./6], [5./6, 5./6]])       
        self.hmm4.UpdateIterB( gamma, obsIndices, B_bar )
        self.failUnless( allclose(result, B_bar))

    def testUpdateIterA_1(self):
        ksi = array([ [[0., 1.],[0., 0.]], [[0., 0.],[0., 1.]] ])
        A_bar = zeros((2,2))
        resA = array([[0., 1.],[0., 1.]])
        self.hmm3._update_iter_A( ksi, A_bar )
        self.failUnless( allclose(resA, A_bar))

    def testUpdateIterA_2(self):
        ksi = array([ [[0.25, 0.25],[0.25, 0.25]], [[0.25, 0.25],[0.25, 0.25]]])
        A_bar = zeros((2,2))
        resA = array([[0.5, 0.5],[0.5, 0.5]])
        self.hmm4._update_iter_A( ksi, A_bar )
        self.failUnless( allclose(resA, A_bar))

    def testNormalizeIterA_1(self):
        A_bar = array([[0., 1.],[0., 1.]])
        SGA = array([1., 1.])
        result = A_bar
        self.hmm3._normalize_iter_A(A_bar, SGA )
        self.failUnless( allclose(A_bar, result))
    
    def testNormalizeIterA_2(self):
        A_bar = array([[0.5, 0.5],[0.5, 0.5]])
        SGA = array([1., 1.])
        result = A_bar
        self.hmm4._normalize_iter_A(A_bar, SGA )
        self.failUnless( allclose(A_bar, result))

    def testNormalizeB_1( self):
        B_bar = array([[1., 0.],[0., 2.]])
        SGB = array([1., 2.])
        result = array([[1., 0.],[0., 1.]])
        self.hmm4.NormalizeB(B_bar, SGB)
        self.failUnless(allclose(B_bar, result))

    def testNormalizeB_2( self):
        B_bar = array([[0.5, 0.5],[0.5, 0.5], [0.5, 0.5]])
        SGB = array([1.5, 1.5])
        result = array([[1./3, 1./3],[1./3, 1./3], [1./3, 1./3]])
        self.hmm4.NormalizeB(B_bar, SGB)
        self.failUnless(allclose(B_bar, result))

    def testStop(self):
        self.hmm3._stop_condition(self.hmm3.A, self.hmm3.pi, self.hmm3.B)
     
    def testFinalStep(self):
        obsIndices = [0, 1, 1]
        gamma = array([[1., 0.],[0., 1.],[0., 1]])
        ksi = array([[[0., 1.],[0., 0.]],[[0., 0.],[0., 1.]] ])
        aBar = array([[0., 1,],[0., 1.]])
        bBar = array([[1., 0.],[0., 1.]])
        piBar = array([1., 0.])
        A , B, PI = self.hmm3._final_step( gamma, ksi, obsIndices )
        self.failUnless( allclose(aBar,A))
        self.failUnless( allclose(bBar, B))
        self.failUnless( allclose(piBar, PI))




class TestFunctionsC (TestFunctions):
    hmmKlass = hmmc.HMM_C

class TestFunctionsF(TestFunctions):
    hmmKlass = hmmf.HMM_F

class TestStates(unittest.TestCase):
    
    def setUp(self):
        self.hmm1 = hmm.HMM(['a', 'b', 'c'], ['1', '2', '3'])
        self.hmm2 = hmm.HMM(['a', 'b'], ['1', '2'])

    def testLearnA_1(self):
        states = ['a','a','b','a','c','b','c','a','b','a','c','b','a']
        result = array([[0.2, 0.4, 0.4],[0.75, 0, 0.25],[1./3, 2./3, 0]])
        self.hmm1._learn_A(states)
        self.failUnless( allclose(result, self.hmm1.A))

    def testLearnA_2(self):
        states = ['a','b','a','a','a','b','a']
        result = array([[0.5, 0.5, 0.],[1., 0., 0.],[0., 0., 1.]])
        self.hmm1._learn_A(states)
        self.failUnless( allclose(result, self.hmm1.A))

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

    def testViterbi_1(self):
        """Test the chain (1,1,1) [determ]"""
        chain = ['1'] * 3
        result = ['a', 'a', 'a']
        self._analyze_chain(self.hmm, chain, result)
		
    def testViterbi_2(self):
        """test the chain (2,1,1,1,1,1) [determ]"""
        result = ['a', 'a', 'a', 'a', 'a', 'a']
        chain = ['2'] + ['1'] * 5
        self._analyze_chain( self.hmm, chain, result)

    def testViterbi_3(self):
        """test the chain (3,2,2,2,2,2) [determ]"""
        chain = ['3'] + ['2'] * 5
        result = ['b', 'b', 'b', 'b', 'b', 'b']
        self._analyze_chain( self.hmm, chain, result)

    def testViterbi_4(self):
        """test the chain (2,2,3,3,3,2) [determ]"""
        chain = ['2','2','3','3','3','2']
        result = ['b', 'b', 'b', 'b', 'b', 'b']
        self._analyze_chain( self.hmm, chain, result)

    def testViterbi_5(self):
        """test the chain (2,2,2,2,2,3) [determ]"""
        chain = ['2','2','2','2','2','3']
        result = ['b', 'b', 'b', 'b', 'b', 'b']
        self._analyze_chain( self.hmm, chain, result)


class TestDeterministicHmmC(TestDeterministic):
    hmmKlass = hmmc.HMM_C

class TestDeterministicHmmF(TestDeterministic):
    hmmKlass = hmmf.HMM_F

class TestBaumWelch(unittest.TestCase):
    """Test the Baumwelch algorithm"""

    def setUp(self):
        self.aHMM = hmm.HMM( ['a', 'b'], ['1', '2','3'],None,None,None)
        self.aHMMC = hmmc.HMM_C( ['a', 'b'], ['1', '2','3'],None,None,None)
        self.aHMMF = hmmf.HMM_F( ['a', 'b'], ['1', '2','3'],None,None,None)
        
        self.aHMM_1 = hmm.HMM( ['a','b'], ['1', '2','3'], 
                            array([[0.7, 0.3],[0.2, 0.8]]),
                            array([[0.2, 0.4], [0.6, 0.2], [0.2, 0.4]]),
                            array([0.2, 0.8]))
        self.aHMM_C = hmmc.HMM_C( ['a','b'], ['1', '2','3'], 
                            array([[0.7, 0.3],[0.2, 0.8]]),
                            array([[0.2, 0.4], [0.6, 0.2], [0.2, 0.4]]),
                            array([0.2, 0.8]))
        self.aHMM_F = hmmf.HMM_F( ['a','b'], ['1', '2','3'], 
                            array([[0.7, 0.3],[0.2, 0.8]]),
                            array([[0.2, 0.4], [0.6, 0.2], [0.2, 0.4]]),
                            array([0.2, 0.8]))
        self.det = hmm.HMM(['a'], ['1', '2'])
        self.test = hmm.HMM( range(20), range(40) )

    def _learn_compare(self, chain, state=None):      
        niterHMM = self.aHMM.learn(chain, state)
        niterHMMC = self.aHMMC.learn(chain, state)
        niterHMMF = self.aHMMF.learn(chain, state)
        
        self.failUnless(allclose(self.aHMMC.A, self.aHMM.A))
        self.failUnless(allclose(self.aHMMF.A, self.aHMM.A))
        self.failUnless(allclose(self.aHMMC.B, self.aHMM.B))
        self.failUnless(allclose(self.aHMMF.B, self.aHMM.B))
        self.failUnless(allclose(self.aHMMC.pi, self.aHMM.pi))
        self.failUnless(allclose(self.aHMMF.pi, self.aHMM.pi))

    def testUpdateIterB(self):
        B_bar = array([[ 0.,0.],[ 0.,0.],[ 0.,0.]])
        B_barF = array([[ 0.,0.],[ 0.,0.],[ 0.,0.]])
        B_barC = array([[ 0.,0.],[ 0.,0.],[ 0.,0.]])
        gamma = array([[ 0.17584567,0.82415433],[ 0.43775031,0.56224969],
                    [ 0.43195352,0.56804648],[ 0.44859571,0.55140429],
                    [ 0.43240921,0.56759079],[ 0.44861501,0.55138499],
                    [ 0.43241002,0.56758998],[ 0.448615,0.551385  ],
                    [ 0.43240908,0.56759092],[ 0.44859262,0.55140738],
                    [ 0.43188047,0.56811953],[ 0.43601172,0.56398828],
                    [ 0.13479001,0.86520999],[ 0.13445915,0.86554085],
                    [ 0.41918731,0.58081269],[ 0.44750776,0.55249224],
                    [ 0.41943579,0.58056421],[ 0.14038371,0.85961629],
                    [ 0.41931846,0.58068154],[ 0.44469141,0.55530859]])

        obsIndices = [0,1,0,1,0,1,2,1,2,1,0,1,0,1,0,1,2,1,0,2]
        result = array([[5./6, 5./6], [5./6, 5./6], [5./6, 5./6]])       
        self.aHMM.UpdateIterB( gamma, obsIndices, B_bar )
        self.aHMMC.UpdateIterB( gamma, obsIndices, B_barC )
        self.aHMMF.UpdateIterB( gamma, obsIndices, B_barF )
        self.failUnless( allclose(B_bar, B_barC))
        self.failUnless( allclose(B_bar, B_barF))
   
    def testBaumwelch_1(self):
        """test the observations (1,2,1,2,1,2,1,2,1,2) """
        chain = ['1','2'] * 5 
        self._learn_compare(chain)

    def testBaumwelch_2(self):
        """test the observations (1,1,1,1,1,2,2,2,2,2) """
        chain =  ['1'] * 5 + ['2'] * 5
        self._learn_compare(chain)

    def testBaumwelch_3(self):
        """test the observations (3,3,3,3,3,3,3,3,3,3) """
        chain = ['3'] * 10
        self._learn_compare(chain)

    def testBaumwelch_4(self):
        """test the observations (1,1,2,2,3,1,3,1,2,1) """
        chain = ['1','1','2','2','3','1','3','1','2','1']
        self._learn_compare(chain)
          
    def testBaumwelch_5(self):
        chain = ['1','2'] * 5
        states = None    #  algorithm not implemented
        self._learn_compare(chain,states)

    def testBaumwelch_6(self):
        chain = ['2'] * 2
        resA = self.det.A
        resB = array([[0.],[1.]])
        respi = self.det.pi
        nit, lc = self.det.learn(chain)
        self.failUnless( allclose(resA, self.det.A))
        self.failUnless( allclose(resB, self.det.B))
        self.failUnless( allclose(respi, self.det.pi))

    def testBaumwelch_7(self):
        observation = self.test.simulate(10)
        nit, lc = self.test.learn(observation)
        self.test.checkHMM()
        #nit, lc = self.test.learn(observation)
        #self.test.checkHMM()
        #nit, lc = self.test.learn(observation)
        #ce test peut provoquer 'warning' (cf test_time.py) 

    def testMultiple_learn_1(self):
        chains = []
        for i in xrange(10):
            chains.append(self.aHMM.simulate(10))
        A = self.aHMM.A
        B = self.aHMM.B
        PI = self.aHMM.pi
        nit = self.aHMM.multiple_learn(chains)
        self.aHMM.checkHMM()
        self.failUnless( allclose(self.aHMM.A, A))
        self.failUnless( allclose(self.aHMM.B, B))
        self.failUnless( allclose(self.aHMM.pi, PI))

    def testMultiple_learn_2(self):
        chains = [ ['1','2','2','2','2'],['1','2','2','2','2','2','2'],
                    ['2','2','2','2','2','2','2']]
        nit = self.aHMM_1.multiple_learn(chains)
        self.aHMM_1.checkHMM()
        nit = self.aHMM_C.multiple_learn(chains)
        self.aHMM_C.checkHMM()
        nit = self.aHMM_F.multiple_learn(chains)
        self.aHMM_F.checkHMM()
        self.failUnless( allclose(self.aHMM_1.A, self.aHMM_C.A))
        self.failUnless( allclose(self.aHMM_1.A, self.aHMM_F.A))
        self.failUnless( allclose(self.aHMM_1.B, self.aHMM_C.B))
        self.failUnless( allclose(self.aHMM_1.B, self.aHMM_F.B))
        self.failUnless( allclose(self.aHMM_1.pi, self.aHMM_C.pi))
        self.failUnless( allclose(self.aHMM_1.pi, self.aHMM_F.pi))

    def testMultiple_learn_3(self):
        chains = [ ['1','2','2','2','2'],['1','2','2','2','2','2','2'],
                    ['2','2','2','2','2','2','2']]
        states = None    #  not implemented
        nit = self.aHMM_1.multiple_learn(chains, states)
        self.aHMM_1.checkHMM()
        nit = self.aHMM_C.multiple_learn(chains)
        self.aHMM_C.checkHMM()
        nit = self.aHMM_F.multiple_learn(chains)
        self.aHMM_F.checkHMM()
        self.failUnless( allclose(self.aHMM_1.A, self.aHMM_C.A))
        self.failUnless( allclose(self.aHMM_1.A, self.aHMM_F.A))
        self.failUnless( allclose(self.aHMM_1.B, self.aHMM_C.B))
        self.failUnless( allclose(self.aHMM_1.B, self.aHMM_F.B))
        self.failUnless( allclose(self.aHMM_1.pi, self.aHMM_C.pi))
        self.failUnless( allclose(self.aHMM_1.pi, self.aHMM_F.pi))

    def testMultiple_learn_4(self):
        chains = [ ['2'] * 2, ['2'] * 3,['2'] * 4]
        resA = self.det.A
        resB = array([[0.],[1.]])
        respi = self.det.pi
        nit, lc = self.det.multiple_learn(chains)
        self.failUnless( allclose(resA, self.det.A))
        self.failUnless( allclose(resB, self.det.B))
        self.failUnless( allclose(respi, self.det.pi))

    def testMultiple_learn_5(self):
        chains = []
        for i in xrange(10):
            chains.append(self.test.simulate(10))
        self.test.setRandomProba()
        nit, lc = self.test.multiple_learn(chains)
        self.test.checkHMM()


class TestPickle(unittest.TestCase):
    """ test the pickle implementation """
    
    hmmKlass = hmm.HMM

    def setUp(self):
        self.hmm1 = self.hmmKlass( ['a', 'b'], ['1', '2','3'],None,None,None)
        self.hmm2 = self.hmmKlass( ['a', 'b'], ['1', '2','3'],None,None,None)
        self.hmm2.setRandomProba()
        self.hmm3 = self.hmmKlass( ['a'], ['1'],None,None,None)
        self.hmm4 = self.hmmKlass( [], [],None,None,None)

    def _compare(self, h1, h2, SaveState=None):
        self.failUnless( allclose(h1.A, h2.A))
        self.failUnless( allclose(h1.B, h2.B))
        self.failUnless( allclose(h1.pi,h2.pi))
        self.failUnless( allclose(h1.N, h2.N))
        self.failUnless( allclose(h1.M, h2.M))
        if SaveState:
            self.failUnless( h1.X_index == h2.X_index)  
            self.failUnless( h1.omega_X == h2.omega_X)  
            self.failUnless( h1.omega_O == h2.omega_O)  
            self.failUnless( h1.O_index == h2.O_index)      

    def testPickle_1(self):
        f = open('save.data', 'w')
        self.hmm1.saveHMM(f)
        f.close()
        f = open('save.data', 'r')
        self.hmm2.loadHMM(f)
        f.close()
        self.hmm2.checkHMM()
        self._compare(self.hmm1, self.hmm2)
        os.remove('save.data')


    def testPickle_2(self):
        f = open('save.data', 'w')
        self.hmm1.saveHMM(f,1)
        f.close()
        f = open('save.data', 'r')
        self.hmm3.loadHMM(f)
        f.close()
        self.hmm3.checkHMM()
        self._compare(self.hmm1, self.hmm3)
        os.remove('save.data')

    def testPickle_3(self):
        f = open('save.data', 'w')
        self.hmm1.saveHMM(f,1)
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
