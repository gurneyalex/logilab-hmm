

from numpy import array
import logilab.hmm.hmm as hmm
import unittest

verbose=1

def show_analysis(h,chain):
    if verbose:
        print "Chain      : ", chain
        print "analyse    : ", h.analyze(chain)
        print "analyse_log: ", h.analyze_log(chain)


class TestDeterministic(unittest.TestCase):
    """Test the viterbi algorithm on a deterministic chain"""
    def setUp(self):
        self.hmm = hmm.HMM( ['a', 'b'], ['1', '2', '3'],
                        array([[1.0, 0.0],
                               [0.0, 1.0]]),
                        array([[1.0, 0.0],
                               [ .5,  .5],
                               [0.0, 1.0]]),
                        array([0.5, 0.5]))

    def _analyze_chain(self, mc, chain, result = None):
        an1 = mc.analyze(chain)
        an2 = mc.analyze_log(chain)
        self.assertEqual(an1, an2)
        if result:
            self.assertEqual(an1, result)
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
        
class TestSimple(unittest.TestCase):
    """Test the viterbi algorithm on a deterministic chain"""
    def setUp(self):
        self.hmm1 = hmm.HMM(['a', 'b'], ['1', '2', '3'],
                        array([[1.0, 0.0], [0.0, 1.0]]),
                        array([[1.0, 0], [.5, .5], [0, 1.0]]),
                        array([0.5, 0.5]))
        self.hmm2 = hmm.HMM(['a', 'b'], ['1', '2', '3'],
                        array([[.3, .7], [.5, .5]]),
                        array([[.5, 0], [.5, .5], [0, .5]]),
                        array([.9, .1]))


    def _analyze_chain(self, mc, chain, result = None):
        an1 = self.hmm1.analyze(chain)
        an2 = self.hmm1.analyze_log(chain)
        self.assertEqual(an1, an2)
        if result:
            self.assertEqual(an1, result)
            self.assertEqual(an1, result)

    def testViterbi_1(self):
        """Test the chain (1,1,1) [simple]"""
        chain = ['1'] * 3
        result = ['a', 'a', 'a']
        self._analyze_chain(self.hmm1, chain, result)

    def testViterbi_2(self):
        """test the chain (2,1,1,1,1,1) [simple]"""
        result = ['a', 'a', 'a', 'a', 'a', 'a']
        chain = ['3'] + ['1'] * 5
        self._analyze_chain(self.hmm1, chain, result)

    def testViterbi_3(self):
        """test the chain (3,2,2,2,2,2) [simple]"""
        chain = ['3'] + ['2'] * 5
        result = ['b', 'b', 'b', 'b', 'b', 'b']
        self._analyze_chain(self.hmm1, chain, result)

#suite= unittest.makeSuite(TestLikelihood,'test')

def test1_analyze( h, chain ):
    """simple test"""
    print "Chain      : ", chain
    print "analyse    : ", h.analyze(chain)
    print "analyse_log: ", h.analyze_log(chain)
    
def test1():
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

def test2():
    """This test will display the computed likelyhood of some sentences given
    some predetermined transition and observation matrices"""
    nom = 'soleil ville parc chat chien jaune souris poule jardin bec griffe sel poulet poivre'.split()
    verbe = 'brillait mange chasse court dort griffe est ressemble'.split()
    adj = 'grand petit gentil endormi jaune grande petite gentille endormie'.split()
    adv = 'vigoureusement rapidement vite'.split()
    det = 'le la les un une des'.split()
    pro = 'je tu il elle on nous vous ils elles le la les lui moi toi eux'.split()
    pre = 'à pour sur sous près de du au avec sans'.split()
    univers = []
    for mot in nom + verbe + adj + adv + det + pro + pre:
        univers.append(mot)
    test = HMM(['adj', 'nom', 'verbe', 'adv', 'det', 'pro', 'pre'], univers)
    test.A[:,:] = 0.0 # clear transition proba
    test.setTransitionProba('det', 'adj', .5)
    test.setTransitionProba('det', 'nom', .5)
    test.setTransitionProba('nom', 'adj', .2)
    test.setTransitionProba('nom', 'verbe', .2)
    test.setTransitionProba('nom', 'nom', .2)
    test.setTransitionProba('nom', 'pro', .2)
    test.setTransitionProba('nom', 'adv', .1)
    test.setTransitionProba('nom', 'pre', .1)
    test.setTransitionProba('pro', 'adj', .2)
    test.setTransitionProba('pro', 'verbe', .2)
    test.setTransitionProba('pro', 'nom', .2)
    test.setTransitionProba('pro', 'pro', .2)
    test.setTransitionProba('pro', 'adv', .1)
    test.setTransitionProba('pro', 'pre', .1)
    test.setTransitionProba('adj', 'adj', .2)
    test.setTransitionProba('adj', 'nom', .6)
    test.setTransitionProba('adj', 'pre', .1)
    test.setTransitionProba('adj', 'verbe', .1)
    test.setTransitionProba('pre', 'det', .8)
    test.setTransitionProba('pre', 'nom', .2)
    test.setTransitionProba('verbe', 'verbe', .2)
    test.setTransitionProba('verbe', 'adv', .2)
    test.setTransitionProba('verbe', 'det', .3)
    test.setTransitionProba('verbe', 'pre', .3)
    test.setTransitionProba('adv', 'pre', .3)
    test.setTransitionProba('adv', 'verbe', .4)
    test.setTransitionProba('adv', 'det', .3)
    test.checkHMM()
    for liste, state in [ (nom, 'nom'), (verbe, 'verbe'), (adj, 'adj'),
                          (adv, 'adv'), (det, 'det'), (pro, 'pro'),
                          (pre, 'pre')]:
        taille = len(liste)
        proba = 1.0 / taille
        for mot in liste:
            test.setObservationProba(state, mot, proba)
        test.setInitialProba(state, 1. / 7)

    phrases = ('un grand soleil jaune brillait vigoureusement sur la ville endormie',
               'le petit chat mange des souris',
               'je mange du poulet au poivre sans sel',
               )
               
    for p in phrases:
        p = p.split()
        a = test.analyze(p)
        for i in range(len(p)):
            p[i] = (p[i], a[i])
        print p


def test3():
    """This test compares the results of some of the C and Python functions
    used by the algorithm, with the assumption that the C and Python functions
    should return the same results"""
    nom = 'soleil ville parc chat chien jaune souris poule jardin bec griffe sel poulet poivre'.split()
    verbe = 'brillait mange chasse court dort griffe est ressemble'.split()
    adj = 'grand petit gentil endormi jaune grande petite gentille endormie'.split()
    adv = 'vigoureusement rapidement vite'.split()
    det = 'le la les un une des'.split()
    pro = 'je tu il elle on nous vous ils elles le la les lui moi toi eux'.split()
    pre = 'à pour sur sous près de du au avec sans'.split()
    univers = []
    for mot in nom + verbe + adj + adv + det + pro + pre:
        univers.append(mot)
    test = HMM(['adj', 'nom', 'verbe', 'adv', 'det', 'pro', 'pre'], univers)

    test.setRandomProba()

    Obs = test._getObservationIndices('le petit chat mange des souris'.split())
    Bo = test._getObservations(Obs)
    r1, scale = _alpha_scaled(test.A, Bo, test.pi)
    r1_opt, scale_opt = _alpha_scaled_prof(test.A, Bo, test.pi)
    r2 = _beta_scaled(test.A, Bo, scale)
    r2_opt = _beta_scaled_prof(test.A, Bo, scale_opt)

    print "Alpha"
    print r1
    if  (allclose(r1_opt, r1)):
        print "Python and C algorithms returned the same results"
    else:
        print "There was significant differences between the results from C and python"
        print r1_opt - r1
    print "Maximum difference was:", _max(_max(abs(r1_opt - r1)))
    print "Beta"
    print r2
    if  (allclose(r2_opt, r2)):
        print "Python and C algorithms returned the same results"
    else:
        print "There was significant differences between the results from C and python"
        print r2_opt - r2
    print "Maximum difference was:", _max(_max(abs(r2_opt - r2)))
    
def test4():
    """A simple simulation test"""
    test = HMM(['a', 'b'], ['s1', 's2', 's3'],
               array([[.3, .7], [.5, .5]]),
               array([[.5, 0], [.5, .5], [0, .5]]),
               array([.9, .1]))
    test.dump()
    print test.simulate(10)
    print test.simulate(10, 1)

def test5():
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
    test.setRandomProba()
    print 'Training model...'
    test.learn(sample, None, 3000)
    print 'trained values'
    print 'A =', test.A
    print 'B =', test.B
    print 'pi =', test.pi

def test6():
    """Same as test5 but with a bigger state space and observations values"""
    test = HMM(range(5), range(10))
    test.setRandomProba()
    print 'Original'
    print 'A =', test.A
    print 'B =', test.B
    print 'pi =', test.pi
    print
    print 'Generating sample data...'
    sample = test.simulate(10000)
    print 'Randomizing model...'
    test.setRandomProba()
    print 'Training model...'
    test.learn(sample, None, 10000)
    print 'trained values'
    print 'A =', test.A
    print 'B =', test.B
    print 'pi=', test.pi

def test7():
    """Tests saving and loading a MC"""
    test = HMM(range(3), range(5))
    f = open('/tmp/hmm_pickle', 'w')
    test.saveHMM(f, 1)
    f.close()
    f = open('/tmp/hmm_pickle')
    test2 = HMM([], [])
    test2.loadHMM(f)
    print test2.omega_X, test2.omega_O
    print test2.A
    print test2.B
    print test2.pi
    
def test8():
    """Same as test6 but learning over several observations from
    the same chain"""
    test = HMM(range(10), range(50))
    print 'Generating sample data...'
    l = []
    test.setRandomProba()
    for i in range(100):
        obs = test.simulate(100)
        l.append(obs)
    print 'Original'
    print 'A =', test.A
    print 'B =', test.B
    print 'pi =', test.pi
    print
    print 'Randomizing model...'
    test.setRandomProba()
    print 'Training model...'
    test.multiple_learn(l)
    print 'trained values'
    print 'A =', test.A
    print 'B =', test.B
    print 'pi =', test.pi


def deterministic_hmm():
    """Returns the matrices of a deterministic HMM"""
    test = HMM( ['a', 'b'], ['s1', 's2', 's3'],
                [[0.0, 1.0], [1.0, 0.0]],
                [[0.8, 0.0], [0.0, 0.8], [0.2, 0.2]], [0.7, 0.3])
    return test

def norm2(m):
    """Returns the norm2 of a matrix"""
    v = reshape(m, (product(m.shape), ))
    return sqrt(dot(v, v)) / product(m.shape)

def test9_errors(gene, test):
    """Compute an error (distance) between two chains"""
    error1 = norm2(gene.A - test.A)
    error2 = norm2(gene.B - test.B)
    error2bis = norm2(gene.B - test.B[:, ::-1])
    if error2 < error2bis:
        error3 = norm2( gene.pi - test.pi )
    else:
        error2 = error2bis
        error3 = norm2( gene.pi - test.pi[::-1] )
    return error1, error2, error3

def test9_display(errors):
    """Displays the computed errors"""
    for t in errors:
        print "Test ", t[0], "Iterations:", t[1], "ErrA:", t[2], "ErrB", t[3],
        print "ErrPi", t[4], "Avg time: ", t[6]
    
def test9(n=10):
    """This test generate a simple HMM (determinist state transitions)
    And check if the algoritm converge in less than 1000 iterations"""
    gene = deterministic_hmm()
    data = gene.simulate(500)
    test = HMM(['a', 'b'], ['s1', 's2', 's3'])
    errors = []
    from time import time
    for i in xrange(n):
        t1 = time()
        iteration, curve = test.learn(data)
        t2 = time()
        error1, error2, error3 = test9_errors( gene, test )
        print "A: ", test.A
        print "B: ", test.B
        errors.append([i, iteration, error1, error2, error3,
                       curve, (t2 - t1) / iteration])
        test.setRandomProba()
    test9_display(errors)
    return errors

def test10(n=10):
    """This test generate a simple HMM (determinist state transitions)
    And check if the algoritm converge in less than 1000 iterations"""
    gene = deterministic_hmm()
    print "Generating data..."
    data = [ gene.simulate(20) for i in range(100) ]
    test = HMM(['a', 'b'], ['s1', 's2', 's3'])
    errors = []
    for i in xrange(n):
        print "round ", i
        iteration, curve = test.multiple_learn(data)
        error1, error2, error3 = test9_errors( gene, test )
        print "A: ", test.A
        print "B: ", test.B
        print "Pi:", test.pi
        errors.append([i, iteration, error1, error2, error3, curve, 0])
        test.setRandomProba()
    test9_display(errors)
    return errors, test
    
if __name__ == '__main__':
    for test_num in range(1, 11):
        print '-' * 80
        print 'start test', test_num
        globals()['test%s' % test_num]()


if __name__ == '__main__':
    unittest.main()
