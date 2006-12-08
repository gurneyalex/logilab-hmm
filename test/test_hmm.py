

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
        
if __name__ == '__main__':
    unittest.main()
