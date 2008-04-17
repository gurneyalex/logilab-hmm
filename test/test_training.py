import autopath
from support import timecall, deterministic_hmm

from logilab.hmm.hmm import HMM
from logilab.hmm.hmmc import HMM_C
from logilab.hmm.hmmf import HMM_F

from numpy import array, take, alltrue, allclose
from numpy.linalg import norm

RTOL = 1e-7
ATOL = 1e-10




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
    test = HMM(range(3), range(4))
    test.setRandomProba()
    print 'Original'
    print 'A =', test.A
    print 'B =', test.B
    print 'pi =', test.pi
    print
    print 'Generating sample data...'
    sample = test.simulate(1000)
    print 'Randomizing model...'
    test.setRandomProba()
    print 'Training model...'
    test.learn(sample, None, 1000)
    print 'trained values'
    print 'A =', test.A
    print 'B =', test.B
    print 'pi=', test.pi


def test8():
    """Same as test6 but learning over several observations from
    the same chain"""
    test = HMM(range(10), range(50))
    print 'Generating sample data...'
    l = []
    test.setRandomProba()
    for i in range(10):
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


def test9_errors(gene, test):
    """Compute an error (distance) between two chains"""
    error1 = norm(gene.A - test.A)
    error2 = norm(gene.B - test.B)
    error2bis = norm(gene.B - test.B[:, ::-1])
    if error2 < error2bis:
        error3 = norm( gene.pi - test.pi )
    else:
        error2 = error2bis
        error3 = norm( gene.pi - test.pi[::-1] )
    return error1, error2, error3

#def test9_errors( gene, test):
#    gA, gB, gPI = gene.normalize( [1,0] )
#    tA, tB, tPI = test.normalize( [1,0] )
#    if norm(gB-tB)>norm(gene.B-test.B):
#        gA, gB, gPI = gene.A, gene.B, gene.pi
#        tA, tB, tPI = test.A, test.B, test.pi
#    return norm( gA-tA ), norm( gB-tB ), norm( gPI - tPI )


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


def test10(HMM, n=10): 
    """This test generate a simple HMM (determinist state transitions)
    And check if the algoritm converge in less than 1000 iterations"""
    S,V,A,B,PI = deterministic_hmm()
    gene = HMM( S, V, A, B, PI )
    print "Generating data..."
    data = [ gene.simulate(20) for i in range(100) ] 
    test = HMM(['a', 'b'], ['s1', 's2', 's3'])
    errors = []
    for i in xrange(n):
        print "round ", i
        test.setRandomProba()
        iteration, curve = test.multiple_learn(data)        
        error1, error2, error3 = test9_errors( gene, test )
        _A, _B, _pi = test.normalize()
        print "A: ", _A
        print "B: ", _B
        print "Pi:", _pi
        errors.append([i, iteration, error1, error2, error3, curve, 0])
    test9_display(errors)
    return errors, test

if __name__ == '__main__':
    #test6()
    #test8()
    #test10(HMM_C)
    test10(HMM_F)
    #test10(HMM)
