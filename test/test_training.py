from support import deterministic_hmm

from logilab.hmm.hmm import HMM
from logilab.hmm.hmmc import HMM_C
from logilab.hmm.hmmf import HMM_F
import logilab.hmm.hmmS as hmmS
from numpy.linalg import norm

RTOL = 1e-7
ATOL = 1e-10




def test5():
    """Train a model over some simulated values from an initial
    model"""
    test = HMM(['a', 'b'], ['s1', 's2', 's3'])
    test.set_random_proba()
    print 'Original'
    print 'A =', test.A
    print 'B =', test.B
    print 'pi =', test.pi
    print
    print 'Generating sample data...'
    sample =  test.simulate(500)
    print 'Randomizing model...'
    test.set_random_proba()
    print 'Training model...'
    test.learn(sample, None, 3000)
    print 'trained values'
    print 'A =', test.A
    print 'B =', test.B
    print 'pi =', test.pi

def test6():
    """Same as test5 but with a bigger state space and observations values"""
    test = HMM(range(3), range(4))
    test.set_random_proba()
    print 'Original'
    print 'A =', test.A
    print 'B =', test.B
    print 'pi =', test.pi
    print
    print 'Generating sample data...'
    sample = test.simulate(1000)
    print 'Randomizing model...'
    test.set_random_proba()
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
    test.set_random_proba()
    for i in range(10):
        obs = test.simulate(100)
        l.append(obs)
    print 'Original'
    print 'A =', test.A
    print 'B =', test.B
    print 'pi =', test.pi
    print
    print 'Randomizing model...'
    test.set_random_proba()
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

def test9_display(errors):
    """Displays the computed errors"""
    for t in errors:
        print "Test ", t[0], "Iterations:", t[1], "ErrA:", t[2], "ErrB", t[3],
        print "ErrPi", t[4], "Avg time: ", t[6]

def test9(n=10):
    """This test generate a simple HMM (determinist state transitions)
    And check if the algoritm converge in less than 1000 iterations"""
    S, V, A0, B0, PI0 = deterministic_hmm()
    gene = HMM(S, V, A0, B0, PI0)
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
        test.set_random_proba()
    test9_display(errors)
    return errors

def test10(HMM, n=10): 
    """This test generate a simple HMM (determinist state transitions)
    And check if the algoritm converge in less than 1000 iterations"""
    S, V, A, B, PI = deterministic_hmm()
    gene = HMM( S, V, A, B, PI )
    print "Generating data..."
    data = [ gene.simulate(70) for i in range(10) ] 
    test = HMM(['a', 'b'], ['s1', 's2', 's3'])
    errors = []
    for i in xrange(n):
        print "round ", i
        test.set_random_proba()
        iteration, curve = test.multiple_learn(data)        
        error1, error2, error3 = test9_errors( gene, test )
        _A, _B, _pi = test.normalize()
        errors.append([i, iteration, error1, error2, error3, curve, 0])
    test9_display(errors)
    return errors, test

def test11(HMM, n=10): 
    """This test generate a simple HMM (determinist state transitions)
    And check if the algoritm converge in less than 1000 iterations"""
    S,V,A,B,PI = deterministic_hmm()
    gene = HMM( S, V, A, B, PI )
    print "Generating data..."
    data = [ gene.simulate(30) for i in range(30) ] 
    test = HMM(['a', 'b'], ['s1', 's2', 's3'])
    errorsPall = []
    errorsPk = []
    errorsUnit = []
    for i in xrange(n):
        print "round ", i
        test.set_random_proba()
        A = test.A
        B = test.B
        pi = test.pi
        test.ensemble_averaging(data, "Pall", 1000, 0)        
        errorPall1, errorPall2, errorPall3 = test9_errors( gene, test )
        test.A = A
        test.B = B
        test.pi = pi
        test.ensemble_averaging(data, "Pk", 1000, 0)        
        errorPk1, errorPk2, errorPk3 = test9_errors( gene, test )
        test.A = A
        test.B = B
        test.pi = pi
        test.ensemble_averaging(data, "unit", 1000, 0)        
        errorUnit1, errorUnit2, errorUnit3 = test9_errors( gene, test )
        _A, _B, _pi = test.normalize()
        iteration = 1
        curve = 1
        errorsPall.append([i, iteration, errorPall1, errorPall2, errorPall3, curve, 0])
        errorsPk.append([i, iteration, errorPk1, errorPk2, errorPk3, curve, 0])
        errorsUnit.append([i, iteration, errorUnit1, errorUnit2, errorUnit3, curve, 0])
    print "-----------------Pall----------------"
    test9_display(errorsPall)
    print "------------------Pk-----------------"
    test9_display(errorsPk)
    print "-----------------Unit----------------"
    test9_display(errorsUnit)
    return errorsUnit, test

def test12(HMMS, n=10): 
    """This test generate a simple HMM (determinist state transitions)
    And check if the algoritm converge in less than 1000 iterations"""
    S,V,A,B,PI = deterministic_hmm()
    gene = HMM( S, V, A, B, PI )
    print "Generating data..."
    setObservation = []
    setState = []
    o = []
    s = []
    chains = []
    for i in xrange(30):
        chains.append(gene.simulate(30, 1))
        for couple in chains[i]:
            o.append(couple[1])
            s.append(couple[0])
        setObservation.append(o)
        setState.append(s)
        o = []
        s = []

    test = HMMS(['a', 'b'], ['s1', 's2', 's3'])
    errorsPall = []
    errorsPk = []
    errorsUnit = []
    errors_mult_l = []
    for i in xrange(n):
        print "round ", i
        test.set_random_proba()
        A = test.A
        B = test.B
        pi = test.pi
        test.ensemble_averaging(setObservation, setState, "Pall", 1000, 0)        
        errorPall1, errorPall2, errorPall3 = test9_errors( gene, test )
        test.A = A
        test.B = B
        test.pi = pi
        test.ensemble_averaging(setObservation, setState, "Pk", 1000, 0)    
        errorPk1, errorPk2, errorPk3 = test9_errors( gene, test )
        test.A = A
        test.B = B
        test.pi = pi
        test.ensemble_averaging(setObservation, setState, "unit", 1000, 0)
        errorUnit1, errorUnit2, errorUnit3 = test9_errors( gene, test )
        test.A = A
        test.B = B
        test.pi = pi
        test.multiple_learn(setObservation, setState, 1000, 0)
        error_mult_l1, errorUnit_mult_l2, error_mut_l3 = test9_errors( gene, test )
        _A, _B, _pi = test.normalize()
        iteration = 1
        curve = 1

        errorsPall.append([i, iteration, errorPall1, errorPall2, errorPall3, curve, 0])
        errorsPk.append([i, iteration, errorPk1, errorPk2, errorPk3, curve, 0])
        errorsUnit.append([i, iteration, errorUnit1, errorUnit2, errorUnit3, curve, 0])
        errors_mult_l.append([i, iteration, error_mult_l1, errorUnit_mult_l2, error_mut_l3, curve, 0])

    print "-----------------Pall----------------"
    test9_display(errorsPall)
    print "------------------Pk-----------------"
    test9_display(errorsPk)
    print "-----------------Unit----------------"
    test9_display(errorsUnit)
    print "-----------multiple_learn------------"
    test9_display(errors_mult_l)
    return errorsUnit, test

if __name__ == '__main__':
    #test6()
    #test8()
    #test10(HMM_C)
    #test10(HMM_F)
    #test11(HMM)
    test12(hmmS.HMMS_C)
