from numpy import array
from logilab.hmm.hmm import HMM

def test4():
    """A simple simulation test"""
    test = HMM(['a', 'b'], ['s1', 's2', 's3'],
               array([[.3, .7], [.5, .5]]),
               array([[.5, 0], [.5, .5], [0, .5]]),
               array([.9, .1]))
    test.dump()
    print test.simulate(10)
    print test.simulate(10, 1)

if __name__ == "__main__":
    test4()
