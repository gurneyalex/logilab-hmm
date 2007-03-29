

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
