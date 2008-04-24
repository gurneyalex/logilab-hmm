import time

verbose=1

def set_verbose( v ):
    global verbose
    verbose = v

def show_analysis(hmm, chain):
    if verbose:
        print "Chain      : ", chain
        print "analyse    : ", hmm.analyze(chain)
        print "analyse_log: ", hmm.analyze_log(chain)

# try to run the call for at most 2sec.
AVG = 2.

def timecall( desc, func, *args):
    # warm up call
    func(*args)
    t0 = time.time()
    func(*args)
    t1 = time.time()
    count = int(AVG / (t1 - t0))
    if count < 2:
        count = 2
    S = 0.
    _min = t1 - t0
    for i in xrange(count):
        t0 = time.time()
        func(*args)
        t1 = time.time()
        t = t1 - t0
        S += t
        if t < _min:
            _min = t
    un = "s"
    _avg = S / count
    if _min < 1.:
        un = "ms"
        _min *= 1000
        _avg *= 1000
    print "%s: avg = %8.2f%s ; min = %8.2f%s ; runs = %d" % (desc, _avg, un,  _min, un, count)

def timecall_one( desc, func, *args):
    t0 = time.time()
    func(*args)
    t1 = time.time()
    t = t1 - t0
    if t < 1.:
        un = "ms"
        t *= 1000
    else:
        un = "s"
    print "%s: avg = %8.2f%s ; runs = %d" % (desc, t, un, 1)
    

def deterministic_hmm():
    """Returns the matrices of a deterministic HMM"""
    S = ['a', 'b']
    V = ['s1', 's2', 's3']
    A0 = [[0.0, 1.0], [1.0, 0.0]]
    B0 = [[0.8, 0.0], [0.0, 0.8], [0.2, 0.2]]
    PI0 = [0.7, 0.3]
    return S, V, A0, B0, PI0

def norm2(m):
    """Returns the norm2 of a matrix"""
    v = reshape(m, (product(m.shape)))
    return sqrt(dot(v, v)) / product(m.shape)

STATES = "abcdefghijklmnopqrstuvwxyz"
VALUES = [ "s%02d" % i for i in range(100) ]

def deterministic_hmm_gen( NSTATES = (2,4,10,15,20),
                           NVALUES = range(5,100,10)
                           ):
    """Generates 5-tuples descriptions of various
    state-deterministic HMM
    """
    for nstate in (2, 4, 10, 15, 20):
        for nobs in range(5, 100, 10):
            states = list(STATES[:nstate])
            values = list(VALUES[:nobs])
            ID = identity(nstate)
            A = concatenate((ID[1:, :], ID[0:1, :] ), 0)
            pi = zeros( (nstate), float )
            pi[0] = 1.
            B = zeros( (nobs, nstate), float )
            bi = zeros( (nobs), float )
            for k in range(nstate):
                bi[:] = .5 / (nobs - 1)
                bi[k] = .5
                B[:, k] = bi
            yield states, values, A, B, pi
