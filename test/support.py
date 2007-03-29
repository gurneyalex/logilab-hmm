




verbose=1

def set_verbose( v ):
    global verbose
    verbose = v

def show_analysis(h,chain):
    if verbose:
        print "Chain      : ", chain
        print "analyse    : ", h.analyze(chain)
        print "analyse_log: ", h.analyze_log(chain)


import time

AVG = 2.

def timecall( desc, func, *args):
    # warm up call
    func(*args)
    t0 = time.time()
    func(*args)
    t1 = time.time()
    COUNT = AVG/(t1-t0)
    if COUNT<2:
        COUNT = 2
    S = 0.
    _min = t1-t0
    for i in xrange(COUNT):
        t0 = time.time()
        func(*args)
        t1 = time.time()
        t = t1-t0
        S += t
        if t<_min:
            _min = t
    un = "s"
    _avg = S/COUNT
    if _min < 1.:
        un = "ms"
        _min *= 1000
        _avg *= 1000
    print "%s: avg = %8.2f%s ; min = %8.2f%s ; runs = %d" % (desc, _avg, un,  _min, un, COUNT)
