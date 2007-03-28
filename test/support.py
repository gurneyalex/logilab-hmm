




verbose=1

def set_verbose( v ):
    global verbose
    verbose = v

def show_analysis(h,chain):
    if verbose:
        print "Chain      : ", chain
        print "analyse    : ", h.analyze(chain)
        print "analyse_log: ", h.analyze_log(chain)

