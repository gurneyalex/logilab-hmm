"""This module uses the C module to implement parts of hmm operations"""


from hmm import HMM

try:
    # prefer local version (for dev)
    import _hmm
except ImportError:
    # Let this ImportError be handled by the caller module
    import logilab.hmm._hmm as _hmm
    # TODO : check for compatibility

raise DeprecationWarning("The C version of HMM is deprecated and will be removed")

class CHMM(HMM):
    pass
