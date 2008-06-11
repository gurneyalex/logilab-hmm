"""This module uses the Fortran module to implement parts of hmm operations"""


from hmm import HMM

try:
    # prefer local version (for dev)
    import _hmmf
except ImportError:
    # Let this ImportError be handled by the caller module
    import logilab.hmm._hmmf as _hmmf
    # TODO : check for compatibility


class HMM_F(HMM):
    "HMM"

    ORDER = "F"
    
    alpha_scaled = staticmethod(_hmmf.hmm_for.alpha_scaled)
    beta_scaled = staticmethod(_hmmf.hmm_for.beta_scaled)
    ksi = staticmethod(_hmmf.hmm_for.hmm_ksi)
#    update_iter_B = staticmethod(_hmmf.hmm_for.update_iter_b)
# XXX: there should be *NO* copy, we must fix the function prototype instead
    @staticmethod
    def update_iter_B(gamma, obsIndices, B_Bar):
        B_Bar[:] =  _hmmf.hmm_for.update_iter_b(gamma, obsIndices, B_Bar)
    correct_M = staticmethod(_hmmf.hmm_for.correctm)
#    normalize_B = staticmethod(_hmmf.hmm_for.normalize_b)
    @staticmethod
    def normalize_B(B_Bar, sigma_gamma_B):
        B_Bar[:] = _hmmf.hmm_for.normalize_b(B_Bar, sigma_gamma_B)
    _gamma = staticmethod(_hmmf.hmm_for.gamma)

class HMM_F_prof(HMM):
    "HMM"

    ORDER = "F"
    
    alpha_scaled = staticmethod(_hmmf.hmm_for.alpha_scaled)
    beta_scaled = staticmethod(_hmmf.hmm_for.beta_scaled)
    ksi = staticmethod(_hmmf.hmm_for.hmm_ksi)
    update_iter_B = staticmethod(_hmmf.hmm_for.update_iter_b)
    correct_M = staticmethod(_hmmf.hmm_for.correctm)
    normalize_B = staticmethod(_hmmf.hmm_for.normalize_b)
