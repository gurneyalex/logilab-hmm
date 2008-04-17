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
    
    AlphaScaled = staticmethod(_hmmf.hmm_for.alpha_scaled)
    BetaScaled = staticmethod(_hmmf.hmm_for.beta_scaled)
    Ksi = staticmethod(_hmmf.hmm_for.hmm_ksi)
#    UpdateIterB = staticmethod(_hmmf.hmm_for.update_iter_b)
    @staticmethod
    def UpdateIterB(gamma, obsIndices, B_Bar):
        B_Bar[:] =  _hmmf.hmm_for.update_iter_b(gamma, obsIndices, B_Bar)
    CorrectM = staticmethod(_hmmf.hmm_for.correctm)
#    NormalizeB = staticmethod(_hmmf.hmm_for.normalize_b)
    @staticmethod
    def NormalizeB(B_Bar, sigma_gamma_B):
        B_Bar[:] = _hmmf.hmm_for.normalize_b(B_Bar, sigma_gamma_B)
    _gamma = staticmethod(_hmmf.hmm_for.gamma)

class HMM_F_prof(HMM):
    "HMM"

    ORDER = "F"
    
    AlphaScaled = staticmethod(_hmmf.hmm_for.alpha_scaled)
    BetaScaled = staticmethod(_hmmf.hmm_for.beta_scaled)
    Ksi = staticmethod(_hmmf.hmm_for.hmm_ksi)
    UpdateIterB = staticmethod(_hmmf.hmm_for.update_iter_b)
    CorrectM = staticmethod(_hmmf.hmm_for.correctm)
    NormalizeB = staticmethod(_hmmf.hmm_for.normalize_b)
