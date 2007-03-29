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
    
    AlphaScaled = staticmethod(_hmmf.hmm_for.alpha_scaled)
    BetaScaled = staticmethod(_hmmf.hmm_for.beta_scaled)
    Ksi = staticmethod(_hmmf.hmm_for.hmm_ksi)
    UpdateIterB = staticmethod(_hmmf.hmm_for.update_iter_b)
    CorrectM = staticmethod(_hmmf.hmm_for.correctm)
    NormalizeB = staticmethod(_hmmf.hmm_for.normalize_b)


class HMM_F_prof(HMM):
    "HMM"
    AlphaScaled = staticmethod(_hmmf.hmm_for.alpha_scaled)
    BetaScaled = staticmethod(_hmmf.hmm_for.beta_scaled)
    Ksi = staticmethod(_hmmf.hmm_for.hmm_ksi)
    UpdateIterB = staticmethod(_hmmf.hmm_for.update_iter_b)
    CorrectM = staticmethod(_hmmf.hmm_for.correctm)
    NormalizeB = staticmethod(_hmmf.hmm_for.normalize_b)
