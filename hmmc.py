"""This module uses the C module to implement parts of hmm operations"""


from hmm import HMM

try:
    # prefer local version (for dev)
    import _hmm_c as _hmm
except ImportError:
    # Let this ImportError be handled by the caller module
    import logilab.hmm._hmm_c as _hmm
    # TODO : check for compatibility

try:
    raise DeprecationWarning("The C version of HMM is deprecated and will be removed")
except:
    pass


class HMM_C(HMM):
    "HMM"
    
    AlphaScaled = _hmm._hmm_alpha_scaled
    BetaScaled = _hmm._hmm_beta_scaled
    Ksi = _hmm._hmm_ksi
    UpdateIterB = _hmm._hmm_update_iter_B
    CorrectM = _hmm._hmm_correctm
    NormalizeB = _hmm._hmm_normalize_B

def _alpha_scaled_prof(A, Bo, pi):
    """See _alpha_scaled. This is a wrapper for the C version
    of the function."""
    return _hmm._hmm_alpha_scaled( A, Bo, pi )

def _beta_scaled_prof( A, Bo, scale_factors ):
    """See _beta_scaled. This is a wrapper for the C version
    of the function."""
    return _hmm._hmm_beta_scaled( A, Bo, scale_factors )

def _ksi_prof( A, Bo, alpha, beta ):
    """See _ksi. This is a wrapper for the C version
    of the function."""
    return _hmm._hmm_ksi( A, Bo, alpha, beta )

def _update_iter_B_prof( gamma, obsIndices, B_bar ):
    """See _update_iter_B. This function is a wqrapper for the
    C version of _update_iter_B."""
    _hmm._hmm_update_iter_B( gamma, obsIndices, B_bar )

def _correctm_prof( M, k, v ):
    return _hmm._hmm_correctm( M, k, v )

def _normalize_B_prof( B_bar, sigma_gamma_B ):
    _hmm._hmm_normalize_B( B_bar, sigma_gamma_B )

class HMM_C_prof(HMM):
    AlphaScaled = _alpha_scaled_prof
    BetaScaled = _beta_scaled_prof
    Ksi = _ksi_prof            
    UpdateIterB = _update_iter_B_prof
    CorrectM = _correctm_prof
    NormalizeB = _normalize_B_prof
