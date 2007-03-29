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
    
    AlphaScaled = staticmethod(_hmm._hmm_alpha_scaled)
    BetaScaled = staticmethod(_hmm._hmm_beta_scaled)
    Ksi = staticmethod(_hmm._hmm_ksi)
    UpdateIterB = staticmethod(_hmm._hmm_update_iter_B)
    CorrectM = staticmethod(_hmm._hmm_correctm)
    NormalizeB = staticmethod(_hmm._hmm_normalize_B)


class HMM_C_prof(HMM):
    def AlphaScaled(self, A, Bo, pi):
        """See _alpha_scaled. This is a wrapper for the C version
        of the function."""
        return _hmm._hmm_alpha_scaled( A, Bo, pi )

    def BetaScaled( self, A, Bo, scale_factors ):
        """See _beta_scaled. This is a wrapper for the C version
        of the function."""
        return _hmm._hmm_beta_scaled( A, Bo, scale_factors )

    def Ksi( self, A, Bo, alpha, beta ):
        """See _ksi. This is a wrapper for the C version
        of the function."""
        return _hmm._hmm_ksi( A, Bo, alpha, beta )

    def UpdateIterB( self, gamma, obsIndices, B_bar ):
        """See _update_iter_B. This function is a wqrapper for the
        C version of _update_iter_B."""
        _hmm._hmm_update_iter_B( gamma, obsIndices, B_bar )

    def CorrectM( self, M, k, v ):
        return _hmm._hmm_correctm( M, k, v )

    def NormalizeB( self, B_bar, sigma_gamma_B ):
        _hmm._hmm_normalize_B( B_bar, sigma_gamma_B )
