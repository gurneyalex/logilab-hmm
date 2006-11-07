# pylint: disable-msg=C0103, C0101
# -*- coding: ISO-8859-1 -*-

# Copyright (c) 2002 LOGILAB S.A. (Paris, FRANCE).
# http://www.logilab.fr/ -- mailto:contact@logilab.fr
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 2 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc.,
# 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.

"""Hidden Markov Models in Python
Implementation based on _A Tutorial on Hidden Markov Models and Selected
Applications in Speech Recognition_, by Lawrence Rabiner, IEEE, 1989.
This module uses numeric python multyarrays to improve performance and
reduce memory usage"""

__revision__ = "$Id: hmm.py,v 1.15 2005-02-25 20:40:35 nico Exp $"

from Numeric import array, Float, ones, zeros, cumsum, searchsorted, \
     argmax, multiarray, reshape, add, allclose, floor, where, \
     product, sqrt, dot, multiply, alltrue, log, Int, equal, NewAxis, \
     take, put
from RandomArray import random
from exceptions import RuntimeError
import cPickle

# Display log likelyhood every DISPITER iterations while learning
DISPITER = 10

# Force this to zero before calling HMM constructor if you want to
# use the python implementation of (alpha,beta,ksi...)
# Allowable values for use_hmm:
#  0 : use the python defined functions
#  1 : use the C optimized functions
#  2 : use the python wrapper for the C functions (for profiling purposes)
use__hmm = 1
try:
    import _hmm
except ImportError:
    try:
        import logilab.hmm._hmm
    except ImportError:
        use__hmm = 0
        print "Could not load _hmm, Falling back to python implementation"


matrixproduct = multiarray.matrixproduct

EPSILON = 1e-9
SMALLESTFLOAT = 1e-320
# These are the tolerances used with 'allclose()'
# For now we don't take the size of the model into
# account, but we really should. That's why you
# may want to tune this according to the size
# of your model.
# Since we use probability matrices the absolute
# tolerance will almost always come into play
# allclose(x,y) uses |x-y|<ATOL+RTOL*|y|
# because RTOL*|y| will be of the same order of magnitude
# as ATOL
# One more note, for some applications, the B matrix will be
# a lot larger than A. This is why we check A, and pi first
# and then B. This is also why you may want to tune tolerances
# so that alpha is less likely to trigger the test
alpha_ATOL = 1e-9
alpha_RTOL = 1e-6
beta_ATOL = 1e-8
beta_RTOL = 1e-5

## HMM Helper functions
## These functions are defined outside the HMM class
## because they should have the same prototype as their
## C counterpart

def _alpha_scaled(A, Bo, pi):
    """Internal method.
    Computes forward probabilities values, using a rescaling methods
    alpha_scaled[t,i]=Normalisation(P(O(1)...O(t),Q(t)=Si|model))
    Bo is the "slice" of the observation probability matrix corresponding
    to the observations (ie Bo=take(B,observation_indices)).
    For each t, c(t)=1./sum(alpha(t,i)), and C(t)=product(k=0..t,c(t))
    and alpha_scaled(t,i)=alpha(t,i)*C(t)
    The function returns: (alpha_scaled,C(t))
    """
    T = len(Bo)
    N = A.shape[0]
    alpha_t = Bo[0] * pi                # (19)
    scaling_factors = zeros( T, Float )
    scaling_factors[0] = 1./add.reduce(alpha_t)
    alpha_scaled = zeros( (T, N), Float)
    alpha_scaled[0] = alpha_t*scaling_factors[0]
    for i in xrange(1, T):
        alpha_t = matrixproduct(alpha_scaled[i-1], A)*Bo[i]  # (92a)
        scaling_t = 1./add.reduce(alpha_t)
        scaling_factors[i] = scaling_t
        alpha_scaled[i] = alpha_t*scaling_t      # (92b)
    return alpha_scaled, scaling_factors

def _alpha_scaled_prof(A, Bo, pi):
    """See _alpha_scaled. This is a wrapper for the C version
    of the function."""
    return _hmm._hmm_alpha_scaled( A, Bo, pi )

def _beta_scaled( A, Bo, scale_factors ):
    """Computes backward probabilities
    beta(t,i)=P(O(t+1),...,O(T),Q(t)=Si|model)
    Or if scale_factors is not None:
    beta_scaled(t,i)=beta(t,i)*C(t) (From the result of _alpha_scaled)
    Bo is the same as in function _alpha
    """
    T = len(Bo)
    N = A.shape[0]
    scale_factors = scale_factors
    beta = zeros( (T, N), Float )
    tmp = zeros( N, Float )
    beta[-1] = ones( N, Float ) * scale_factors[-1]         # (24)
    for t  in xrange( T-2, -1, -1 ):
        multiply( scale_factors[t], Bo[t+1], tmp )
        multiply( tmp, beta[t+1], tmp )
        beta[t] = matrixproduct( A, tmp )    # (25)
    return beta

def _beta_scaled_prof( A, Bo, scale_factors ):
    """See _beta_scaled. This is a wrapper for the C version
    of the function."""
    return _hmm._hmm_beta_scaled( A, Bo, scale_factors )

def _ksi( A, Bo, alpha, beta ):
    """Compute ksi(t,i,j)=P(q_t=Si,q_(t+1)=Sj|model)"""
    N = A.shape[0]
    T = len(Bo)
    ksi = zeros( (T-1, N, N), Float )
    tmp = Bo * beta
    for t in range(T-1):
        # This does transpose[alpha].(B[obs]*beta[t+1])
        # (where . is matrixproduct)
        ksit = ksi[t, :, :]
        multiply( A, tmp[t+1], ksit )
        multiply( ksit, alpha[t, :, NewAxis], ksit )
        ksi_sum = add.reduce( ksit.flat )
        ksit /= ksi_sum
    return ksi

def _ksi_prof( A, Bo, alpha, beta ):
    """See _ksi. This is a wrapper for the C version
    of the function."""
    return _hmm._hmm_ksi( A, Bo, alpha, beta )

def _clear( M ):
    """Returns a matrix the same size as M and of type Float.
    Could be the same matrix as M or a new one."""
    return zeros( M.shape, Float )

def _clear_prof( M ):
    """Returns a zeroed matrix the same size and type as M.
    This function is a wrapper around the C function, it actually
    returns the same reference if M is of type Float."""
    return _hmm._array_set( M )

def _update_iter_B( gamma, obsIndices, B_bar ):
    """Updates the estimation of the observations probabilities.
    This function is used during the learning process."""
    # Contrary to the equations in the paper from rabiner
    # For B we sum over all the observations.
    # We cannot do this for A because it doesn't make sense
    # But for B we need to take into account the last symbol
    # in the chain (If we don't it leads to such things as
    # the probability of a fullstop at the end of a sentence
    # is zero!!)
    for i in xrange(len(obsIndices)):     # (110) numerateur
        B_bar[obsIndices[i]] += gamma[i]

def _update_iter_B_prof( gamma, obsIndices, B_bar ):
    """See _update_iter_B. This function is a wqrapper for the
    C version of _update_iter_B."""
    _hmm._hmm_update_iter_B( gamma, obsIndices, B_bar )


def _correctm( M, k, p ):
    """This function is a hack. It looks for states with 0 probabilities, and
    changes this probability to a uniform probability. This avoids divide by zero
    errors, and doesn't change the result of the algorithm.
    You can only have 0 probabilities if your observation matrix contains symbols
    that don't appear in your observations AND the initial state transition and
    observation probabilities are such that a state is reachable only if you observe
    those symbols.
    Parameters are:
    M the matrix
    k the axis along which we need a pdf
    p the value to replace with (usually 1/M.shape[k])
    """
    D = equal( add.reduce( M, k ), 0.0)
    if k == 1:
        for i in xrange(M.shape[0]):
            if D[i]:
                M[i, :] = p
    elif k == 0:
        for i in xrange(M.shape[1]):
            if D[i]:
                M[:, i] = p
    else:
        raise "Not Implemented"
    return M

def _correctm_prof( M, k, v ):
    return _hmm._hmm_correctm( M, k, v )

def _normalize_B( B_bar, sigma_gamma_B ):
    """Internal function.
    Normalize the estimations of matrix A.
    Make sure we get rid of lines that contains only zeroes."""
    sigma_gamma_B = 1./where( sigma_gamma_B, sigma_gamma_B, 1)
    B_bar *= sigma_gamma_B    # (110)

def _normalize_B_prof( B_bar, sigma_gamma_B ):
    _hmm._hmm_normalize_B( B_bar, sigma_gamma_B )

## ----------------------------------------------------------------------

class HMM:
    """A Hidden Markov Model implementation
    Methods are provided for computing the probabitility of a sequence of
    observation, computing the most probable state transitions leading to
    a sequence of observations, as well as supervised and unsupervised
    training of the model and generation of sequences of observations
    (simulation).
    The notations (member variables and some method names), especially greek
    letter names are directly inspired by [Rabiner89] mentionned above.
    Comments in the source code mentionning a number are references to
    equations in the algorithm descriptions of that paper."""

    def __init__(self, state_list, observation_list,
                 transition_proba = None,
                 observation_proba = None,
                 initial_state_proba = None):
        """Builds a new Hidden Markov Model
        state_list is the list of state symbols [q_0...q_(N-1)]
        observation_list is the list of observation symbols [v_0...v_(M-1)]
        transition_proba is the transition probability matrix
            [a_ij] a_ij = Pr(X_(t+1)=q_j|X_t=q_i)
        observation_proba is the observation probablility matrix
            [b_ik] b_ik = Pr(O_t=v_k|X_t=q_i)
        initial_state_proba is the initial state distribution
            [pi_i] pi_i = Pr(X_0=q_i)"""
        self.N = len(state_list)
        self.M = len(observation_list)
        self.omega_X = state_list
        self.omega_O = observation_list
        if transition_proba is not None:
            self.A = array( transition_proba, Float )
        else:
            self.A = ones( (self.N, self.N), Float) / self.N
            
        if observation_proba is not None:
            self.B = array(observation_proba, Float)
        else:
            self.B = ones( (self.M, self.N), Float) / self.M
            
        if initial_state_proba is not None:
            self.pi = array( initial_state_proba, Float )
        else:
            self.pi = ones( (self.N,), Float ) / self.N

        # dimensional assertions
        self.checkHMM()
        self.makeIndexes()
        self.makeFuncObjects()

    def makeFuncObjects(self):
        """This methods defines function references according
        to the global variable use__hmm. Depending on its value
        the function references point to the python, C-implementation
        or python-C wrappers. This function is called by init so the
        implementation used depends on the value of use__hmm at the
        time the constructor is called."""
        if use__hmm == 1: ## Optimised version
            self.AlphaScaled = _hmm._hmm_alpha_scaled
            self.BetaScaled = _hmm._hmm_beta_scaled
            self.Ksi = _hmm._hmm_ksi
            self.Clear = _hmm._array_set
            self.UpdateIterB = _hmm._hmm_update_iter_B
            self.CorrectM = _hmm._hmm_correctm
            self.Allclose = _hmm._array_allclose
            self.NormalizeB = _hmm._hmm_normalize_B
        elif use__hmm == 2: ## Profiling version
            self.AlphaScaled = _alpha_scaled_prof
            self.BetaScaled = _beta_scaled_prof
            self.Ksi = _ksi_prof
            self.Clear = _clear_prof
            self.UpdateIterB = _update_iter_B_prof
            self.CorrectM = _correctm_prof
            self.Allclose = _hmm._array_allclose
            self.NormalizeB = _normalize_B_prof
        else: ## Default to python version
            self.AlphaScaled = _alpha_scaled
            self.BetaScaled = _beta_scaled
            self.Ksi = _ksi
            self.Clear = _clear
            self.UpdateIterB = _update_iter_B
            self.CorrectM = _correctm
            self.Allclose = allclose
            self.NormalizeB = _normalize_B
    
        
    def makeIndexes(self):
        """Creates the reverse table that maps states/observations names
        to their index in the probabilities array"""
        self.X_index = {}
        for i in range(self.N):
            self.X_index[self.omega_X[i]] = i
        self.O_index = {}
        for i in range(self.M):
            self.O_index[self.omega_O[i]] = i
            
    def saveHMM( self, f, saveState = None ):
        """Save the data for this class using cPickle.
        NOTE: don't use cPickle directly if your data uses
        too much memory. The pickle implementation of arrays
        just (well, not exactly) does a big binary copy of itself
        into a string and let pickle save the string object.
        So USE this function if your data becomes too big.
        As a side note, pickle will fail anyway because we have
        function objects as members of the HMM object. To use pickle
        you need to define __getattr__ __setattr__.
        """
        version = "HMM1.0"
        cPickle.dump( version, f, 1 )
        cPickle.dump( saveState, f, 1 )
        if saveState:
            cPickle.dump( self.omega_X, f, 1 )
            cPickle.dump( self.omega_O, f, 1 )
        cPickle.dump( self.N, f, 1 )
        cPickle.dump( self.M, f, 1 )
        cPickle.dump( self.A, f, 1 )
        cPickle.dump( self.pi, f, 1 )
        for i in xrange(self.M):
            cPickle.dump( self.B[i, :], f, 1 )
        
    def loadHMM( self, f ):
        """Use this function if you saved your data using
        saveHMM."""
        version = cPickle.load(f)
        if version == "HMM1.0":
            saveState = cPickle.load(f)
            if saveState:
                self.omega_X = cPickle.load(f)
                self.omega_O = cPickle.load(f)
            self.N = cPickle.load(f)
            self.M = cPickle.load(f)
            if saveState:
                self.makeIndexes()
            self.A = cPickle.load(f)
            self.pi = cPickle.load(f)
            self.B = zeros( (self.M, self.N), Float )
            for i in xrange(self.M):
                self.B[i, :] = cPickle.load(f)
        else:
            raise RuntimeError, "File format not recognized"

    def checkHMM(self):
        """This function will asserts if the internal state of the class
        is inconsistent. (Checks that the matrices' sizes are correct and
        that they represents probabilities)."""
        assert self.A.shape == (self.N, self.N), \
               """transition_proba must be a N*N matrix, where N is len(state_list)"""
        assert self.B.shape == (self.M, self.N), \
               """transition_proba must be a M*N matrix, where N is len(state_list)
               and M is len(observation_list)"""
        assert self.pi.shape == (self.N, ), \
               """transition_proba must be a N element vector,
               where N is len(state_list)"""
        reduced = add.reduce(self.A,1) - 1
        assert (alltrue(reduced < EPSILON) and \
                alltrue(reduced > -EPSILON) and \
                alltrue(alltrue(self.A<=1.0)) and \
                alltrue(alltrue(self.A>=0.0))),\
                """transition_proba must be a probability matrix"""
        reduced = add.reduce(self.B,0) - 1
        assert (alltrue(reduced < EPSILON) and \
                alltrue(reduced > -EPSILON) and \
                alltrue(alltrue(self.B<=1.0)) and \
                alltrue(alltrue(self.B>=0.0))),\
                """each column of observation_proba must be a probability
                vector"""
        if len(self.pi)==0: # a zero length vector is reduced to a scalar
            return          # and makes the following test fail
        reduced = add.reduce(self.pi) - 1
        assert (reduced < EPSILON and reduced > -EPSILON and \
                alltrue(self.pi<=1.0) and \
                alltrue(self.pi>=0.0)), \
                """initial_state_proba must be a probability vector"""


    def dump(self):
        """Helper method for debugging"""
        print "=" * 80
        print self
        print "States: ", self.N
        print "Observations: ", self.M
        print "-" * 80
        print "State transition probabilities:"
        print self.A
        print "-" * 80
        print "Observation probabilities:"
        print self.B
        print "-" * 80
        
    def __getinitargs__(self):
        """helper method for pickling"""
        return self.omega_X, self.omega_O, self.A, self.B, self.pi

    def setRandomProba(self):
        """Assigns random probability to all three matrices A, B and pi"""
        self.setRandomTransitionProba()
        self.setRandomObservationProba()
        self.setRandomInitialProba()

    def resetTransitionProba(self):
        """This resets the state transition matrix to zero. Use it
        only if you want to use setTransitionProba on some coefficients."""
        multiply( self.A, 0.0, self.A )

    def resetObservationProba(self):
        """This resets the observation matrix to zero. Use it
        only if you want to use setObservationProba on some coefficients."""
        multiply( self.B, 0.0, self.B )

    def resetInitialProba(self):
        """This resets the initial distribution matrix to zero. Use it
        only if you want to use setInitialProba on some coefficients."""
        multiply( self.pi, 0.0, self.pi )

    def setRandomTransitionProba(self):
        """set transition probability matrix to some random values"""
        self.A = random( self.A.shape )
        self.A /= add.reduce( self.A, 1 )[:, NewAxis] # normalization

    def setRandomObservationProba(self):
        """set observation probability matrix to some random values"""
        self.B = random( self.B.shape )
        self.B /= add.reduce( self.B ) # normalization

    def setRandomInitialProba(self):
        """set initial state probability matrix to some random values"""
        self.pi = random( self.pi.shape )
        self.pi /= add.reduce( self.pi ) # normalization

    def setTransitionProba( self, state1, state2, value ):
        """set the probability of a transition form 'state1' to 'state2'"""
        self.A[ self.X_index[state1], self.X_index[state2] ] = value

    def setObservationProba( self, state, obs, value ):
        """set the probability of generating observation 'obs'
        when in state 'state'"""
        self.B[ self.O_index[obs], self.X_index[state] ] = value

    def setInitialProba( self, state, value ):
        """set the probability of being initially in state 'state'"""
        self.pi[self.X_index[state]] = value

    def _getObservationIndices( self, observations ):
        """return observation indices"""
##        return [self.O_index[o] for o in observations]
        indices = zeros( len(observations), Int )
        k = 0
        for o in observations:
            indices[k] = self.O_index[o]
            k += 1
        return indices
    
    def simulate( self, length, show_hidden = 0 ):
        """generates a random sequence of observations of given length
        if show_hidden is true, returns a liste of (state,observation)"""
        cumA = cumsum( self.A, 1 )
        cumB = cumsum( self.B )
        r0 = random()
        state = searchsorted( cumsum(self.pi), r0)
        seq = []
        states = []
        
        for i in xrange(length):
            states.append(state)
            r1 = random()
            symbol = self.omega_O[ searchsorted( cumB[:, state], r1 ) ]
            if show_hidden:
                seq.append( (self.omega_X[state], symbol) )
            else:
                seq.append(symbol)
            r2 = random()
            state = searchsorted( cumA[state, :], r2 )
        return seq

    def analyze( self, observations ):
        """use Viterbi algorithm to
        find the states corresponding to the observations"""
        B = self.B
        A = self.A
        T = len(observations)
        N = self.N
        Omega_X = self.omega_X
        obs = self._getObservationIndices(observations)
        # initialisation
        delta = zeros( N, Float )
        tmp = zeros( N, Float )
        delta = B[obs[0]] * self.pi    # (32a)
        delta_t = zeros( N, Float )
        psi = zeros( (T, N), Int )       # (32b)
        # recursion
        for t in xrange(1, T):
            O_t = obs[t]
            for j in range(N):
                multiply( delta, A[:, j], tmp )
                idx = psi[t, j] = argmax(tmp)        # (33b)
                delta_t[j] = tmp[idx] * B[O_t, j]  # (33a)
            delta, delta_t = delta_t, delta

        # reconstruction
        i_star = [argmax(delta)]                         # (34b)
        for psi_t in psi[-1:0:-1]:
            i_star.append( psi_t[i_star[-1]] )                 # (35)
        trajectory = [Omega_X[i] for i in i_star]
        trajectory.reverse() # put time back in the right direction
        return trajectory
        
    def analyze_log( self, observations ):
        """use a modified Viterbi algorithm (using log P) to
        find the states corresponding to the observations."""
##      Since we use log(), we replace log(0) by the variable
##      hmm.SMALLESTFLOAT, change it if you find it not small enough. 
        B = self.B
        A = self.A
        T = len(observations)
        N = self.N
        M = self.M
        Omega_X = self.omega_X
        obs = self._getObservationIndices(observations)
        k = equal( A, 0.0 ) * SMALLESTFLOAT
        logA = log( A + k )
        logB = zeros( (M, N), Float)
        for i in xrange(M):
            t = B[i, :]
            k = equal( t, 0.0 ) * SMALLESTFLOAT
            logB[i] = log( k + t )
        # initialisation
        psi = zeros( N, Float )
        psi_t = zeros( N, Float )
        logPi = log( self.pi + equal( self.pi, 0.0 ) * SMALLESTFLOAT )
        add( logB[obs[0]], logPi, psi) # (105a)
        Q = zeros( (T, N), Int )
        # recursion
        tmp = zeros( N, Float )
        for t in xrange( 1, T ):
            O_t = obs[t]
            for j in xrange(N):
                tmp = psi + logA[:, j]
                idx = Q[t, j] = argmax(tmp)
                psi_t[j] = tmp[idx] + logB[O_t, j] # (105b)
            psi, psi_t = psi_t, psi

        # reconstruction
        q_star = [argmax(psi)]                         # (105c)
        for q_t in Q[-1:0:-1]:
            q_star.append( q_t[q_star[-1]] )                 # (35)
        trajectory = [Omega_X[i] for i in q_star]
        trajectory.reverse() # put time back in the right direction
        return trajectory

    def log_likelihood( self, observations, trajectory ):
        """return log_likelihood"""
        obs = self._getObservationIndices(observations)
        states = [ self.X_index[s] for s in trajectory ]
        res = 0
        N = self.N
        M = self.M
        logB = zeros( (M, N), Float)
        for i in xrange(M):
            t = self.B[i, :]
            k = equal(t, 0.0) * SMALLESTFLOAT
            logB[i] = log(k + t)
        for o, s in zip( obs, states ):
            res += logB[o, s]
        return res

    def learn( self, observations, states = None, maxiter = 1000 ):
        """Train the model according to one sequence of observations"""
        if states is None:
            obs = self._getObservationIndices(observations)
            iter = self._baumWelch( obs, maxiter )
        else:
            # FIXME: implement supervised learning algorithm
            # should not be too difficult
            pass
        return iter

    def _getObservations( self, obsIndices ):
        """Extract the lines of the observations probability matrix corresponding
        to the actual observations."""
        return take( self.B, obsIndices )

    def _likelihood( self, scale_factors ):
        """This function computes the log likelihood
        of the training set using the precomputed
        alpha probabilities (sum(k=0..N,alpha(T,k)).
        It should increase during the learning process."""
        return -add.reduce( log(scale_factors) )

    def multiple_learn(self, m_observations,
                       states = None,
                       maxiter = 1000 ):
        """Uses Baum-Welch algorithm to learn the probabilities on multiple
        observations sequences
        """
        # remove empty lists
        m_observations = filter( lambda x: x, m_observations )
        K = len( m_observations )
        learning_curve = []
        sigma_gamma_A = zeros( (self.N, ), Float )
        sigma_gamma_B = zeros( (self.N, ), Float )
        A_bar  = zeros( (self.N, self.N), Float )
        B_bar  = zeros( (self.M, self.N), Float )
        pi_bar = zeros( self.N, Float )
        if DISPITER == 0:
            dispiter = maxiter
        else:
            dispiter = DISPITER
        obs_list = []
        for k in range(K):
            observations = m_observations[k]
            obsIndices = self._getObservationIndices(observations)
            obs_list.append( obsIndices )
        for iter in xrange( 1, maxiter + 1 ):
            total_likelihood = 0
            for k in range(K):
                obsIndices = obs_list[k]
                Bo = take(self.B, obsIndices)
                alpha, scale_factors = self.AlphaScaled( self.A, Bo, self.pi )
                beta  = self.BetaScaled( self.A, Bo, scale_factors )
                ksi   = self.Ksi( self.A, Bo, alpha, beta )
                gamma = self._gamma( alpha, beta, scale_factors )
                pi_bar += gamma[0]
                self._update_iter_gamma( gamma, sigma_gamma_A, sigma_gamma_B )
                self._update_iter_A( ksi, A_bar )
                self.UpdateIterB( gamma, obsIndices, B_bar )
                total_likelihood += self._likelihood( scale_factors )
                
            #end for k in range(K)
            self._normalize_iter_A( A_bar, sigma_gamma_A )
            self.NormalizeB( B_bar, sigma_gamma_B )
            pi_bar /= K
            # Correct A_bar and B_bar in case 0 probabilities slipped in
            self.CorrectM( A_bar, 1, 1. / self.N )
            self.CorrectM( B_bar, 0, 1. / self.M )
            learning_curve.append( total_likelihood )
            if (iter % dispiter) == 0:
                print "Iter ", iter, " log=", total_likelihood
            if self._stop_condition( A_bar, pi_bar, B_bar ):
                print 'Converged in %d iterations' % iter
                break
            self.A, A_bar   = A_bar, self.A
            self.B, B_bar   = B_bar, self.B
            self.pi, pi_bar = pi_bar, self.pi
            A_bar = self.Clear( A_bar )
            B_bar = self.Clear( B_bar )
            pi_bar = self.Clear( pi_bar )
            sigma_gamma_A = self.Clear( sigma_gamma_A )
            sigma_gamma_B = self.Clear( sigma_gamma_B )
        else:
            print "The Baum-Welch algorithm had not converged in %d iterations" % maxiter
        return iter, learning_curve

    def _baumWelch( self, obsIndices, maxiter ):
        """Uses Baum-Welch algorithm to learn the probabilities
        Scaling on the forward and backward values is automatically
        performed when numerical problems (underflow) are encountered.
        Each iteration prints a dot on stderr, or a star if scaling was
        applied"""
        B  = self.B
        A  = self.A
        pi = self.pi
        learning_curve = []
        if DISPITER == 0:
            dispiter = maxiter
        else:
            dispiter = DISPITER
        Bo = take( B, obsIndices )
        for iter in xrange( 1, maxiter + 1 ):
            alpha, scale_factors = self.AlphaScaled( A, Bo, pi )
            beta = self.BetaScaled( self.A, Bo, scale_factors )
            ksi = self.Ksi( self.A, Bo, alpha, beta )
            gamma = self._gamma( alpha, beta, scale_factors )
            A_bar, B_bar, pi_bar = self._final_step( gamma, ksi, obsIndices )
            learning_curve.append( self._likelihood(scale_factors) )
            if (iter % dispiter) == 0:
                print "Iter ", iter, " log=", learning_curve[-1]
            if self._stop_condition( A_bar, pi_bar, B_bar):
                print 'Converged in %d iterations' % iter
                break
            else:
                self.A = A = A_bar
                self.B = B = B_bar
                self.pi = pi = pi_bar
        else:
            print "The Baum-Welsh algorithm did not converge in %d iterations" % maxiter
        return iter, learning_curve

    def _update_iter( self,
                      gamma, ksi, obsIndices,
                      sigma_gamma_A, sigma_gamma_B,
                      A_bar, B_bar, pi_bar ):
        """Internal function.
        Make an update to the current estimation of the matrices A,B,pi.
        The matrices are not normalized, and we keep track of normalization
        factors in sigma_gamma_[AB]. The matrices are updated in-place."""

    def _update_iter_gamma( self, gamma, sigma_gamma_A, sigma_gamma_B ):
        """update iter gamma"""
        sigma_gamma_kA = add.reduce(gamma[:-1])
        sigma_gamma_A += sigma_gamma_kA       # (109) et (110) denominateur
        sigma_gamma_B += sigma_gamma_kA + gamma[-1]

    def _update_iter_A( self, ksi, A_bar ):
        """update iter A"""
        A_bar_k = add.reduce( ksi )
        add( A_bar, A_bar_k, A_bar )           # (109) numerateur
    

    def _normalize_iter_A( self, A_bar, sigma_gamma_A ):
        """Internal function.
        Normalize the estimations of matrix A.
        Make sure we get rid of lines that contains only zeroes."""
        # replace 0 with 1 to avoid div error
        # it doesn't matter if it is one or anything else since
        # sigma_gamma(i)=0 implies A(i,:)=0 and B(i,:)=0
        sigma_gamma_A = 1. / where( sigma_gamma_A, sigma_gamma_A, 1 )
        A_bar *= sigma_gamma_A[:, NewAxis]    # (109)


    def _final_step( self, gamma, ksi, obsIndices ):
        """Compute the new model, using gamma and ksi"""
        sigma_gamma = add.reduce(gamma[:-1])
        ## Compute new PI
        pi_bar = gamma[0]                       # (40a)

        ## Compute new A
        A_bar  = add.reduce(ksi)
        A_bar /= sigma_gamma[:, NewAxis] # (40b)
        
        ## Compute new B
        B_bar = zeros( (self.M, self.N), Float )
        for i in xrange( len(obsIndices) - 1 ):
            B_bar[obsIndices[i]] += gamma[i]
        B_bar /= sigma_gamma
        return A_bar, B_bar, pi_bar

    def _final_step2( self, gamma, ksi, obsIndices, A_bar, B_bar, pi_bar ):
        ##_hmm._hmm_final_single( gamma, ksi, obsIndices, A_bar, B_bar, pi_bar )
        return A_bar, B_bar, pi_bar
        
    def _stop_condition( self, A_bar, pi_bar, B_bar ):
        """Returns true if the difference between the estimated model
        and the current model is small enough that we can stop the
        learning process"""
        return (self.Allclose( self.A, A_bar, alpha_RTOL, alpha_ATOL) and 
               self.Allclose( self.pi, pi_bar, alpha_RTOL, alpha_ATOL) and 
               self.Allclose( self.B, B_bar, beta_RTOL, beta_ATOL))
    
    def _gamma(self, alpha, beta, scaling_factors ):
        """Compute gamma(t,i)=P(q_t=Si|model)"""
        g = alpha * beta / scaling_factors[:, NewAxis]
        return g


def test1_analyze( h, chain ):
    """simple test"""
    print "Chain      : ", chain
    print "analyse    : ", h.analyze(chain)
    print "analyse_log: ", h.analyze_log(chain)
    
def test1():
    """Simple test, that will check if the viterbi algorithm
    correctly determine the hidden states of a HMM."""
    test = HMM(['a', 'b'], ['s1', 's2', 's3'],
               array([[.3, .7], [.5, .5]]),
               array([[.5, 0], [.5, .5], [0, .5]]),
               array([.9, .1]))
    test.dump()
    test1_analyze(test, ['s1'] * 3)
    test1_analyze(test, ['s1'] * 3 + ['s3'] * 3)
    test1_analyze(test, ['s1', 's2', 's3'] * 3)

    test.setRandomProba()
    test.dump()
    test1_analyze(test, ['s1'] * 3)
    test1_analyze(test, ['s1'] * 3 + ['s3'] * 3)
    test1_analyze(test, ['s1', 's2', 's3'] * 3)

def test2():
    """This test will display the computed likelyhood of some sentences given
    some predetermined transition and observation matrices"""
    nom = 'soleil ville parc chat chien jaune souris poule jardin bec griffe sel poulet poivre'.split()
    verbe = 'brillait mange chasse court dort griffe est ressemble'.split()
    adj = 'grand petit gentil endormi jaune grande petite gentille endormie'.split()
    adv = 'vigoureusement rapidement vite'.split()
    det = 'le la les un une des'.split()
    pro = 'je tu il elle on nous vous ils elles le la les lui moi toi eux'.split()
    pre = 'à pour sur sous près de du au avec sans'.split()
    univers = []
    for mot in nom + verbe + adj + adv + det + pro + pre:
        univers.append(mot)
    test = HMM(['adj', 'nom', 'verbe', 'adv', 'det', 'pro', 'pre'], univers)
    test.A[:,:] = 0.0 # clear transition proba
    test.setTransitionProba('det', 'adj', .5)
    test.setTransitionProba('det', 'nom', .5)
    test.setTransitionProba('nom', 'adj', .2)
    test.setTransitionProba('nom', 'verbe', .2)
    test.setTransitionProba('nom', 'nom', .2)
    test.setTransitionProba('nom', 'pro', .2)
    test.setTransitionProba('nom', 'adv', .1)
    test.setTransitionProba('nom', 'pre', .1)
    test.setTransitionProba('pro', 'adj', .2)
    test.setTransitionProba('pro', 'verbe', .2)
    test.setTransitionProba('pro', 'nom', .2)
    test.setTransitionProba('pro', 'pro', .2)
    test.setTransitionProba('pro', 'adv', .1)
    test.setTransitionProba('pro', 'pre', .1)
    test.setTransitionProba('adj', 'adj', .2)
    test.setTransitionProba('adj', 'nom', .6)
    test.setTransitionProba('adj', 'pre', .1)
    test.setTransitionProba('adj', 'verbe', .1)
    test.setTransitionProba('pre', 'det', .8)
    test.setTransitionProba('pre', 'nom', .2)
    test.setTransitionProba('verbe', 'verbe', .2)
    test.setTransitionProba('verbe', 'adv', .2)
    test.setTransitionProba('verbe', 'det', .3)
    test.setTransitionProba('verbe', 'pre', .3)
    test.setTransitionProba('adv', 'pre', .3)
    test.setTransitionProba('adv', 'verbe', .4)
    test.setTransitionProba('adv', 'det', .3)
    test.checkHMM()
    for liste, state in [ (nom, 'nom'), (verbe, 'verbe'), (adj, 'adj'),
                          (adv, 'adv'), (det, 'det'), (pro, 'pro'),
                          (pre, 'pre')]:
        taille = len(liste)
        proba = 1.0 / taille
        for mot in liste:
            test.setObservationProba(state, mot, proba)
        test.setInitialProba(state, 1. / 7)

    phrases = ('un grand soleil jaune brillait vigoureusement sur la ville endormie',
               'le petit chat mange des souris',
               'je mange du poulet au poivre sans sel',
               )
               
    for p in phrases:
        p = p.split()
        a = test.analyze(p)
        for i in range(len(p)):
            p[i] = (p[i], a[i])
        print p


def test3():
    """This test compares the results of some of the C and Python functions
    used by the algorithm, with the assumption that the C and Python functions
    should return the same results"""
    nom = 'soleil ville parc chat chien jaune souris poule jardin bec griffe sel poulet poivre'.split()
    verbe = 'brillait mange chasse court dort griffe est ressemble'.split()
    adj = 'grand petit gentil endormi jaune grande petite gentille endormie'.split()
    adv = 'vigoureusement rapidement vite'.split()
    det = 'le la les un une des'.split()
    pro = 'je tu il elle on nous vous ils elles le la les lui moi toi eux'.split()
    pre = 'à pour sur sous près de du au avec sans'.split()
    univers = []
    for mot in nom + verbe + adj + adv + det + pro + pre:
        univers.append(mot)
    test = HMM(['adj', 'nom', 'verbe', 'adv', 'det', 'pro', 'pre'], univers)

    test.setRandomProba()

    Obs = test._getObservationIndices('le petit chat mange des souris'.split())
    Bo = test._getObservations(Obs)
    r1, scale = _alpha_scaled(test.A, Bo, test.pi)
    r1_opt, scale_opt = _alpha_scaled_prof(test.A, Bo, test.pi)
    r2 = _beta_scaled(test.A, Bo, scale)
    r2_opt = _beta_scaled_prof(test.A, Bo, scale_opt)

    print "Alpha"
    print r1
    if  (allclose(r1_opt, r1)):
        print "Python and C algorithms returned the same results"
    else:
        print "There was significant differences between the results from C and python"
        print r1_opt - r1
    print "Maximum difference was:", max(max(abs(r1_opt - r1)))
    print "Beta"
    print r2
    if  (allclose(r2_opt, r2)):
        print "Python and C algorithms returned the same results"
    else:
        print "There was significant differences between the results from C and python"
        print r2_opt - r2
    print "Maximum difference was:", max(max(abs(r2_opt - r2)))
    
def test4():
    """A simple simulation test"""
    test = HMM(['a', 'b'], ['s1', 's2', 's3'],
               array([[.3, .7], [.5, .5]]),
               array([[.5, 0], [.5, .5], [0, .5]]),
               array([.9, .1]))
    test.dump()
    print test.simulate(10)
    print test.simulate(10, 1)

def test5():
    """Train a model over some simulated values from an initial
    model"""
    test = HMM(['a', 'b'], ['s1', 's2', 's3'])
    test.setRandomProba()
    print 'Original'
    print 'A =', test.A
    print 'B =', test.B
    print 'pi =', test.pi
    print
    print 'Generating sample data...'
    sample =  test.simulate(500)
    print 'Randomizing model...'
    test.setRandomProba()
    print 'Training model...'
    test.learn(sample, None, 3000)
    print 'trained values'
    print 'A =', test.A
    print 'B =', test.B
    print 'pi =', test.pi

def test6():
    """Same as test5 but with a bigger state space and observations values"""
    test = HMM(range(5), range(10))
    test.setRandomProba()
    print 'Original'
    print 'A =', test.A
    print 'B =', test.B
    print 'pi =', test.pi
    print
    print 'Generating sample data...'
    sample = test.simulate(10000)
    print 'Randomizing model...'
    test.setRandomProba()
    print 'Training model...'
    test.learn(sample, None, 10000)
    print 'trained values'
    print 'A =', test.A
    print 'B =', test.B
    print 'pi=', test.pi

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
    
def test8():
    """Same as test6 but learning over several observations from
    the same chain"""
    test = HMM(range(10), range(50))
    print 'Generating sample data...'
    l = []
    test.setRandomProba()
    for i in range(100):
        obs = test.simulate(100)
        l.append(obs)
    print 'Original'
    print 'A =', test.A
    print 'B =', test.B
    print 'pi =', test.pi
    print
    print 'Randomizing model...'
    test.setRandomProba()
    print 'Training model...'
    test.multiple_learn(l)
    print 'trained values'
    print 'A =', test.A
    print 'B =', test.B
    print 'pi =', test.pi


def deterministic_hmm():
    """Returns the matrices of a deterministic HMM"""
    test = HMM( ['a', 'b'], ['s1', 's2', 's3'],
                [[0.0, 1.0], [1.0, 0.0]],
                [[0.8, 0.0], [0.0, 0.8], [0.2, 0.2]], [0.7, 0.3])
    return test

def norm2(m):
    """Returns the norm2 of a matrix"""
    v = reshape(m, (product(m.shape), ))
    return sqrt(dot(v, v)) / product(m.shape)

def test9_errors(gene, test):
    """Compute an error (distance) between two chains"""
    error1 = norm2(gene.A - test.A)
    error2 = norm2(gene.B - test.B)
    error2bis = norm2(gene.B - test.B[:, ::-1])
    if error2 < error2bis:
        error3 = norm2( gene.pi - test.pi )
    else:
        error2 = error2bis
        error3 = norm2( gene.pi - test.pi[::-1] )
    return error1, error2, error3

def test9_display(errors):
    """Displays the computed errors"""
    for t in errors:
        print "Test ", t[0], "Iterations:", t[1], "ErrA:", t[2], "ErrB", t[3],
        print "ErrPi", t[4], "Avg time: ", t[6]
    
def test9(n=10):
    """This test generate a simple HMM (determinist state transitions)
    And check if the algoritm converge in less than 1000 iterations"""
    gene = deterministic_hmm()
    data = gene.simulate(500)
    test = HMM(['a', 'b'], ['s1', 's2', 's3'])
    errors = []
    from time import time
    for i in xrange(n):
        t1 = time()
        iteration, curve = test.learn(data)
        t2 = time()
        error1, error2, error3 = test9_errors( gene, test )
        print "A: ", test.A
        print "B: ", test.B
        errors.append([i, iteration, error1, error2, error3,
                       curve, (t2 - t1) / iteration])
        test.setRandomProba()
    test9_display(errors)
    return errors

def test10(n=10):
    """This test generate a simple HMM (determinist state transitions)
    And check if the algoritm converge in less than 1000 iterations"""
    gene = deterministic_hmm()
    print "Generating data..."
    data = [ gene.simulate(20) for i in range(100) ]
    test = HMM(['a', 'b'], ['s1', 's2', 's3'])
    errors = []
    for i in xrange(n):
        print "round ", i
        iteration, curve = test.multiple_learn(data)
        error1, error2, error3 = test9_errors( gene, test )
        print "A: ", test.A
        print "B: ", test.B
        print "Pi:", test.pi
        errors.append([i, iteration, error1, error2, error3, curve, 0])
        test.setRandomProba()
    test9_display(errors)
    return errors, test
    
if __name__ == '__main__':
    for test_num in range(1, 11):
        print '-' * 80
        print 'start test', test_num
        globals()['test%s' % test_num]()

