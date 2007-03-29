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

from numpy import array, ones, zeros, cumsum, searchsorted, \
     reshape, add, allclose, floor, where, \
     product, sqrt, dot, multiply, alltrue, log, equal, newaxis, \
     take, empty_like, isfortran
from numpy.random import random
from exceptions import RuntimeError
import cPickle

# Display log likelyhood every DISPITER iterations while learning
DISPITER = 10


matrixproduct = dot

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
    T = Bo.shape[0]
    N = A.shape[0]
    alpha_t = Bo[0] * pi                # (19)
    scaling_factors = zeros( T, float )
    scaling_factors[0] = 1./add.reduce(alpha_t)
    alpha_scaled = zeros( (T, N), float)
    alpha_scaled[0] = alpha_t*scaling_factors[0]
    for i in xrange(1, T):
        alpha_t = dot(alpha_scaled[i-1], A)*Bo[i]  # (92a)
        scaling_t = 1./add.reduce(alpha_t)
        scaling_factors[i] = scaling_t
        alpha_scaled[i] = alpha_t*scaling_t      # (92b)
    return alpha_scaled, scaling_factors


def _beta_scaled( A, Bo, scale_factors ):
    """Computes backward probabilities
    beta(t,i)=P(O(t+1),...,O(T),Q(t)=Si|model)
    Or if scale_factors is not None:
    beta_scaled(t,i)=beta(t,i)*C(t) (From the result of _alpha_scaled)
    Bo is the same as in function _alpha
    """
    T,N = Bo.shape
    assert N == A.shape[0]
    scale_factors = scale_factors
    beta = zeros( (T, N), float )
    tmp = zeros( N, float )
    beta[-1] = ones( N, float ) * scale_factors[-1]         # (24)
    for t  in xrange( T-2, -1, -1 ):
        multiply( scale_factors[t], Bo[t+1], tmp )
        multiply( tmp, beta[t+1], tmp )
        beta[t] = matrixproduct( A, tmp )    # (25)
    return beta


def _ksi( A, Bo, alpha, beta ):
    """Compute ksi(t,i,j)=P(q_t=Si,q_(t+1)=Sj|model)"""
    N = A.shape[0]
    T = len(Bo)
    ksi = zeros( (T-1, N, N), float )
    tmp = Bo * beta
    for t in range(T-1):
        # This does transpose[alpha].(B[obs]*beta[t+1])
        # (where . is matrixproduct)
        ksit = ksi[t, :, :]
        multiply( A, tmp[t+1], ksit )
        multiply( ksit, alpha[t, :, newaxis], ksit )
        ksi_sum = add.reduce( ksit.flat )
        ksit /= ksi_sum
    return ksi

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


def _normalize_B( B_bar, sigma_gamma_B ):
    """Internal function.
    Normalize the estimations of matrix A.
    Make sure we get rid of lines that contains only zeroes."""
    sigma_gamma_B = 1./where( sigma_gamma_B, sigma_gamma_B, 1)
    B_bar *= sigma_gamma_B    # (110)


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

    AlphaScaled = staticmethod(_alpha_scaled)
    BetaScaled = staticmethod(_beta_scaled)
    Ksi = staticmethod(_ksi)
    UpdateIterB = staticmethod(_update_iter_B)
    CorrectM = staticmethod(_correctm)
    NormalizeB = staticmethod(_normalize_B)

    ORDER = "C"
    
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
        if transition_proba is None:
            transition_proba = ones( (self.N, self.N), float) / self.N
        if observation_proba is None:
            observation_proba = ones( (self.M, self.N), float) / self.M
        if initial_state_proba is None:
            initial_state_proba = ones( (self.N,), float ) / self.N

        self.A = array(transition_proba, float, order=self.ORDER)
        self.B = array(observation_proba, float, order=self.ORDER)
        self.pi = array(initial_state_proba, float, order=self.ORDER)
        # dimensional assertions
        self.checkHMM()
        self.makeIndexes()
        
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
            self.B = zeros( (self.M, self.N), float, self.ORDER )
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
        self.A /= add.reduce( self.A, 1 )[:, newaxis] # normalization

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
        indices = zeros( len(observations), int )
        k = 0
        for o in observations:
            indices[k] = self.O_index[o]
            k += 1
        return indices
    
    def simulate( self, length, show_hidden = 0 ):
        """generates a random sequence of observations of given length
        if show_hidden is true, returns a liste of (state,observation)"""
        cumA = cumsum( self.A, 1 )
        cumB = cumsum( self.B, 0 )
        r0 = random()
        state = searchsorted( cumsum(self.pi,0), r0)
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
        delta = zeros( N, float )
        tmp = zeros( N, float )
        delta = B[obs[0]] * self.pi    # (32a)
        delta_t = zeros( N, float )
        psi = zeros( (T, N), int )       # (32b)
        # recursion
        for t in xrange(1, T):
            O_t = obs[t]
            for j in range(N):
                multiply( delta, A[:, j], tmp )
                idx = psi[t, j] = tmp.argmax()        # (33b)
                delta_t[j] = tmp[idx] * B[O_t, j]  # (33a)
            delta, delta_t = delta_t, delta

        # reconstruction
        i_star = [delta.argmax()]                         # (34b)
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
        logB = zeros( (M, N), float)
        for i in xrange(M):
            t = B[i, :]
            k = equal( t, 0.0 ) * SMALLESTFLOAT
            logB[i] = log( k + t )
        # initialisation
        psi = zeros( N, float )
        psi_t = zeros( N, float )
        logPi = log( self.pi + equal( self.pi, 0.0 ) * SMALLESTFLOAT )
        add( logB[obs[0]], logPi, psi) # (105a)
        Q = zeros( (T, N), int )
        # recursion
        tmp = zeros( N, float )
        for t in xrange( 1, T ):
            O_t = obs[t]
            for j in xrange(N):
                tmp = psi + logA[:, j]
                idx = Q[t, j] = tmp.argmax()
                psi_t[j] = tmp[idx] + logB[O_t, j] # (105b)
            psi, psi_t = psi_t, psi

        # reconstruction
        q_star = [psi.argmax()]                         # (105c)
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
        logB = zeros( (M, N), float)
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
        return take( self.B, obsIndices, 0 )

    def _likelihood( self, scale_factors ):
        """This function computes the log likelihood
        of the training set using the precomputed
        alpha probabilities (sum(k=0..N,alpha(T,k)).
        It should increase during the learning process."""
        t = where( scale_factors==0.0, SMALLESTFLOAT, scale_factors )
        return -add.reduce( log(t) )

    def multiple_learn(self, m_observations,
                       states = None,
                       maxiter = 1000 ):
        """Uses Baum-Welch algorithm to learn the probabilities on multiple
        observations sequences
        """
        assert isfortran(self.B)
        # remove empty lists
        m_observations = filter( lambda x: x, m_observations )
        K = len( m_observations )
        learning_curve = []
        sigma_gamma_A = zeros( (self.N, ), float, order=self.ORDER )
        sigma_gamma_B = zeros( (self.N, ), float, order=self.ORDER )
        A_bar  = zeros( (self.N, self.N), float, order=self.ORDER )
        B_bar  = zeros( (self.M, self.N), float, order=self.ORDER )
        pi_bar = zeros( self.N, float, order=self.ORDER )
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
                Bo = take(self.B, obsIndices, 0)
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
            A_bar[...] = 0
            B_bar[...] = 0
            pi_bar[...] = 0
            sigma_gamma_A[...] = 0
            sigma_gamma_B[...] = 0
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
        Bo = take( B, obsIndices, 0 )
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
        A_bar *= sigma_gamma_A[:, newaxis]    # (109)


    def _final_step( self, gamma, ksi, obsIndices ):
        """Compute the new model, using gamma and ksi"""
        sigma_gamma = add.reduce(gamma[:-1])
        ## Compute new PI
        pi_bar = gamma[0]                       # (40a)

        ## Compute new A
        A_bar  = add.reduce(ksi)
        A_bar /= sigma_gamma[:, newaxis] # (40b)
        
        ## Compute new B
        B_bar = zeros( (self.M, self.N), float )
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
        return (allclose( self.A, A_bar, alpha_RTOL, alpha_ATOL) and 
               allclose( self.pi, pi_bar, alpha_RTOL, alpha_ATOL) and 
               allclose( self.B, B_bar, beta_RTOL, beta_ATOL))
    
    def _gamma(self, alpha, beta, scaling_factors ):
        """Compute gamma(t,i)=P(q_t=Si|model)"""
        g = alpha * beta / scaling_factors[:, newaxis]
        return g



    def normalize(self, P = None):
        """This can be used after a learning pass to
        reorder the states so that the s_i -> s_i transition
        probability are ordered s(0,0)>s(1,1) ... > s(n,n)

        the permutation of states can be passed as a parameter
        """
        if P is None:
            P = self.A.diagonal().argsort()
        A = empty_like(self.A)
        PI = empty_like(self.pi)
        B = empty_like(self.B)
        N = A.shape[0]
        for i in xrange(N):
            pi = P[i]
            for j in xrange(N):
                pj = P[j]
                A[i,j] = self.A[pi,pj]
            B[:,i] = self.B[:,pi]
            PI[i] = self.pi[pi]
        return A,B,PI
