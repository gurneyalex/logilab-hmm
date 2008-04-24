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
Applications in Speech Recognition_, by Lawrence Rabiner, IEEE, 1989,
 _Improved Estimation of Hidden Markov Model b.Parameters from Multiple 
Observation Sequences_, by Richard I. A. Davis, Brian C. Lovell, Terry Caelli,
 2002, _Improved Ensemble Training for Hidden Markov Models Using Random 
Relative Node Permutations_, by Richard I. A. Davis and Brian C. Lovell, 2003.
This module is an heritage of the module HMM and uses numeric python 
multyarrays to improve performance and reduce memory usage """

from numpy import zeros, add, allclose, newaxis, take
import logilab.hmm.hmm as hmm
import logilab.hmm.hmmc as hmmc
import logilab.hmm.hmmf as hmmf


# Display log likelyhood every DISPITER iterations while learning
DISPITER = 100
# mother class

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


class HMMSMixIn: 
    """A Hidden Markov Model implementation
    Methods are provided for computing the most probable state transitions 
    leading to a sequence of observations, as well as supervised and 
    unsupervised training of the model knowing states and generation of 
    sequences of observations (simulation).
    The notations (member variables and some method names), especially greek
    letter names are directly inspired by [Rabiner89] mentionned above.
    Comments in the source code mentionning a number are references to
    equations in the algorithm descriptions of that paper."""

    def _learn_A(self, states):
        """Train the transition probability matrix according to one 
        sequence of states"""
        T = len(states)
        N = self.N
        self.A = zeros((N, N))
        for k in range(T-1):
            Si = states[k]
            Sj = states[k+1]
            i = self.X_index[Si]
            j = self.X_index[Sj]
            self.A[i, j] += 1
        for i in range(N):
            if add.reduce(self.A, 1)[i]==0:
                self.A[i, i] = 1
        self.A *= 1./add.reduce(self.A, 1)[:, newaxis]

    def _multiple_learn_A(self, setStates):
        """Train the transition probability matrix according to a multiple 
        states sequences"""
        h = HMMS(self.omega_X, [1])
        A_bar = zeros((self.N, self.N), float)
        h.A = self.A
        for seq in setStates:
            h._learn_A(seq)
            A_bar += h.A
        self.A = A_bar / len(setStates)

    def learn( self, observations, states, maxiter = 1000, impr=1 ):
        """Train the model according to one sequence of observations and one
        sequence of states"""
        obs = self._get_observationIndices(observations)
        iter = self._baum_welch( obs, states, maxiter, impr )
        return iter

    def multiple_learn(self, m_observations, setStates,
                       maxiter = 1000, impr=1 ):
        """Uses Baum-Welch algorithm to learn the probabilities on multiple
        observations sequences and states sequences
        """
        # remove empty lists
        m_observations = filter( lambda x: x, m_observations )
        setO =  set()   # set of obsevations        
        K = len( m_observations )
        learning_curve = []
        sigma_gamma_B = zeros( (self.N, ), float, order=self.ORDER )
        A_bar  = zeros( (self.N, self.N), float, order=self.ORDER )
        B_bar  = zeros( (self.M, self.N), float, order=self.ORDER )
        pi_bar = zeros( self.N, float, order=self.ORDER )
        self._multiple_learn_A(setStates)
        if DISPITER == 0:
            dispiter = maxiter
        else:
            dispiter = DISPITER
        obs_list = []
        for k in range(K):
            observations = m_observations[k]
            obsIndices = self._get_observationIndices(observations)
            obs_list.append( obsIndices )
            setO = setO | set(observations)  # add new elements observed
        for iter in xrange( 1, maxiter + 1 ):
            total_likelihood = 0
            for k in range(K):
                obsIndices = obs_list[k]
                Bo = take(self.B, obsIndices, 0)
                alpha, scale_factors = self.alpha_scaled( self.A, Bo, self.pi )
                beta  = self.beta_scaled( self.A, Bo, scale_factors )
                gamma = self._gamma( alpha, beta, scale_factors )
                pi_bar += gamma[0]
                self._update_iter_gamma( gamma, sigma_gamma_B )
                self.update_iter_B( gamma, obsIndices, B_bar )
                total_likelihood += self._likelihood( scale_factors )
                
            #end for k in range(K)
            self.normalize_B( B_bar, sigma_gamma_B )
            pi_bar /= K
            # Correct A_bar and B_bar in case 0 probabilities slipped in
            self.correct_M( B_bar, 0, 1. / self.M )
            learning_curve.append( total_likelihood )
            if (iter % dispiter) == 0:
                if impr:
                    print "Iter ", iter, " log=", total_likelihood
            if self._stop_condition(pi_bar, B_bar ):
                if impr:
                    print 'Converged in %d iterations' % iter
                break
            
            self.B, B_bar   = B_bar, self.B
            self.pi, pi_bar = pi_bar, self.pi
            A_bar.fill(0)
            B_bar.fill(0)
            pi_bar.fill(0)
            sigma_gamma_B.fill(0)
        else:
            if impr:
                print "The Baum-Welch algorithm did not converge",
                print " in %d iterations" % maxiter
        self._mask()
        # Correct B in case 0 probabilities slipped in
        setO = set(self.omega_O) - setO
        while setO != set():
            e = setO.pop()
            e = self._get_observationIndices([e])
            self.B[e[0]] = 0
        return iter, learning_curve

    def _baum_welch( self, obsIndices, states, maxiter=1000, impr=1 ):
        """Uses Baum-Welch algorithm to learn the probabilities knowing the
         states Scaling on the forward and backward values is automatically
        performed when numerical problems (underflow) are encountered.
        Each iteration prints a dot on stderr, or a star if scaling was
        applied"""
        B  = self.B
        self._learn_A(states)
        A  = self.A
        pi = self.pi
        learning_curve = []
        if DISPITER == 0:
            dispiter = maxiter
        else:
            dispiter = DISPITER
        Bo = take( B, obsIndices, 0 )
        for iter in xrange( 1, maxiter + 1 ):
            alpha, scale_factors = self.alpha_scaled( A, Bo, pi )
            beta = self.beta_scaled( self.A, Bo, scale_factors )
            gamma = self._gamma( alpha, beta, scale_factors )
            B_bar, pi_bar = self._final_step( gamma, obsIndices )
            learning_curve.append( self._likelihood(scale_factors) )
            if impr:
                if (iter % dispiter) == 0:
                    print "Iter ", iter, " log=", learning_curve[-1]
            if self._stop_condition(pi_bar, B_bar):
                if impr:
                    print 'Converged in %d iterations' % iter
                break
            else:
                self.B = B = B_bar
                self.pi = pi = pi_bar
                self._mask()
        else:
            if impr:
                print "The Baum-Welch algorithm did not converge"
                print " in %d iterations" % maxiter
        return iter, learning_curve

    def _stop_condition( self, pi_bar, B_bar ):
        """Returns true if the difference between the estimated model
        and the current model is small enough that we can stop the
        learning process"""
        return (allclose( self.pi, pi_bar, alpha_RTOL, alpha_ATOL) and 
               allclose( self.B, B_bar, beta_RTOL, beta_ATOL))

    def _final_step( self, gamma, obsIndices ):
        """Compute the new model, using gamma"""
        sigma_gamma_B = add.reduce(gamma)
        for i in range(len(sigma_gamma_B)):
            if sigma_gamma_B[i] < EPSILON:
                sigma_gamma_B[i] = 1
        ## Compute new PI
        pi_bar = gamma[0]                       # (40a)
        
        ## Compute new B
        B_bar = zeros( (self.M, self.N), float )
        for i in xrange( len(obsIndices) ):
            B_bar[obsIndices[i]] += gamma[i]
        B_bar /= sigma_gamma_B
        return B_bar, pi_bar
    
    def _update_iter_gamma( self, gamma, sigma_gamma_B ):
        """update iter gamma"""
        sigma_gamma_k = add.reduce(gamma[:-1])
        sigma_gamma_B += sigma_gamma_k + gamma[-1]

    def ensemble_averaging(self, setObservations, setStates, 
                        weighting_factor="unit", maxiter=1000, impr=1):
        """Uses ensemble averaging method to learn the probabilities on 
        multiple observations sequences and states sequences"""
        N = self.N
        W = 0
        hmmk = self.__class__(self.omega_X, self.omega_O)
        A_bar = zeros( (N, N))
        B_bar = zeros( (self.M, N))
        pi_bar = zeros(N)
        for k, obs in enumerate(setObservations):
            hmmk.A = self.A
            hmmk.B = self.B
            hmmk.pi = self.pi
            obsIndices = self._get_observationIndices(obs)
            state = setStates[k]
            hmmk._baum_welch(obsIndices, state, maxiter, impr)
            if weighting_factor == "Pall":
                Wk = hmmk._weighting_factor_Pall(setObservations)
            elif weighting_factor == "Pk":
                Wk = hmmk._weighting_factor_Pk(obs)
            else:
                Wk = 1
            A_bar = A_bar + Wk * hmmk.A
            B_bar = B_bar + Wk * hmmk.B
            pi_bar = pi_bar + Wk * hmmk.pi
            W = W + Wk
        if W == 0:
            W = 1
            print "The ensemble averaging method did not converge" 
        else:
            self.A = A_bar / W
            self.B = B_bar / W
            self.pi = pi_bar / W
            self._mask()


class HMMS_C(HMMSMixIn, hmmc.HMM_C):
    pass

class HMMS_F(HMMSMixIn, hmmf.HMM_F):
    pass

class HMMS(HMMSMixIn, hmm.HMM):
    pass


