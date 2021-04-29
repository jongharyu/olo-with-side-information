from collections import defaultdict

import numpy as np
from scipy.special import betaln, logsumexp

from ctw import ContextTree


class LinearRegressionWithAbsoluteLoss:
    def __init__(self):
        pass

    @staticmethod
    def compute_loss(w, data):
        x, y = data
        return np.abs(w @ x - y)

    @staticmethod
    def compute_subgradient(w, data):
        r"""
        Compute the subgradient of the absolute function with linear regression

        .. math::
            l(w) = \abs{\langle w, x\rangle - y}

        Parameters
        ----------
        w: weight vector
        data: (feature, response)

        Returns
        -------
        g: subgradient
        """
        x, y = data
        return -x if w @ x < y else x


class OnlineConvexOptimizer:
    def __init__(self, dim, problem):
        """
        Subgradient based online convex optimization

        Parameters
        ----------
        dim
        """
        self.dim = dim

        assert hasattr(problem, 'compute_loss') and hasattr(problem, 'compute_subgradient')
        self.problem = problem
        self.compute_loss = problem.compute_loss
        self.compute_subgradient = problem.compute_subgradient

        # initialize statistics
        self.w = np.zeros(self.dim)
        self.cum_loss = 0
        self.cum_reward = 0
        self._losses = []  # list of losses
        self._lin_losses = []  # list of linearized losses

    def reset_stats(self):
        self.w = np.zeros(self.dim)
        self.cum_loss = 0
        self.cum_reward = 0
        self._losses = []
        self._lin_losses = []
        return self

    def get_side_information(self, data=None):
        return None

    def get_action(self, h_t=None):
        raise NotImplementedError

    def update(self, g_t, h_t=None, data=None):
        return self.update_losses(g_t, data)

    def update_losses(self, g_t, data):
        # compute and store losses
        loss = self.compute_loss(self.w, data)
        lin_loss = self.w @ g_t

        self._losses.append(loss)
        self.cum_loss += loss
        self._lin_losses.append(lin_loss)
        self.cum_reward -= lin_loss
        return self

    def fit_single(self, data):
        # get side information
        h_t = self.get_side_information(data)

        # get action
        self.w = self.get_action(h_t)

        # compute subgradient
        g_t = self.compute_subgradient(self.w, data)

        # update
        self.update(g_t, h_t, data)

        return self

    def fit(self, X, y):
        for i in range(len(X)):
            self.fit_single((X[i], y[i]))

        return self

    @property
    def losses(self):
        return np.array(self._losses)

    @property
    def lin_losses(self):
        return np.array(self._lin_losses)


class OnlineGradientDescent(OnlineConvexOptimizer):
    def __init__(self, dim, lr_scale, problem=LinearRegressionWithAbsoluteLoss()):
        super().__init__(dim, problem)

        # hyperparameters
        self.lr_scale = lr_scale

        # initialize variables
        self.g_prev = None

    def lr(self, t):
        # time varying learning rate
        return self.lr_scale / np.sqrt(t)

    def get_action(self, h_t=None):
        t = len(self._losses) + 1
        if t > 1:
            assert self.g_prev is not None
            return self.w - self.lr(t) * self.g_prev  # no projection since unconstrained
        else:
            return self.w

    def update(self, g_t, h_t=None, data=None):
        super().update(g_t, h_t, data)
        self.g_prev = g_t
        return self


class DimensionFreeExponentiatedGradient(OnlineConvexOptimizer):
    def __init__(self, dim, a=1, L=1, delta=1, problem=LinearRegressionWithAbsoluteLoss()):
        """
        Dimension-free Exponentiated Gradient (DFEG)

        References
        ----------
        [1] ...

        Parameters
        ----------
        dim
        a
        delta
        L: Lipschitz constant
        problem
        """
        super().__init__(dim, problem)

        # hyperparameters
        self.a = a
        self.delta = delta
        self.L = L

        # initialize variables
        self.th = np.zeros(self.dim)
        self.H = self.delta

    def reset_stats(self):
        super().reset_stats()
        self.th = np.zeros(self.dim)
        self.H = self.delta
        return self

    def get_action(self, h_t=None):
        # Set parameters
        self.H += self.L ** 2  # assumed ||x_t|| = 1; TODO: check if this requires to set normalize=True
        alpha = self.a * np.sqrt(self.H)
        beta = self.H ** (3 / 2)

        # Compute w
        norm_th = np.linalg.norm(self.th)
        if norm_th == 0:
            w = np.zeros(self.dim)
        else:
            w = (self.th / norm_th) * (np.exp(norm_th / alpha) / beta)  # the EG step

        return w

    def update(self, g_t, h_t=None, data=None):
        super().update(g_t, h_t, data)
        self.th -= g_t
        return self


class AdaptiveNormal(OnlineConvexOptimizer):
    def __init__(self, dim, a=1, L=1, eps=1, problem=LinearRegressionWithAbsoluteLoss()):
        """
        Adaptive Normal (AdaNormal)

        References
        ----------
        [1] ...

        Parameters
        ----------
        dim
        a
        L: Lipschitz constant
        eps
        problem
        """
        super().__init__(dim, problem)

        # hyperparameters
        self.a = a
        self.eps = eps
        self.L = L

        # initialize variables
        self.th = np.zeros(dim)

    def reset_stats(self):
        super().reset_stats()
        self.th = np.zeros(self.dim)
        return self

    def get_action(self, h_t=None):
        t = len(self._losses) + 1

        # Set action
        norm_th = np.linalg.norm(self.th)
        if norm_th == 0:
            w = np.zeros(self.dim)
        else:
            term1 = np.exp(((norm_th + self.L) ** 2) / (2 * self.a * (t + 1)))
            term1 -= np.exp(((norm_th - self.L) ** 2) / (2 * self.a * (t + 1)))
            term2 = (2 * self.L * (np.log(t + 2)) ** 2) ** (-1)
            w = self.th * self.eps * term1 * term2 / norm_th  # the AdaNormal step

        return w

    def update(self, g_t, h_t=None, data=None):
        super().update(g_t, h_t, data)
        self.th -= g_t
        return self


class Quantizer:
    def __init__(self, quantizer_vector):
        self.quantizer_vector = quantizer_vector

    def __call__(self, g):
        return None if self.quantizer_vector is None else \
            np.sign(self.quantizer_vector @ g + 1e-10).astype(int)


class CoinBetting(OnlineConvexOptimizer):
    def __init__(self, dim, init_wealth=1, quantizer=None, problem=LinearRegressionWithAbsoluteLoss()):
        """
        A placeholder for coin betting with side information

        Parameters
        ----------
        init_wealth
        quantizer
        """
        super().__init__(dim=dim, problem=problem)

        # hyperparameter
        self.init_wealth = init_wealth
        self.quantizer = quantizer

        # initialize parameters
        self._log_potential = 0

    def reset_stats(self):
        super().reset_stats()
        self._log_potential = 0
        return self

    def get_vector_betting(self, h_t):
        raise NotImplementedError

    def get_action(self, h_t=None):
        return self.get_vector_betting(h_t) * self.wealth

    @property
    def wealth(self):
        return self.init_wealth + self.cum_reward

    @property
    def log_potential(self):
        return self._log_potential


class KTCoinBetting(CoinBetting):
    def __init__(self, dim, init_wealth=1, quantizer=None, problem=LinearRegressionWithAbsoluteLoss()):
        super().__init__(dim, init_wealth=init_wealth, quantizer=quantizer, problem=problem)

        # internal statistics
        self.counter = defaultdict(int)
        self.g_cum = defaultdict(lambda: np.zeros(dim))  # cumulative gradients

    def reset_stats(self):
        super().reset_stats()
        self.counter = defaultdict(int)
        self.g_cum = defaultdict(lambda: np.zeros(self.dim))
        return self

    def get_vector_betting(self, h_t):
        return self.g_cum[h_t] / (self.counter[h_t] + 1)  # vectorial KT betting

    def update(self, g_t, h_t=None, data=None):
        super().update(g_t, h_t, data)
        # update log_potential, counter, cumulative gradient
        self.counter[h_t] += 1
        self.g_cum[h_t] += -g_t  # accumulate -g since we are in a loss minimization framework
        self._log_potential = self.compute_log_kt_potential(h_t)  # \psi(g^{t}(h_t))
        return self

    def compute_log_kt_potential(self, h_t):
        # return log KT potential with side information
        t = self.counter[h_t]
        x = np.sqrt((self.g_cum[h_t] ** 2).sum())  # l2 norm of \sum g^{t-1}(h_t)
        return t * np.log(2) + betaln((t + x + 1) / 2, (t - x + 1) / 2) - betaln(.5, .5)


class KTCoinBettingWithMarkovSideInformation(KTCoinBetting):
    def __init__(self, dim, init_wealth=1, depth=1, quantizer=None, problem=LinearRegressionWithAbsoluteLoss()):
        """
        Implement coin betting with a fixed order Markov type side information with a binary quantizer

        Parameters
        ----------
        depth
        """
        super().__init__(dim, init_wealth, quantizer, problem=problem)

        # hyperparameter
        self.depth = depth

        # internal statistics
        self.suffix = [1 for _ in range(self.depth)]  # initialize quantized suffix [Q(g_{t-d}) for d in [1,...,D]]

    def reset_stats(self):
        super().reset_stats()
        self.suffix = [1 for _ in range(self.depth)]  # initialize quantized suffix [Q(g_{t-d}) for d in [1,...,D]]
        return self

    def get_side_information(self, data=None):
        return tuple(self.suffix)  # h_t = Q(g_{t-1}^{t-D})

    def update(self, g_t, h_t=None, data=None):
        super().update(g_t, h_t, data)
        suffix = self.suffix[1:] + [self.quantizer(g_t)]  # update the suffix
        self.suffix = suffix[:self.depth]  # to handle the degenerate case when depth=0
        return self


class KTCoinBettingWithHint(KTCoinBetting):
    def __init__(self, dim, init_wealth=1, quantizer=None, problem=LinearRegressionWithAbsoluteLoss()):
        """
        Coin betting with a fixed order Markov type side information with a binary quantizer
        """
        super().__init__(dim, init_wealth, quantizer, problem=problem)

    def get_side_information(self, data=None):
        return self.quantizer(self.compute_subgradient(self.w, data))


class ContextTreeWeighting(CoinBetting):
    def __init__(self, dim, init_wealth=1, max_depth=1, alpha=.5, quantizer=None, problem=LinearRegressionWithAbsoluteLoss()):
        super().__init__(dim, init_wealth, quantizer, problem=problem)

        # hyperparameters
        self.alpha = alpha
        self.max_depth = max_depth

        # internal statistics
        self.context_tree = ContextTree(max_depth=self.max_depth, alpha=self.alpha, dim=dim)
        self.suffix = [1 for _ in range(self.max_depth)]  # initialize quantized suffix [Q(g_{t-d}) for d in [1,...,D]]

    def reset_stats(self):
        super().reset_stats()
        self.context_tree = ContextTree(max_depth=self.max_depth, alpha=self.alpha, dim=self.dim)
        self.suffix = [1 for _ in range(self.max_depth)]  # initialize quantized suffix [Q(g_{t-d}) for d in [1,...,D]]
        return self

    def get_side_information(self, data=None):
        return tuple(self.suffix)  # h_t = Q(g_{t-1}^{t-D})

    def get_vector_betting(self, suffix):
        return self.context_tree.v_ctw(state=suffix)

    def update(self, g_t, h_t=None, data=None):
        super().update(g_t, h_t, data)
        self.context_tree.update(state=h_t, g_new=-g_t)
        suffix = self.suffix[1:] + [self.quantizer(g_t)]  # update the suffix
        self.suffix = suffix[:self.max_depth]  # to handle the degenerate case when max_depth=0
        return self

    @property
    def log_potential(self):
        return self.context_tree.log_potential


class CombineAlgorithms(OnlineConvexOptimizer):
    def __init__(self, dim, algorithms):
        super().__init__(dim, algorithms[0].problem)
        self.algorithms = algorithms
        self.reset_stats()

    def reset_stats(self):
        for algorithm in self.algorithms:
            algorithm.reset_stats()
        return self

    def get_side_information(self, data=None):
        h_ts = []
        for i, algorithm in enumerate(self.algorithms):
            h_t = algorithm.get_side_information(data)
            h_ts.append(h_t)
        return h_ts

    def get_action(self, h_ts=None):
        raise NotImplementedError

    def update(self, g_t, h_ts=None, data=None):
        super().update(g_t, h_ts, data)
        # update individual algorithms
        for i, algorithm in enumerate(self.algorithms):
            algorithm.update(g_t, h_ts[i], data)  # update losses based on each algorithm's action
            # Note: Addition depends on the losses of the constituent algorithms, but
            #       Mixture only requires each algorithm to maintain side information correctly
        return self


class Addition(CombineAlgorithms):
    def __init__(self, dim, algorithms):
        super().__init__(dim, algorithms)

    def get_action(self, h_ts=None):
        w = np.zeros(self.dim)
        for i, algorithm in enumerate(self.algorithms):
            # the correct implementation
            algorithm.w = algorithm.get_action(h_ts[i])  # set action for each algorithm
            w += algorithm.w

        return w


class AddingVectorBettings(CombineAlgorithms):
    def __init__(self, dim, algorithms, init_wealth=1):
        super().__init__(dim, algorithms)
        for algorithm in algorithms:
            assert hasattr(algorithm, 'get_vector_betting')

        self.init_wealth = init_wealth

    def get_vector_betting(self, h_ts=None):
        v = np.zeros(self.dim)
        for i, algorithm in enumerate(self.algorithms):
            # TODO: somehow it works very well?
            v += algorithm.get_vector_betting(h_ts[i])

        return v

    @property
    def wealth(self):
        return self.init_wealth  # + self.cum_reward

    def get_action(self, h_ts=None):
        return self.wealth * self.get_vector_betting(h_ts)


class Mixture(CombineAlgorithms):
    def __init__(self, dim, algorithms, init_wealth=1, weights=None):
        super().__init__(dim, algorithms)
        for algorithm in algorithms:
            assert hasattr(algorithm, 'log_potential')
            assert hasattr(algorithm, 'get_vector_betting')

        self.init_wealth = init_wealth
        if weights is None:
            weights = np.ones(len(self.algorithms))
        self.log_weights = np.log(weights)

    def get_vector_betting(self, h_ts=None):
        vs = []
        log_weighted_potentials = []  # log(w_m * psi_m)

        for i, algorithm in enumerate(self.algorithms):
            # v = algorithm.get_vector_betting(h_ts[i])
            # algorithm.w = algorithm.wealth * v
            # vs.append(v)
            vs.append(algorithm.get_vector_betting(h_ts[i]))
            log_weighted_potentials.append(self.log_weights[i] + algorithm.log_potential)

        log_potential_mixture = logsumexp(log_weighted_potentials)
        weights = np.exp(np.array(log_weighted_potentials) - log_potential_mixture)
        v_mixture = np.sum(weights[:, np.newaxis] * np.array(vs), axis=0)
        # v_mixture = np.sum(np.array(vs), axis=0) / len(self.algorithms)
        return v_mixture

    @property
    def wealth(self):
        return self.init_wealth + self.cum_reward

    def get_action(self, h_ts=None):
        return self.wealth * self.get_vector_betting(h_ts)
