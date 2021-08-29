from collections import defaultdict
from typing import List

import numpy as np
from scipy.special import betaln, logsumexp

from context_tree_base import ContextTreeWeightingOLOBase, ContextTreeAdditionOLOBase
from problems import ConvexProblem, Portfolio
from side_information import SideInformation, NoSideInformation, QuantizedHint, MarkovSideInformation, MutlipleSideInformation


class OnlineConvexOptimizer:
    def __init__(self, dim, init_w=None,
                 problem: ConvexProblem = None, side_information: SideInformation = None):
        """
        Subgradient based online convex optimization (with optional side information)

        Parameters
        ----------
        dim
        """
        self.dim = dim

        assert hasattr(problem, 'compute_loss') and hasattr(problem, 'compute_subgradient')
        self.problem = problem

        self.side_information = side_information if side_information else NoSideInformation()
        self.init_w = np.zeros(self.dim) if init_w is None else init_w

        # initialize statistics
        self.w = None
        self.cum_loss = None
        self.cum_reward = None
        self._losses = None
        self._lin_losses = None

        self.reset()

    def reset(self):
        self.w = self.init_w.copy()
        self.cum_loss = 0
        self.cum_reward = 0
        self._losses = []  # list of losses
        self._lin_losses = []  # list of linearized losses
        self.side_information.reset()
        return self

    def get_action(self, h_t=None):
        raise NotImplementedError

    def update(self, g_t, h_t=None, data=None):
        self.side_information.update(g_t)
        return self._update_losses(g_t, data)

    def _update_losses(self, g_t, data):
        # compute and store losses
        loss = self.problem.compute_loss(self.w, data)
        lin_loss = self.w @ g_t

        self._losses.append(loss)
        self.cum_loss += loss
        self._lin_losses.append(lin_loss)
        self.cum_reward -= lin_loss

        return self

    def fit_single(self, data):
        # get side information
        if isinstance(self.side_information, QuantizedHint):
            h_t = self.side_information.get(self.problem.compute_subgradient(self.w, data))
        else:
            h_t = self.side_information.get()

        # get action
        self.w = self.get_action(h_t)  # Warning: this is a crucial step!
        if isinstance(self.problem, Portfolio):
            self.w /= self.w.sum()

        # compute subgradient
        g_t = self.problem.compute_subgradient(self.w, data)

        # update
        self.update(g_t, h_t, data)

        return self

    def fit(self, data):
        for i in range(len(data[0])):
            self.fit_single([data_[i] for data_ in data])
        return self

    @property
    def losses(self):
        return np.array(self._losses)

    @property
    def lin_losses(self):
        return np.array(self._lin_losses)


class OnlineGradientDescent(OnlineConvexOptimizer):
    def __init__(self, dim, lr_scale, init_w=None,
                 problem: ConvexProblem = None, side_information: SideInformation = None):
        # hyperparameters
        self.lr_scale = lr_scale

        # initialize variables
        self.counter = None
        self.g_cum = None

        super().__init__(dim, init_w, problem, side_information)

    def reset(self):
        super().reset()
        self.counter = defaultdict(int)
        self.g_cum = defaultdict(lambda: self.init_w.copy())  # weighted cumulative gradients
        return self

    def lr(self, t):
        # time varying learning rate
        return self.lr_scale / np.sqrt(t)

    def get_action(self, h_t=None):
        return self.g_cum[h_t]

    def update(self, g_t, h_t=None, data=None):
        super().update(g_t, h_t, data)
        self.counter[h_t] += 1
        self.g_cum[h_t] -= self.lr(self.counter[h_t]) * g_t  # no projection since unconstrained
        return self


class DimensionFreeExponentiatedGradient(OnlineConvexOptimizer):
    def __init__(self, dim, init_w=None,
                 problem: ConvexProblem = None, side_information: SideInformation = None,
                 a=1, L=1, delta=1):
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
        # hyperparameters
        self.a = a
        self.delta = delta
        self.L = L

        # initialize variables
        self.th = None
        self.H = None

        super().__init__(dim, init_w, problem, side_information)

    def reset(self):
        super().reset()
        self.th = defaultdict(lambda: self.init_w.copy())
        self.H = defaultdict(lambda: self.delta)
        return self

    def get_action(self, h_t=None):
        # Set parameters
        self.H[h_t] += self.L ** 2  # assumed ||x_t|| = 1
        alpha = self.a * np.sqrt(self.H[h_t])
        beta = self.H[h_t] ** (3 / 2)

        # Compute w
        norm_th = np.linalg.norm(self.th[h_t])
        if norm_th == 0:
            w = np.zeros(self.dim)
        else:
            w = (self.th[h_t] / norm_th) * (np.exp(norm_th / alpha) / beta)  # the EG step

        return w

    def update(self, g_t, h_t=None, data=None):
        super().update(g_t, h_t, data)
        self.th[h_t] -= g_t
        return self


class AdaptiveNormal(OnlineConvexOptimizer):
    def __init__(self, dim, init_w=None,
                 problem: ConvexProblem = None, side_information: SideInformation = None,
                 a=1, L=1, eps=1):
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
        # hyperparameters
        self.a = a
        self.eps = eps
        self.L = L

        # initialize variables
        self.th = None
        self.counter = None

        super().__init__(dim, init_w, problem, side_information)

    def reset(self):
        super().reset()
        self.th = defaultdict(lambda: self.init_w.copy())
        self.counter = defaultdict(int)
        return self

    def get_action(self, h_t=None):
        t = self.counter[h_t] + 1

        # Set action
        norm_th = np.linalg.norm(self.th[h_t])
        if norm_th == 0:
            w = np.zeros(self.dim)
        else:
            term1 = np.exp(((norm_th + self.L) ** 2) / (2 * self.a * (t + 1)))
            term1 -= np.exp(((norm_th - self.L) ** 2) / (2 * self.a * (t + 1)))
            term2 = (2 * self.L * (np.log(t + 2)) ** 2) ** (-1)
            w = self.th[h_t] * self.eps * term1 * term2 / norm_th  # the AdaNormal step

        return w

    def update(self, g_t, h_t=None, data=None):
        super().update(g_t, h_t, data)
        self.th[h_t] -= g_t
        self.counter[h_t] += 1
        return self


class CoinBettingOLO(OnlineConvexOptimizer):
    def __init__(self, dim, init_w=None,
                 problem: ConvexProblem = None, side_information: SideInformation = None,
                 init_wealth=1):
        """
        A placeholder for coin betting with side information

        Parameters
        ----------
        init_wealth
        """
        # hyperparameter
        self.init_wealth = init_wealth

        # initialize parameters
        self._log_potential = None

        super().__init__(dim, init_w=init_w, problem=problem, side_information=side_information)

    def reset(self):
        super().reset()
        self._log_potential = 0
        return self

    def get_vector_betting(self, h_t):
        raise NotImplementedError

    def get_action(self, h_t=None):
        return self.init_w + self.get_vector_betting(h_t) * self.wealth

    @property
    def wealth(self):
        return self.init_wealth + self.cum_reward

    @property
    def log_potential(self):
        return self._log_potential


class KTOLO(CoinBettingOLO):
    def __init__(self, dim, init_w=None,
                 problem: ConvexProblem = None, side_information: SideInformation = None,
                 init_wealth=1):
        # internal statistics
        self.counter = None
        self.g_cum = None

        super().__init__(dim, init_w=init_w, problem=problem, side_information=side_information, init_wealth=init_wealth)

    def reset(self):
        super().reset()
        self.counter = defaultdict(int)
        self.g_cum = defaultdict(lambda: np.zeros(self.dim))  # cumulative gradients
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


class CombineAlgorithms(OnlineConvexOptimizer):
    def __init__(self, dim, algorithms: List[OnlineConvexOptimizer]):
        self.algorithms = algorithms
        super().__init__(
            dim, algorithms[0].problem,
            side_information=MutlipleSideInformation([algorithm.side_information for algorithm in algorithms]))

    def reset(self):
        super().reset()
        for algorithm in self.algorithms:
            algorithm.reset()
        return self

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


class AddingVectorBettings(CombineAlgorithms):
    def __init__(self, dim, algorithms: List[CoinBettingOLO], init_wealth=1):
        for algorithm in algorithms:
            assert hasattr(algorithm, 'get_vector_betting')

        self.init_wealth = init_wealth
        super().__init__(dim, algorithms)

    def get_vector_betting(self, h_ts=None):
        v = np.zeros(self.dim)
        for i, algorithm in enumerate(self.algorithms):
            # TODO: somehow it works very well?
            v += algorithm.get_vector_betting(h_ts[i])

        return v

    @property
    def wealth(self):
        return self.init_wealth + self.cum_reward

    def get_action(self, h_ts=None):
        return self.init_w + self.wealth * self.get_vector_betting(h_ts)


class Addition(CombineAlgorithms):
    def __init__(self, dim, algorithms: List[OnlineConvexOptimizer]):
        super().__init__(dim, algorithms)

    def get_action(self, h_ts=None):
        w = np.zeros(self.dim)
        for i, algorithm in enumerate(self.algorithms):
            # the correct implementation of (Cutkosky, 2019)
            algorithm.w = algorithm.get_action(h_ts[i])  # set action for each algorithm
            w += algorithm.w

        return w


class Mixture(CombineAlgorithms):
    def __init__(self, dim, algorithms: List[CoinBettingOLO], init_wealth=1, weights=None):
        for algorithm in algorithms:
            assert hasattr(algorithm, 'log_potential')
            assert hasattr(algorithm, 'get_vector_betting')

        self.init_wealth = init_wealth
        if weights is None:
            weights = np.ones(len(algorithms))
        self.log_weights = np.log(weights)

        super().__init__(dim, algorithms)

    def get_vector_betting(self, h_ts=None):
        vs = []
        log_weighted_potentials = []  # log(w_m * psi_m)

        for i, algorithm in enumerate(self.algorithms):
            vs.append(algorithm.get_vector_betting(h_ts[i]))
            log_weighted_potentials.append(self.log_weights[i] + algorithm.log_potential)

        log_potential_mixture = logsumexp(log_weighted_potentials)
        weights = np.exp(np.array(log_weighted_potentials) - log_potential_mixture)
        v_mixture = np.sum(weights[:, np.newaxis] * np.array(vs), axis=0)
        return v_mixture

    @property
    def wealth(self):
        return self.init_wealth + self.cum_reward

    def get_action(self, h_ts=None):
        return self.init_w + self.wealth * self.get_vector_betting(h_ts)


class ContextTreeWeightingOLO(CoinBettingOLO):
    def __init__(self, dim, init_w=None,
                 problem: ConvexProblem = None,
                 init_wealth=1, max_depth=1, alpha=.5, quantizer=None):
        # hyperparameters
        self.alpha = alpha
        self.max_depth = max_depth

        # internal statistics
        self.context_tree = None

        super().__init__(dim, init_w=init_w,
                         problem=problem, side_information=MarkovSideInformation(max_depth, quantizer=quantizer),
                         init_wealth=init_wealth)

    def reset(self):
        super().reset()
        self.context_tree = ContextTreeWeightingOLOBase(max_depth=self.max_depth, alpha=self.alpha, dim=self.dim)
        return self

    def get_vector_betting(self, suffix):
        return self.context_tree.v_recur(state=suffix)

    def update(self, g_t, h_t=None, data=None):
        super().update(g_t, h_t, data)
        self.context_tree.update(state=h_t, g_new=-g_t)
        return self

    @property
    def log_potential(self):
        return self.context_tree.log_potential


class ContextTreeAdditionOLO(OnlineConvexOptimizer):
    def __init__(self, dim, init_w=None,
                 problem: ConvexProblem = None,
                 max_depth=1, quantizer=None, get_base_algorithm=None):
        # hyperparameters
        self.max_depth = max_depth
        self.get_base_algorithm = get_base_algorithm

        # internal statistics
        self.context_tree = None

        super().__init__(dim, init_w=init_w,
                         problem=problem, side_information=MarkovSideInformation(max_depth, quantizer=quantizer))

    def get_action(self, h_t=None):
        return self.context_tree.w_recur(h_t)

    def reset(self):
        super().reset()
        self.context_tree = ContextTreeAdditionOLOBase(max_depth=self.max_depth,
                                                       get_base_algorithm=self.get_base_algorithm)
        return self

    def update(self, g_t, h_t=None, data=None):
        super().update(g_t, h_t, data)
        self.context_tree.update(state=h_t, g_new=g_t)
        return self
