from collections import defaultdict

import numpy as np

from ctw import ContextTree


class OnlineLinearRegressionWithAbsoluteLoss:
    def __init__(self, dim):
        self._losses = []  # list of losses
        self._lin_losses = []  # list of linearized losses

        self.dim = dim
        self.w = np.zeros(self.dim)

    @property
    def losses(self):
        return np.array(self._losses)

    @property
    def lin_losses(self):
        return np.array(self._lin_losses)

    @property
    def cumulative_loss(self):
        return self.losses.sum()

    def get_side_information(self, *args, **kwargs):
        return None

    def get_action(self, h_t=None):
        raise NotImplementedError

    def update(self, g_t, h_t=None):
        raise NotImplementedError

    @staticmethod
    def loss(w, data):
        x, y = data
        return np.abs(w @ x - y)

    @staticmethod
    def subgradient(w, data):
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

    def fit(self, x_t, y_t):
        h_t = self.get_side_information()
        return self._fit(x_t, y_t, h_t)

    def _fit(self, x_t, y_t, h_t):
        # get action
        self.w = self.get_action(h_t)

        # compute subgradient
        g_t = self.subgradient(self.w, (x_t, y_t))

        # compute and store losses
        self._losses.append(self.loss(self.w, (x_t, y_t)))
        self._lin_losses.append(g_t @ self.w)

        # update
        self.update(g_t, h_t)

        return self

    def fit_batch(self, X, y):
        for i in range(len(X)):
            self.fit(X[i], y[i])
        return self


class OnlineGradientDescent(OnlineLinearRegressionWithAbsoluteLoss):
    def __init__(self, dim, lr_scale):
        super().__init__(dim)

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

    def update(self, g_t, h_t=None):
        self.g_prev = g_t
        return self


class DimensionFreeExponentiatedGradient(OnlineLinearRegressionWithAbsoluteLoss):
    def __init__(self, dim, a=1, L=1, delta=1):
        """
        Dimension-free Exponentiated Gradient (DFEG)

        References
        ----------
        [1] ...

        Parameters
        ----------
        a
        delta
        L: Lipschitz constant
        """
        super().__init__(dim)
        # hyperparameters
        self.a = a
        self.delta = delta
        self.L = L

        # initialize variables
        self.w = np.zeros(self.dim)
        self.th = np.zeros(self.dim)
        self.H = self.delta

    def get_action(self, h_t=None):
        # Set parameters
        self.H += self.L ** 2  # assumed ||x_t|| = 1; TODO: check if this requires to set normalize=True
        alpha = self.a * np.sqrt(self.H)
        beta = self.H ** (3 / 2)

        # Set w
        norm_th = np.linalg.norm(self.th)
        if norm_th == 0:
            w = np.zeros(self.dim)
        else:
            w = (self.th / norm_th) * (np.exp(norm_th / alpha) / beta)  # the EG step

        return w

    def update(self, g_t, h_t=None):
        self.th -= g_t
        return self


class AdaptiveNormal(OnlineLinearRegressionWithAbsoluteLoss):
    def __init__(self, dim, a=1, L=1, eps=1):
        """
        Adaptive Normal (AdaNormal)

        References
        ----------
        [1] ...

        Parameters
        ----------
        a
        L: Lipschitz constant
        eps
        """
        super().__init__(dim)

        # hyperparameters
        self.a = a
        self.eps = eps
        self.L = L

        # initialize variables
        self.th = np.zeros(dim)

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

    def update(self, g_t, h_t=None):
        self.th -= g_t
        return self


class CoinBetting(OnlineLinearRegressionWithAbsoluteLoss):
    def __init__(self, dim, init_wealth=1):
        super().__init__(dim)
        self.init_wealth = init_wealth
        self.g_cum = np.zeros(self.dim)

    @property
    def wealth(self):
        return self.init_wealth - self.lin_losses.sum()

    def get_action(self, h_t=None):
        t = len(self._losses) + 1
        v = self.g_cum / t  # vectorial KT betting
        w = v * self.wealth

        return w

    def update(self, g_t, h_t=None):
        self.g_cum -= g_t
        return self


class Quantizer:
    def __init__(self, quantizer_vector):
        self.quantizer_vector = quantizer_vector

    def quantize(self, g):
        return np.sign(self.quantizer_vector @ g + 1e-10).astype(int)


class CoinBettingWithQuantizedSideInformation(CoinBetting, Quantizer):
    def __init__(self, dim, init_wealth=1, quantizer_vector=None):
        """
        A placeholder for coin betting with side information with a binary quantizer

        Parameters
        ----------
        dim
        init_wealth
        quantizer_vector
        """
        CoinBetting.__init__(self, dim, init_wealth)
        Quantizer.__init__(self, quantizer_vector)

        # initialize parameters
        self.counter = defaultdict(int)
        self.g_cum = defaultdict(lambda: np.zeros(dim))  # cumulative gradients

    def get_action(self, h_t=None):
        assert h_t is not None
        v = self.g_cum[h_t] / (self.counter[h_t] + 1)  # vectorial KT betting
        w = v * self.wealth
        return w

    def update(self, g_t, h_t=None):
        self.counter[h_t] += 1
        self.g_cum[h_t] += -g_t  # accumulate -g since we are in a loss minimization framework
        return self


class CoinBettingWithFixedDepthSideInformation(CoinBettingWithQuantizedSideInformation):
    def __init__(self, dim, init_wealth=1, depth=1, quantizer_vector=None):
        """
        Implement coin betting with a fixed order Markov type side information with a binary quantizer

        Parameters
        ----------
        dim
        init_wealth
        depth
        quantizer_vector
        """
        super().__init__(dim, init_wealth, quantizer_vector)
        self.depth = depth
        self.suffix = [1 for _ in range(self.depth)]  # initialize quantized suffix [Q(g_{t-d}) for d in [1,...,D]]

    def get_side_information(self):
        return tuple(self.suffix)  # h_t = Q(g_{t-1}^{t-D})

    def update(self, g_t, h_t=None):
        super().update(g_t, h_t)
        suffix = self.suffix[1:] + [self.quantize(g_t)]  # update the suffix
        self.suffix = suffix[:self.depth]  # to handle the degenerate case when depth=0
        return self


class CoinBettingWithHint(CoinBettingWithQuantizedSideInformation, Quantizer):
    def __init__(self, dim, init_wealth=1, quantizer_vector=None):
        """
        Implement coin betting with a fixed order Markov type side information with a binary quantizer

        Parameters
        ----------
        dim
        init_wealth
        quantizer_vector
        """
        CoinBettingWithQuantizedSideInformation.__init__(self, dim, init_wealth, quantizer_vector)

        # initialize parameters
        self.counter = defaultdict(int)
        self.g_cum = defaultdict(lambda: np.zeros(dim))  # cumulative gradients

    def get_side_information(self, x_t, y_t):
        return self.quantize(self.subgradient(self.w, (x_t, y_t)))

    def fit(self, x_t, y_t):
        h_t = self.get_side_information(x_t, y_t)
        return self._fit(x_t, y_t, h_t)


class ContextTreeWeighting(OnlineLinearRegressionWithAbsoluteLoss, Quantizer):
    def __init__(self, dim, init_wealth=1, max_depth=1, alpha=.5, quantizer_vector=None):
        OnlineLinearRegressionWithAbsoluteLoss.__init__(self, dim)
        Quantizer.__init__(self, quantizer_vector)

        self.init_wealth = init_wealth
        self.max_depth = max_depth
        self.alpha = alpha

        self.suffix = [1 for _ in range(self.max_depth)]  # initialize quantized suffix [Q(g_{t-d}) for d in [1,...,D]]
        self.context_tree = ContextTree(max_depth=self.max_depth, alpha=self.alpha, dim=dim)

    @property
    def wealth(self):
        return self.init_wealth - self.lin_losses.sum()

    def get_side_information(self):
        return tuple(self.suffix)  # h_t = Q(g_{t-1}^{t-D})

    def get_action(self, suffix=None):
        v = self.context_tree.v_ctw(state=suffix)
        w = v * self.wealth
        return w

    def update(self, g_t, h_t=None):
        self.context_tree.update(state=h_t, g_new=-g_t)
        suffix = self.suffix[1:] + [self.quantize(g_t)]  # update the suffix
        self.suffix = suffix[:self.max_depth]  # to handle the degenerate case when max_depth=0
        return self


class CombineOLOAlgorithms(OnlineLinearRegressionWithAbsoluteLoss):
    def __init__(self, dim, algorithms):
        super().__init__(dim)
        self.algorithms = algorithms

    def fit(self, x_t, y_t):
        # # get action
        # self.w = np.zeros(self.dim)
        # for algorithm in self.algorithms:
        #     self.w += algorithm.fit(x_t, y_t).w
        #
        # # compute subgradient
        # g_t = self.subgradient(self.w, (x_t, y_t))
        #
        # # compute and store losses
        # self._losses.append(self.loss(self.w, (x_t, y_t)))
        # self._lin_losses.append(g_t @ self.w)

        # get action
        self.w = np.zeros(self.dim)
        h_ts = []
        for algorithm in self.algorithms:
            h_t = algorithm.get_side_information()
            self.w += algorithm.get_action(h_t)
            h_ts.append(h_t)

        # compute subgradient
        g_t = self.subgradient(self.w, (x_t, y_t))

        # compute and store losses
        self._losses.append(self.loss(self.w, (x_t, y_t)))
        self._lin_losses.append(g_t @ self.w)

        # update
        for i, algorithm in enumerate(self.algorithms):
            algorithm._losses.append(self.loss(algorithm.w, (x_t, y_t)))
            algorithm._lin_losses.append(g_t @ algorithm.w)
            algorithm.update(g_t, h_ts[i])

        return self

    def get_side_information(self):
        return None

    def get_action(self, h_t=None):
        return None

    def update(self, g_t, h_t=None):
        return self
