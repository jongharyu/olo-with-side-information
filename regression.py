from collections import defaultdict

import numpy as np


class OnlineLinearRegressionWithAbsoluteLoss:
    def __init__(self):
        self.losses = None
        self.lin_losses = None
        self.w = None

    @staticmethod
    def loss(w, x, y):
        return np.abs(w @ x - y)

    @staticmethod
    def subgradient(w, x, y):
        r"""
        Compute the subgradient of the absolute function with linear regression

        .. math::
            l(w) = \abs{\langle w, x\rangle - y}

        Parameters
        ----------
        w: weight vector
        x: feature (data point)
        y: response

        Returns
        -------
        g: subgradient
        """
        return -x if w @ x < y else x

    def fit(self, X, y):
        pass

    @property
    def cumulative_loss(self):
        return self.losses.sum()


class OnlineGradientDescent(OnlineLinearRegressionWithAbsoluteLoss):
    def __init__(self, lr_scale):
        super().__init__()
        self.lr_scale = lr_scale

    def lr(self, t):
        # time varying learning rate
        return self.lr_scale / np.sqrt(t)

    def fit(self, X, y):
        T, dim = X.shape
        w = np.zeros(dim)

        losses = []  # cumulative loss
        lin_losses = []  # linearized cumulative loss

        for t in range(1, T+1):
            # Receive data point and compute gradient
            x_t, y_t = X[t - 1], y[t - 1]
            g_t = self.subgradient(w, x_t, y_t)

            # Incur loss
            losses.append(self.loss(w, x_t, y_t))
            lin_losses.append(g_t @ w)

            # Update
            w = w - self.lr(t) * g_t  # no projection since unconstrained

        # Store results
        self.losses = np.array(losses)
        self.lin_losses = np.array(lin_losses)
        self.w = w

        return self


class DimensionFreeExponentiatedGradient(OnlineLinearRegressionWithAbsoluteLoss):
    def __init__(self, a=1, L=1, delta=1):
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
        super().__init__()
        self.a = a
        self.delta = delta
        self.L = L

    def fit(self, X, y):
        T, dim = X.shape

        # initializing variables
        th_t = np.zeros(dim)
        H_t = self.delta
        w = None

        losses = []  # cumulative loss
        lin_losses = []  # linearized cumulative loss

        for t in range(1, T + 1):
            # Set w_t
            H_t = H_t + self.L ** 2  # assumed ||x_t|| = 1; TODO: check if this requires to set normalize=True
            alpha_t = self.a * np.sqrt(H_t)
            beta_t = H_t ** (3 / 2)

            norm_th = np.linalg.norm(th_t)
            if norm_th == 0:
                w = np.zeros(dim)
            else:
                w = (th_t / norm_th) * (np.exp(norm_th / alpha_t) / beta_t)  # the EG step

            # Receive data point and compute gradient
            x_t, y_t = X[t - 1], y[t - 1]
            g_t = self.subgradient(w, x_t, y_t)

            # Incur loss
            losses.append(self.loss(w, x_t, y_t))
            lin_losses.append(g_t @ w)

            # Update theta
            th_t = th_t - g_t

        # Store results
        self.losses = np.array(losses)
        self.lin_losses = np.array(lin_losses)
        self.w = w

        return self


class AdaptiveNormal(OnlineLinearRegressionWithAbsoluteLoss):
    def __init__(self, a=1, L=1, eps=1):
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
        super().__init__()
        self.a = a
        self.eps = eps
        self.L = L

    def fit(self, X, y):
        T, dim = X.shape

        # initializing variables
        th_t = np.zeros(dim)
        w = None

        losses = []  # cumulative loss
        lin_losses = []  # linearized cumulative loss

        for t in range(1, T + 1):
            # Set w_t
            norm_th = np.linalg.norm(th_t)
            if norm_th == 0:
                w = np.zeros(dim)
            else:
                term1 = np.exp(((norm_th + self.L) ** 2) / (2 * self.a * (t + 1)))
                term1 -= np.exp(((norm_th - self.L) ** 2) / (2 * self.a * (t + 1)))
                term2 = (2 * self.L * (np.log(t + 2)) ** 2) ** (-1)
                w = th_t * self.eps * term1 * term2 / norm_th  # the AdaNormal step

            # Receive data point and compute gradient
            x_t, y_t = X[t - 1], y[t - 1]
            g_t = self.subgradient(w, x_t, y_t)

            # Incur loss
            losses.append(self.loss(w, x_t, y_t))
            lin_losses.append(g_t @ w)

            # Update theta
            th_t = th_t - g_t

        # Store results
        self.losses = np.array(losses)
        self.lin_losses = np.array(lin_losses)
        self.w = w

        return self


class CoinBetting(OnlineLinearRegressionWithAbsoluteLoss):
    def __init__(self, init_wealth=1):
        super().__init__()
        self.init_wealth = init_wealth

    def fit(self, X, y):
        T, dim = X.shape

        # initialize variables
        w = np.zeros(dim)
        g_cum = np.zeros(dim)  # cumulative gradients

        losses = []  # cumulative loss
        lin_losses = []  # linearized cumulative loss

        for t in range(1, T + 1):
            # Set w_t
            v = g_cum / t  # vectorial KT betting
            w = v * (self.init_wealth - sum(lin_losses))

            # Receive data point and compute gradient
            x_t, y_t = X[t - 1], y[t - 1]
            g_t = self.subgradient(w, x_t, y_t)

            # Incur loss
            losses.append(self.loss(w, x_t, y_t))
            lin_losses.append(g_t @ w)

            # Update
            g_cum = g_cum - g_t

        # Store results
        self.losses = np.array(losses)
        self.lin_losses = np.array(lin_losses)
        self.w = w

        return self


class CoinBettingWithQuantizedSideInformation(CoinBetting):
    def __init__(self, init_wealth=1, depth=1, quantizer_vector=None):
        """
        Implement coin betting with a fixed order Markov type side information with a binary quantizer

        Parameters
        ----------
        init_wealth
        depth
        quantizer_vector
        """
        super().__init__(init_wealth)
        self.quantizer_vector = quantizer_vector
        self.depth = depth

    def quantizer(self, g):
        return np.sign(self.quantizer_vector @ g + 1e-10).astype(int)

    def fit(self, X, y):
        T, dim = X.shape

        # initialize variables
        w = np.zeros(dim)
        counter = defaultdict(int)
        g_cum = defaultdict(lambda: np.zeros(dim))  # cumulative gradients

        losses = []  # cumulative loss
        lin_losses = []  # linearized cumulative loss

        suffix = [1 for _ in range(self.depth)]  # initialize quantized suffix [Q(g_{t-d}) for d in [1,...,D]]
        for t in range(1, T + 1):
            # Find side information
            h_t = tuple(suffix)  # h_t = Q(g_{t-1}^{t-D})

            # Set w_t
            v = g_cum[h_t] / (counter[h_t] + 1)  # vectorial KT betting
            w = v * (self.init_wealth - sum(lin_losses))

            # Receive data point and compute gradient
            x_t, y_t = X[t - 1], y[t - 1]
            g_t = self.subgradient(w, x_t, y_t)

            # Incur loss
            losses.append(self.loss(w, x_t, y_t))
            lin_losses.append(g_t @ w)

            # Update
            counter[h_t] += 1
            g_cum[h_t] += -g_t  # accumulate -g since we are in a loss minimization framework
            suffix = suffix[1:] + [self.quantizer(g_t)]  # update the suffix

        # Store results
        self.losses = np.array(losses)
        self.lin_losses = np.array(lin_losses)
        self.w = w

        return self
