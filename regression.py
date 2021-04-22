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
        '''
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
        '''
        return -x if w @ x < y else x

    def fit(self, X, y):
        pass


class OnlineGradientDescent(OnlineLinearRegressionWithAbsoluteLoss):
    def __init__(self, scale_lr):
        super().__init__()
        self.scale_lr = scale_lr

    def lr(self, t):
        # time varying learning rate
        return self.scale_lr / np.sqrt(t)

    def fit(self, X, y):
        T, dim = X.shape
        w = np.zeros(dim)

        losses = []  # cumulative loss
        lin_losses = []  # linearized cumulative loss

        for t in range(1, T+1):
            # Receive data point
            x_t, y_t = X[t], y[t]

            # Incur loss
            losses.append(self.loss(w, x_t, y_t))
            g_t = self.subgradient(w, x_t, y_t)
            lin_losses.append(g_t @ w)

            # Update
            w = w - self.lr(t) * g_t  # no projection since unconstrained

        # Store results
        self.losses = losses
        self.lin_losses = lin_losses
        self.w = w

        return self
