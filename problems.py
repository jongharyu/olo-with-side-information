import numpy as np


class ConvexProblem:
    def __init__(self):
        pass

    @staticmethod
    def compute_loss(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def compute_subgradient(*args, **kwargs):
        raise NotImplementedError


class Linear(ConvexProblem):
    def __init__(self):
        super().__init__()

    @staticmethod
    def compute_loss(w, data):
        g, = data
        return w @ g

    @staticmethod
    def compute_subgradient(w, data):
        g, = data
        return g


class LinearRegressionWithAbsoluteLoss(ConvexProblem):
    def __init__(self):
        super().__init__()

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


class Portfolio(ConvexProblem):
    def __init__(self):
        super().__init__()

    @staticmethod
    def compute_loss(w, data):
        x, = data
        return -np.log(w @ x)

    @staticmethod
    def compute_subgradient(w, data):
        r"""
        Compute the subgradient of the absolute function with linear regression

        .. math::
            l(w) = -\log\langle w, x\rangle

        Parameters
        ----------
        w: portfolio
        data: (price relative,)

        Returns
        -------
        g: subgradient
        """
        x, = data
        return -x / (w @ x)
