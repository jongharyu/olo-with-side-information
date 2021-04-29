import math

from scipy.special import betaln
import numpy as np


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class ContextTree:
    def __init__(self, max_depth=0, alpha=.5, dim=1):
        self.max_depth = max_depth
        self.children = dict()
        self.alpha = alpha

        self.dim = dim  # dimension of g
        self.counter = 0
        self.g_cum = np.zeros(dim)
        self.log_beta = 0

        self.log_potential = 0

    @property
    def is_leaf(self):
        return self.max_depth == 0

    def add_children(self):
        if not self.is_leaf:
            for _ in range(2):
                self.add_child()

    def add_child(self):
        assert self.max_depth >= 1
        for key in [-1, 1]:
            self.children[key] = ContextTree(max_depth=self.max_depth - 1,
                                             alpha=self.alpha,
                                             dim=self.dim)

    def v_ctw(self, state):
        """
        Compute v_ctw for an upcoming symbol

        Parameters
        ----------
        state: list
            list of indices that represents the current state (context, suffix)

        Returns
        -------
        v_ctw_curr: np.array
            prequential CTW betting vector at the current node
        """
        v_kt_curr = self.v_kt()
        if self.is_leaf:
            assert not state, 'At the leaf node, state must be empty.'  # sanity check
            return v_kt_curr
        else:
            if not self.children:
                self.add_children()

            active_child = self.children[state[-1]]
            weight = sigmoid(self.log_beta)  # = beta / (1 + beta)
            v_ctw_active_child = active_child.v_ctw(state=state[:-1])  # recursive call

            v_ctw_curr = weight * v_kt_curr + (1 - weight) * v_ctw_active_child
            return v_ctw_curr

    def update(self, state, g_new):
        """

        Parameters
        ----------
        state: list
        g_new: np.array
            update node.beta and node.counter accordingly

        Returns
        -------
        log_ctw_potential_new_old_ratio: np.float
        """
        # Compute KT potential and weight with old statistics (without g_t=g_new)
        log_kt_potential_old = self.compute_log_kt_potential()  # \psi(g^{t-1})
        weight = sigmoid(self.log_beta)  # = beta(g^{t-1}) / (1 + beta(g^{t-1}))

        # update with g_t
        self.counter += 1
        self.g_cum += g_new

        # Compute KT potential with new statistics (with g_t=g_new)
        log_kt_potential_new = self.compute_log_kt_potential()  # \psi(g_t)

        log_kt_potential_new_old_ratio = log_kt_potential_new - log_kt_potential_old

        if self.is_leaf:
            assert not state, 'At the leaf node, state must be empty.'  # sanity check
            return log_kt_potential_new_old_ratio
        else:
            if not self.children:
                raise ValueError()  # sanity check

            active_child = self.children[state[-1]]
            log_ctw_potential_new_old_ratio = active_child.update(state=state[:-1], g_new=g_new)
            self.log_potential += log_ctw_potential_new_old_ratio
            self.log_beta += log_kt_potential_new_old_ratio - log_ctw_potential_new_old_ratio

            return weight * log_kt_potential_new_old_ratio + (1 - weight) * log_ctw_potential_new_old_ratio

    def compute_log_kt_potential(self):
        t = self.counter
        x = np.sqrt((self.g_cum ** 2).sum())  # l2 norm of \sum g^{t-1}(s)
        return t * np.log(2) + betaln((t + x + 1) / 2, (t - x + 1) / 2) - betaln(.5, .5)

    def v_kt(self):
        return self.g_cum / (self.counter + 2 * self.alpha)
