import math

import numpy as np
from scipy.special import betaln


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class ContextTree:
    def __init__(self, max_depth=0):
        self.max_depth = max_depth
        self.children = dict()

    @property
    def is_leaf(self):
        return self.max_depth == 0

    def add_children(self):
        if not self.is_leaf:
            for _ in range(2):
                self.add_child()

    def add_child(self):
        assert self.max_depth >= 1
        raise NotImplementedError


class ContextTreeWeightingOLOBase(ContextTree):
    def __init__(self, max_depth=0, dim=1, alpha=.5):
        super().__init__(max_depth)

        # Statistics for KT algorithm
        self.dim = dim  # dimension of g
        self.counter = 0
        self.g_cum = np.zeros(dim)

        # KT algorithm related parameters
        self.alpha = alpha
        self.log_beta = 0
        self.log_potential = 0

    def add_child(self):
        assert self.max_depth >= 1
        for key in [-1, 1]:
            self.children[key] = ContextTreeWeightingOLOBase(max_depth=self.max_depth - 1, alpha=self.alpha, dim=self.dim)

    def v_recur(self, state):
        """
        Compute v_recur for an upcoming symbol

        Parameters
        ----------
        state: list
            list of indices that represents the current state (context, suffix)

        Returns
        -------
        v_ctw_curr: np.array
            prequential CTW betting vector at the current node
        """
        v_kt_curr = self.v_curr()
        if self.is_leaf:
            assert not state, 'At the leaf node, state must be empty.'  # sanity check
            return v_kt_curr
        else:
            if not self.children:
                self.add_children()

            active_child = self.children[state[-1]]
            weight = sigmoid(self.log_beta)  # = beta / (1 + beta)
            v_ctw_active_child = active_child.v_recur(state=state[:-1])  # recursive call

            v_ctw_curr = weight * v_kt_curr + (1 - weight) * v_ctw_active_child
            return v_ctw_curr

    def v_curr(self):
        # Action of the KT algorithm with respect to the subsequence
        return self.g_cum / (self.counter + 2 * self.alpha)

    def compute_log_kt_potential(self):
        t = self.counter
        x = np.sqrt((self.g_cum ** 2).sum())  # l2 norm of \sum g^{t-1}(s)
        return t * np.log(2) + betaln((t + x + 1) / 2, (t - x + 1) / 2) - betaln(.5, .5)

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
            # recursive update
            assert self.children, "Something went wrong: a leaf node had children"  # sanity check

            active_child = self.children[state[-1]]
            log_ctw_potential_new_old_ratio = active_child.update(state=state[:-1], g_new=g_new)
            self.log_potential += log_ctw_potential_new_old_ratio
            self.log_beta += log_kt_potential_new_old_ratio - log_ctw_potential_new_old_ratio

            return weight * log_kt_potential_new_old_ratio + (1 - weight) * log_ctw_potential_new_old_ratio


class ContextTreeAdditionOLOBase(ContextTree):
    def __init__(self, max_depth=0, get_base_algorithm=None):
        super().__init__(max_depth)
        self.get_base_algorithm = get_base_algorithm
        self.base_algorithm = get_base_algorithm()  # a class instance of the base algorithm

    def add_child(self):
        assert self.max_depth >= 1
        for key in [-1, 1]:
            self.children[key] = ContextTreeAdditionOLOBase(max_depth=self.max_depth - 1,
                                                            get_base_algorithm=self.get_base_algorithm)

    def w_recur(self, state: list) -> np.array:
        """
        Compute action for an upcoming symbol

        Parameters
        ----------
        state: list
            list of indices that represents the current state (context, suffix)

        Returns
        -------
        v_cum: np.array
            cumulative action vector up to the current node
        """
        if self.is_leaf:
            assert not state, 'At the leaf node, state must be empty.'  # sanity check
            return self.w_curr
        else:
            if not self.children:
                self.add_children()

            active_child = self.children[state[-1]]
            w_recur_active_child = active_child.w_recur(state=state[:-1])
            w_recur = self.w_curr + w_recur_active_child

            return w_recur

    @property
    def w_curr(self):
        # Action of the base algorithm with respect to the subsequence
        self.base_algorithm.w = self.base_algorithm.get_action()
        return self.base_algorithm.w

    def update(self, state, g_new):
        """
        Parameters
        ----------
        state: list
        g_new: np.array
            update node.beta and node.counter accordingly

        Returns
        -------
        """
        # update with g_t
        # warning: at this point, self.base_algorithm.w must be updated with self.base_algroithm.get_action();
        #          this is currently done in self.w_curr
        self.base_algorithm.update(g_t=g_new, data=(g_new,))

        if self.is_leaf:
            assert not state, 'At the leaf node, state must be empty.'  # sanity check
            return
        else:
            # recursive update
            assert self.children, "Something went wrong: a leaf node had children"  # sanity check

            active_child = self.children[state[-1]]
            return active_child.update(state=state[:-1], g_new=g_new)


