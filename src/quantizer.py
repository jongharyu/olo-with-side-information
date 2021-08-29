import numpy as np


def get_standard_vector(dim, i=0):
    vector = np.zeros(dim)
    vector[i] = 1
    return vector


class Quantizer:
    def __init__(self, quantizer_vector):
        self.quantizer_vector = quantizer_vector

    def __call__(self, g):
        return None if self.quantizer_vector is None else \
            np.sign(self.quantizer_vector @ g + 1e-10).astype(int)
