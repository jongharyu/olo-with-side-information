from quantizer import Quantizer


class SideInformation:
    def __init__(self):
        pass

    def reset(self):
        raise NotImplementedError

    def get(self, *args, **kwargs):
        raise NotImplementedError

    def update(self, g_t):
        raise NotImplementedError


class NoSideInformation(SideInformation):
    def __init__(self):
        super().__init__()

    def reset(self):
        return self

    def get(self, *args, **kwargs):
        return None

    def update(self, g_t):
        return self


class MarkovSideInformation(SideInformation):
    def __init__(self, depth, quantizer: Quantizer = None):
        super().__init__()
        # hyperparameter
        self.depth = depth
        self.quantizer = quantizer

        # initialize internal statistics
        self.suffix = None
        self.reset()

    def reset(self):
        self.suffix = [1 for _ in range(self.depth)]  # initialize quantized suffix [Q(g_{t-d}) for d in [1,...,D]]
        return self

    def get(self):
        return tuple(self.suffix)  # h_t = Q(g_{t-1}^{t-D})

    def update(self, g_t):
        suffix = self.suffix[1:] + [self.quantizer(g_t)]  # update the suffix
        self.suffix = suffix[:self.depth]  # to handle the degenerate case when depth=0
        return self


class QuantizedHint(SideInformation):
    def __init__(self, quantizer: Quantizer = None):
        super().__init__()
        self.quantizer = quantizer

    def reset(self):
        return self

    def get(self, g_future):
        return self.quantizer(g_future)

    def update(self, g_t):
        return self


class MutlipleSideInformation(SideInformation):
    def __init__(self, side_informations=None):
        super().__init__()
        self.side_informations = side_informations

    def reset(self):
        for side_information in self.side_informations:
            side_information.reset()
        return self

    def get(self):
        return [side_information.get() for side_information in self.side_informations]

    def update(self, g_t):
        return self
