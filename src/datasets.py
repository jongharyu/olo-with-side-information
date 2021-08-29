import os
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from quantizer import Quantizer


class RandomBinaryTreeSource:
    def __init__(self, dim, length, quantizer_vector, tree):
        self.dim = dim
        self.length = length
        self.quantizer_vector = quantizer_vector
        self.quantizer = Quantizer(quantizer_vector)
        self.tree = tree  # must be complete
        self.tree_depth = max([len(key) for key in tree])
        self.name = 'Synthetic tree source'
        assert self.tree_depth >= 1

        self.X = self.generate_data()

    def find_state(self, suffix):
        for i in range(1, self.tree_depth + 1):
            if tuple(suffix[-i:]) in self.tree:
                return tuple(suffix[-i:])
        else:
            raise ValueError(
                "The tree is not a proper suffix tree: {} does not contain {}".format(self.tree.keys(), suffix))

    def generate_data(self):
        G = []
        suffix = [1 for _ in range(self.tree_depth)]  # initialize quantized suffix [Q(g_{t-d}) for d in [1,...,D]]
        for t in range(self.length):
            # 1) find state and corresponding parameters based on the suffix and tree
            state = self.find_state(suffix)
            prob = self.tree[state]

            # 2) generate a random vector based on the parameter
            G.append(self.draw_random(prob))

            # 3) update suffix
            suffix = suffix[1:] + [self.quantizer(G[-1])]

        return np.vstack(G)

    def draw_random(self, prob):
        # v = self.draw_random_unit_ball(self.dim)
        # v = self.quantizer(v) * v  # v is now a positively correlated vector with self.quantizer_vector
        v = self.quantizer.quantizer_vector
        u = 2 * (np.random.rand() <= prob) - 1  # with probability prob, u=+1; with prob. 1-prob, u=-1
        return u * v

    @staticmethod
    def draw_random_unit_ball(d):
        x = np.random.normal(0, 1, d)
        e = np.random.exponential(0.5)
        denom = np.sqrt(e + np.sum(x ** 2))
        return x / denom


class RegressionDataset:
    def __init__(self):
        self.X = None
        self.y = None
        self.classification = None
        self.onehot_encoder = None

    @property
    def data(self):
        return self.X, self.y

    def load_and_preprocess(self, *args, **kwargs):
        raise NotImplementedError

    def train_test_split(self, test_size=0.4, seed=0):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=seed)
        return X_train, X_test, y_train, y_test

    @staticmethod
    def center(X):
        mu = X.mean(axis=0, keepdims=True)
        return X - mu, mu

    @staticmethod
    def standardize(X):
        mu = X.mean(axis=0, keepdims=True)
        sigma = X.std(axis=0, keepdims=True)
        sigma[sigma == 0] = 1
        return (X - mu) / sigma, mu, sigma

    @staticmethod
    def rescale(X):
        min = X.min(axis=0, keepdims=True)
        max = X.max(axis=0, keepdims=True)
        return (X - (min + max) / 2) / (max - min)

    @staticmethod
    def normalize(X, bias):
        # instance normalization
        X = X / np.sqrt((X ** 2).sum(axis=1, keepdims=True))
        if bias:
            X = np.hstack([X, np.ones((X.shape[0], 1))]) / np.sqrt(2)
        return X

    @staticmethod
    def batch_normalize(X):
        # instance normalization
        return X / np.max(np.sqrt((X ** 2).sum(axis=1, keepdims=True)))

    def preprocess_attributes(self, X, standardize, normalize, batch_normalize, rescale, bias):
        assert not (normalize and batch_normalize), "Either one of instance normalization or batch normalization must be used"
        assert not (standardize and rescale), "Either one of standardization or rescale must be used"

        if standardize:
            X = self.center(X)[0]
        if rescale:
            X = self.rescale(X)
        if normalize:
            X = self.normalize(X, bias)
        if batch_normalize:
            X = self.batch_normalize(X)

        return X


class CpuSmall(RegressionDataset):
    """
    References
    ----------
    [1] https://www.cs.toronto.edu/~delve/data/comp-activ/desc.html
    [2] ftp://ftp.cs.toronto.edu/pub/neuron/delve/data/tarfiles/comp-activ.tar.gz
    """
    def __init__(self, root='.', standardize=False, normalize=True, batch_normalize=False, rescale=False, bias=False):
        super().__init__()
        self.X, self.y = self.load_and_preprocess(root, standardize, normalize, batch_normalize, rescale, bias)
        self.classification = False
        self.name = 'cpuSmall'

    def load_and_preprocess(self, root, standardize, normalize, batch_normalize, rescale, bias):
        filename = '{}/data/cpuSmall/Prototask.data.gz'.format(root)
        df = pd.read_csv(filename, compression='gzip', header=None, sep=' ', error_bad_lines=False, engine='python')
        X, y = np.array(df[df.columns[:-1]]), np.array(df[df.columns[-1]])
        X = np.log(X + (X.min(axis=0) == 0).astype(int))  # take logarithms
        X = self.preprocess_attributes(X, standardize, normalize, batch_normalize, rescale, bias)

        return X, y


class Houses(RegressionDataset):
    """
    References
    ----------
    [1] http://lib.stat.cmu.edu/datasets/houses.zip
    """
    def __init__(self, root='.', standardize=False, normalize=True, batch_normalize=False, rescale=False, bias=False):
        super().__init__()
        self.X, self.y = self.load_and_preprocess(root, standardize, normalize, batch_normalize, rescale, bias)
        self.classification = False
        self.name = 'Houses'

    def load_and_preprocess(self, root, standardize, normalize, batch_normalize, rescale, bias):
        filename = '{}/data/Houses/cadata_dataonly.txt'.format(root)
        df = pd.read_csv(filename, header=None, sep='  ', error_bad_lines=False, engine='python')

        # Handling the last, exceptional delimiter
        df[[7, 8]] = df[df.columns[-1]].str.split(' ', expand=True)
        df[7] = pd.to_numeric(df[7])
        df[8] = pd.to_numeric(df[8])

        X, y = np.array(df[df.columns[1:]]), np.array(df[df.columns[0]])
        X = self.preprocess_attributes(X, standardize, normalize, batch_normalize, rescale, bias)

        return X, y


class YearPredictionMSD(RegressionDataset):
    """
    References
    ----------
    [1] https://archive.ics.uci.edu/ml/datasets/YearPredictionMSD
    """
    def __init__(self, root='.', standardize=False, normalize=True, batch_normalize=False, rescale=False, bias=False):
        super().__init__()
        self.X, self.y = self.load_and_preprocess(root, standardize, normalize, batch_normalize, rescale, bias)
        self.classification = False
        self.name = 'YearPredictionMSD'

    def load_and_preprocess(self, root, standardize, normalize, batch_normalize, rescale, bias):
        filename = '{}/data/YearPredictionMSD/YearPredictionMSD.txt.zip'.format(root)
        df = pd.read_csv(filename, compression='zip', header=None, sep=',', quotechar='"', error_bad_lines=False)
        X, y = np.array(df[df.columns[1:]]), np.array(df[df.columns[0]])
        X = self.preprocess_attributes(X, standardize, normalize, batch_normalize, rescale, bias)

        return X, y


class BeijingPM2pt5(RegressionDataset):
    """
    References
    ----------
    [1] https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data
    """
    def __init__(self, root='.', standardize=False, normalize=True, batch_normalize=False, rescale=False, bias=False):
        super().__init__()
        self.X, self.y = self.load_and_preprocess(root, standardize, normalize, batch_normalize, rescale, bias)
        self.classification = False
        self.name = 'BeijingPM2.5'

    def load_and_preprocess(self, root, standardize, normalize, batch_normalize, rescale, bias):
        filename = '{}/data/BeijingPM2.5/PRSA_data_2010.1.1-2014.12.31.csv'.format(root)
        df = pd.read_csv(filename, sep=',', error_bad_lines=False, engine='python')
        del df['cbwd'], df['No'], df['year'], df['month'], df['day'], df['hour']
        X = np.array(df[df.columns[1:]])[24:]
        y = np.array(df[df.columns[:1]])[24:]

        # logarithmic scaling
        X[:, 3:] = np.log(1 + X[:, 3:])

        # linear interpolating nan values
        nan_start = -1
        for i in range(len(y)):
            if np.isnan(y[i]):
                if nan_start < 0:  # if we are in the running nan values
                    nan_start = i
            else:
                if nan_start >= 0:
                    slope = (y[i] - y[nan_start - 1]) / (i - nan_start + 1)
                    y[nan_start:i] = y[nan_start - 1] + np.array(
                        [slope * (j - nan_start + 1) for j in range(nan_start, i)])
                    nan_start = -1  # reset

        X = self.preprocess_attributes(X, standardize, normalize, batch_normalize, rescale, bias)

        return X, y


class MetroInterstateTrafficVolume(RegressionDataset):
    """
    References
    ----------
    [1] https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume
    """
    def __init__(self, root='.', standardize=False, normalize=True, batch_normalize=False, rescale=False, bias=False):
        super().__init__()
        self.X, self.y = self.load_and_preprocess(root, standardize, normalize, batch_normalize, rescale, bias)
        self.classification = False
        self.name = 'MetroInterstateTrafficVolume'

    def load_and_preprocess(self, root, standardize, normalize, batch_normalize, rescale, bias):
        filename = '{}/data/MetroInterstateTrafficVolume/Metro_Interstate_Traffic_Volume.csv'.format(root)
        df = pd.read_csv(filename, sep=',', error_bad_lines=False, engine='python')

        # delete unusable columns
        del df['date_time'], df['weather_description']

        # binarize holidays
        holiday = np.array(df['holiday'])
        holiday[~(holiday == 'None')] = 1
        holiday[holiday == 'None'] = -1
        df['holiday'] = holiday

        # binarize weather
        weather = np.array(df['weather_main'])
        weather[~(weather == 'Clear')] = -1
        weather[weather == 'Clear'] = 1
        df['weather_main'] = weather

        # binarize snow
        df['snow_1h'][df['snow_1h'] > 0] = 1
        df['snow_1h'][df['snow_1h'] == 0] = -1

        # scale rain
        scaled_rain = np.log(df['rain_1h'])
        scaled_rain[scaled_rain == -np.inf] = scaled_rain[scaled_rain > -np.inf].min()
        df['rain_1h'] = scaled_rain

        # interpolate zero temperatures
        df['temp'][11898:11902] = df['temp'][11897]
        df['temp'][11946:11952] = df['temp'][11945]

        # Set features and target
        X = np.array(df[df.columns[:-1]]).astype(float)
        y = np.array(df[df.columns[-1]]).astype(float)

        X = self.preprocess_attributes(X, standardize, normalize, batch_normalize, rescale, bias)

        return X, y


class ExampleStocks:
    def __init__(self, root='.'):
        self.X, self.names = self.load_and_preprocess(root)
        self.classification = False
        self.name = 'ExampleStocks'

    def load_and_preprocess(self, root):
        stock_names = {'A': 'Agilent Technologies, Inc.',
                       'B': 'Barnes Group Inc.',
                       'C': 'Citigroup Inc.',
                       'D': 'Dominion Energy, Inc.',
                       'F': 'Ford Motor Company',
                       'M': 'Macy\'s, Inc.',
                       'S': 'Sprint Corporation',
                       'T': 'AT&T Inc.',
                       'X': 'United States Steel Corporation',
                       'Y': 'Alleghany Corporation'}

        path = '{}/data/Portfolio/'.format(root)
        stocks = defaultdict(dict)
        for filename in os.listdir(path):
            if filename.endswith(".csv"):
                df = pd.read_csv('{}/{}'.format(path, filename))
                symbol = os.path.splitext(filename)[0]
                stocks[symbol]['Name'] = stock_names[symbol]
                stocks[symbol]['Relative'] = np.array(df['Close'] / df['Open'])

        X = np.vstack([stocks[key]['Relative'] for key in sorted(stocks.keys())]).T
        names = [stocks[key]['Name'] for key in sorted(stocks.keys())]

        return X, names
