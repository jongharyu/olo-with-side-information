import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


class Dataset:
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
    def normalize(X):
        return X / np.sqrt((X ** 2).sum(axis=1, keepdims=True))

    @staticmethod
    def standardize(X):
        mu = X.mean(axis=0, keepdims=True)
        sigma = X.std(axis=0, keepdims=True)
        sigma[sigma == 0] = 1
        return (X - mu) / sigma, mu, sigma

    def preprocess_attributes(self, X, standardize, normalize):
        if standardize:
            X = self.standardize(X)[0]
        if normalize:
            X = self.normalize(X)

        return X


class CpuSmall(Dataset):
    """
    References
    ----------
    [1] https://www.cs.toronto.edu/~delve/data/comp-activ/desc.html
    [2] ftp://ftp.cs.toronto.edu/pub/neuron/delve/data/tarfiles/comp-activ.tar.gz
    """

    def __init__(self, root='.', standardize=False, normalize=True):
        super().__init__()
        self.X, self.y = self.load_and_preprocess(root, standardize, normalize)
        self.classification = False
        self.name = 'cpuSmall'

    def load_and_preprocess(self, root, standardize, normalize):
        filename = '{}/data/cpuSmall/Prototask.data.gz'.format(root)
        df = pd.read_csv(filename, compression='gzip', header=None, sep=' ', error_bad_lines=False, engine='python')
        X, y = np.array(df[df.columns[:-1]]), np.array(df[df.columns[-1]])
        X = self.preprocess_attributes(X, standardize, normalize)

        return X, y


class Houses(Dataset):
    """
    References
    ----------
    [1] http://lib.stat.cmu.edu/datasets/houses.zip
    """

    def __init__(self, root='.', standardize=False, normalize=True):
        super().__init__()
        self.X, self.y = self.load_and_preprocess(root, standardize, normalize)
        self.classification = False
        self.name = 'Houses'

    def load_and_preprocess(self, root, standardize, normalize):
        filename = '{}/data/Houses/cadata_dataonly.txt'.format(root)
        df = pd.read_csv(filename, header=None, sep='  ', error_bad_lines=False, engine='python')

        # Handling the last, exceptional delimiter
        df[[7, 8]] = df[df.columns[-1]].str.split(' ', expand=True)
        df[7] = pd.to_numeric(df[7])
        df[8] = pd.to_numeric(df[8])

        X, y = np.array(df[df.columns[1:]]), np.array(df[df.columns[0]])
        X = self.preprocess_attributes(X, standardize, normalize)

        return X, y


class YearPredictionMSD(Dataset):
    """
    References
    ----------
    [1] https://archive.ics.uci.edu/ml/datasets/YearPredictionMSD
    """

    def __init__(self, root='.', standardize=False, normalize=True):
        super().__init__()
        self.X, self.y = self.load_and_preprocess(root, standardize, normalize)
        self.classification = False
        self.name = 'YearPredictionMSD'

    def load_and_preprocess(self, root, standardize, normalize):
        filename = '{}/data/YearPredictionMSD/YearPredictionMSD.txt.zip'.format(root)
        df = pd.read_csv(filename, compression='zip', header=None, sep=',', quotechar='"', error_bad_lines=False)
        X, y = np.array(df[df.columns[1:]]), np.array(df[df.columns[0]])
        X = self.preprocess_attributes(X, standardize, normalize)

        return X, y
