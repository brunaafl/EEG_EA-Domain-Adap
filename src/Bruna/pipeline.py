from braindecode.datasets import create_from_X_y

import numpy as np

from numpy import unique

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

from alignment import euclidean_alignment
from dataset import split_runs_EA, split_runs_RA, split_runs_RS


class TransformaParaWindowsDataset(BaseEstimator, TransformerMixin):
    def __init__(self, kw_args=None):
        self.kw_args = kw_args

    def fit(self, X, y=None):
        self.y = y
        # self.X = euclidean_alignment(X.get_data())
        return self

    def transform(self, X, y=None):

        if y is None:
            y = self.y

        if isinstance(X, np.ndarray):

            dataset = create_from_X_y(
                X=X,
                y=y,
                window_size_samples=X.shape[2],
                window_stride_samples=X.shape[2],
                drop_last_window=False,
                sfreq=250.0, )  # X.info["sfreq"]

        else:
            dataset = create_from_X_y(
                X=X.get_data(),
                y=self.y,
                window_size_samples=X.get_data().shape[2],
                window_stride_samples=X.get_data().shape[2],
                drop_last_window=False,
                sfreq=X.info["sfreq"],
            )

        return dataset

    def __sklearn_is_fitted__(self):
        """Return True since Transfomer is stateless."""
        return True


class TransformaParaWindowsDatasetEA(BaseEstimator, TransformerMixin):
    def __init__(self, len_run, atype='euclid', tbreak=None, kw_args=None):
        self.kw_args = kw_args
        self.len_run = len_run
        self.atype = atype
        self.tbreak = tbreak

    def fit(self, X, y=None):
        self.y = y
        return self

    def transform(self, X, y=None):

        if y is None:
            y = self.y

        if isinstance(X, np.ndarray):

            print(X.shape)

            if self.atype == 'euclid':
                X_EA = split_runs_EA(X, self.len_run)
            elif self.atype == 'riemann':
                X_EA = split_runs_RA(X, self.len_run)
            elif self.atype == 'resting':
                X_EA = split_runs_RS(X, self.tbreak, self.len_run)

            print(X_EA.shape)

            dataset = create_from_X_y(
                X=X_EA,
                y=y,
                window_size_samples=X_EA.shape[2],
                window_stride_samples=X_EA.shape[2],
                drop_last_window=False,
                sfreq=250.0, )  # X.info["sfreq"]

        else:

            if self.atype == 'euclid':
                X_EA = split_runs_EA(X.get_data(), self.len_run)
            elif self.atype == 'riemann':
                X_EA = split_runs_RA(X.get_data(), self.len_run)
            elif self.atype == 'resting':
                X_EA = split_runs_RS(X.get_data(), self.tbreak, self.len_run)

            dataset = create_from_X_y(
                X=X_EA,
                y=self.y,
                window_size_samples=X_EA.get_data().shape[2],
                window_stride_samples=X_EA.get_data().shape[2],
                drop_last_window=False,
                sfreq=X.info["sfreq"],
            )

        return dataset

    def __sklearn_is_fitted__(self):
        """Return True since Transfomer is stateless."""
        return True


class ClassifierModel(BaseEstimator, ClassifierMixin):
    def __init__(self, clf, kw_args=None):
        self.clf = clf
        self.classes_ = None
        self.kw_args = kw_args

    def fit(self, X, y=None):
        self.clf.fit(X, y=y, **self.kw_args)
        self.classes_ = unique(y)

        return self.clf

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)