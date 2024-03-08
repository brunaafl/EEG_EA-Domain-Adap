import numpy as np
import copy

from sklearn.base import BaseEstimator, TransformerMixin

from braindecode.datasets import create_from_X_y

from covariance import compute_covariances, mean_group

from riemann import riemannian_alignment, riemannian_resting_alignment, compute_resting_alignment, \
    riemannian_hyb_resting_alignment, compute_hyb_resting_alignment

# Zanini et al., 2019
# Uses data from resting state to align
class TransformRRA(BaseEstimator, TransformerMixin):

    def __init__(self, t_break, size=None, estimator='lwf', r_ref=None, dtype='covmat'):
        self.dtype = dtype
        self.size = size
        self.estimator = estimator
        # Struggling a little on how/where to define it
        self.t_break = t_break
        self.r_ref = r_ref

    def fit(self, X, y=None):
        self.y = y
        return self

    def transform(self, X, y=None):
        if isinstance(X, list):
            X = X[0]

        t_end = self.t_break

        X_rest = X[:, :, t_end:]
        X = X[:, :, :t_end]

        # Compute the covariances
        if self.dtype == 'covmat':
            X = compute_covariances(X, estimator=self.estimator)
        X_rest = compute_covariances(X_rest, estimator=self.estimator)

        if self.size is None:
            self.size = X.shape[0]

        # Training/Offline mode
        if self.r_ref is None:
            align = riemannian_resting_alignment(X, X_rest, size=self.size, dtype=self.dtype)
        # Online mode
        else:
            align = compute_resting_alignment(X, X_rest, r_ra=self.r_ref, dtype=self.dtype)

        return align

    def __sklearn_is_fitted__(self):
        """Return True since Transfomer is stateless."""
        return True

