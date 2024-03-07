import copy

import numpy as np
from numpy import any, iscomplexobj, isfinite, real
from pyriemann.utils.covariance import covariances
from pyriemann.utils.mean import mean_covariance
from scipy.linalg import inv, sqrtm


def riemannian_alignment(data, y=None):
    data = copy.deepcopy(data)

    assert len(data.shape) == 3

    covs = covariances(data, estimator='cov')

    r = mean_covariance(covs, metric='riemann')

    compare = np.allclose(r, np.identity(r.shape[0]))

    if not compare:

        if iscomplexobj(r):
            print("covariance matrix problem")
        if iscomplexobj(sqrtm(r)):
            print("covariance matrix problem sqrt")

        r_op = inv(sqrtm(r))

        if iscomplexobj(r_op):
            print("WARNING! Covariance matrix was not SPD somehow. " +
                  "Can be caused by running ICA-EOG rejection, if " +
                  "not, check data!!")
            r_op = real(r_op).astype(np.float64)
        elif not any(isfinite(r_op)):
            print("WARNING! Not finite values in R Matrix")

        result = np.matmul(r_op, data)

    else:
        print("Already aligned!")
        result = data
        r_op = r

    return result, r_op
