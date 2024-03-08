import copy

import numpy as np
from numpy import any, iscomplexobj, isfinite, real
from pyriemann.utils.covariance import covariances
from pyriemann.utils.geodesic import geodesic_riemann
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


def compute_ref_riemann(data=None, mean=None, dtype='covmat'):
    if dtype != 'covmat':
        covmats = covariances(data, estimator='lwf')
        data = covmats

    if mean is None:
        mean = mean_covariance(data, metric='riemann')

    compare = np.allclose(mean, np.identity(mean.shape[0]))

    if not compare:

        if iscomplexobj(mean):
            print("covariance matrix problem")
        if iscomplexobj(sqrtm(mean)):
            print("covariance matrix problem sqrt")

        r_ra = inv(sqrtm(mean))

        if iscomplexobj(r_ra):
            print("WARNING! Covariance matrix was not SPD somehow. " +
                  "Can be caused by running ICA-EOG rejection, if " +
                  "not, check data!!")
            r_ra = real(r_ra).astype(np.float64)
        elif not any(isfinite(r_ra)):
            print("WARNING! Not finite values in R Matrix")

    else:
        print("Already aligned!")
        r_ra = mean

    return r_ra


def compute_resting_alignment(data, covmats_rest, r_ra=None, mean=None, dtype='covmat'):
    data = copy.deepcopy(data)

    if r_ra is None:
        r_ra = compute_ref_riemann(data=covmats_rest, mean=mean)

    if dtype == "raw":
        result = np.matmul(r_ra, data)

    else:
        result = np.matmul(r_ra, data)
        result = np.matmul(result, r_ra)

    return result


def riemannian_resting_alignment(covmats, covmats_rest, size=24, dtype='covmat'):
    covmats_aux = []
    if size is None:
        m = covmats.shape[0]
    else:
        m = size
    n = covmats.shape[0]
    for k in range(int(n / m)):
        batch = covmats[k * m:(k + 1) * m]
        rest = covmats_rest[k * m:(k + 1) * m]
        batch_RA = compute_resting_alignment(batch, rest, dtype=dtype)
        covmats_aux.append(batch_RA)
    covmats_RA = np.concatenate(covmats_aux)
    return covmats_RA

