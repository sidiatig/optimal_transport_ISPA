import numpy as np
import ot
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from data import fmri_data_cv as fmril

Xs=fmril.x_indi[:40]
Xt = fmril.x_indi[40:80]
ys=fmril.y_target[:40]
yt=fmril.y_target[40:80]

M = ot.dist(Xs, Xt, metric='euclidean')
M = np.sqrt(M)
M = np.asarray(M, dtype=np.float64)

reg = 1e-1

a = np.ones((M.shape[0],), dtype=np.float64) / M.shape[0]
b = np.ones((M.shape[1],), dtype=np.float64) / M.shape[1]
coupling = ot.bregman.sinkhorn_knopp(a,b,M,reg)
transp = coupling / np.sum(coupling, 1)[:, None]
# set nans to 0
transp[~ np.isfinite(transp)] = 0
# compute transported samples
transp_Xs = np.dot(transp, Xt)
logis2 = LogisticRegression(C=0.001)
logis2.fit(transp_Xs,ys)
pre2 = logis2.predict(Xt)
score2 = accuracy_score(yt,pre2)



cpt = 0
err = 1
numItermax=1000
stopThr=1e-9
verbose=False
log=False,
# sinkhorn_knopp
a = np.ones((M.shape[0],), dtype=np.float64) / M.shape[0]
b = np.ones((M.shape[1],), dtype=np.float64) / M.shape[1]

Nini = len(a)
Nfin = len(b)

if len(b.shape) > 1:
    nbb = b.shape[1]
else:
    nbb = 0

if nbb:
    u = np.ones((Nini, nbb)) / Nini
    v = np.ones((Nfin, nbb)) / Nfin
else:
    u = np.ones(Nini) / Nini
    v = np.ones(Nfin) / Nfin

K = np.exp(-M / reg)
Kp = (1 / a).reshape(-1, 1) * K

while (err > stopThr and cpt < numItermax):
    uprev = u
    vprev = v
    KtransposeU = np.dot(K.T, u)
    v = np.divide(b, KtransposeU)
    u = 1. / np.dot(Kp, v)

    if (np.any(KtransposeU == 0) or
            np.any(np.isnan(u)) or np.any(np.isnan(v)) or
            np.any(np.isinf(u)) or np.any(np.isinf(v))):
        # we have reached the machine precision
        # come back to previous solution and quit loop
        print('Warning: numerical errors at iteration', cpt)
        u = uprev
        v = vprev
        break
    if cpt % 10 == 0:
        # we can speed up the process by checking for the error only all
        # the 10th iterations
        if nbb:
            err = np.sum((u - uprev) ** 2) / np.sum((u) ** 2) + \
                  np.sum((v - vprev) ** 2) / np.sum((v) ** 2)
        else:
            transp = u.reshape(-1, 1) * (K * v)
            err = np.linalg.norm((np.sum(transp, axis=0) - b)) ** 2
        # if log:
        #     log['err'].append(err)

        if verbose:
            if cpt % 200 == 0:
                print(
                    '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
            print('{:5d}|{:8e}|'.format(cpt, err))
    cpt = cpt + 1

coupling=u.reshape((-1, 1)) * K * v.reshape((1, -1))
transp = coupling / np.sum(coupling, 1)[:, None]

# set nans to 0
transp[~ np.isfinite(transp)] = 0

# compute transported samples
transp_Xs = np.dot(transp, Xt)

logis2 = LogisticRegression(C=0.001)
logis2.fit(transp_Xs,ys)
pre2 = logis2.predict(Xt)
score2 = accuracy_score(yt,pre2)


