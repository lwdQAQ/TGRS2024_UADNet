import numpy as np
import torch
import matplotlib.pyplot as plt
import math

# endmember normalization
def norm_endmember(M_est, M_gt):
    R = M_gt.shape[1]
    for r in range(R):
        M_est[:, r] = M_est[:, r] / M_est[:, r].max()
        M_gt[:, r] = M_gt[:, r] / M_gt[:, r].max()
    return M_est, M_gt

def norm_abundance(A_est, A_gt):
    # size A_est: [1, R, Nr, Nc]
    # size A_gt: [R, Nr, Nc]
    _, R, Nr, Nc = A_est.shape
    A_est = A_est / (torch.sum(A_est, dim=1))
    A_est = torch.reshape(A_est.squeeze(0), (R, Nr, Nc))
    A_est = A_est.cpu().detach().numpy()
    A_gt = A_gt / (torch.sum(A_gt, dim=0))
    A_gt = A_gt.cpu().detach().numpy()
    return A_est, A_gt

def norm_energymap(E_est):
    return (E_est - E_est.min()) / (E_est.max() - E_est.min())

def AbundanceRMSE(A_est, A_gt):
    RMSE = np.sqrt(((A_est - A_gt) ** 2).mean())
    return RMSE

def EndmemberSAD(M_est, M_gt):
    cos_sim = np.dot(M_est, M_gt) / (np.linalg.norm(M_est) * np.linalg.norm(M_gt))
    SAD_sim = np.arccos(cos_sim)
    return SAD_sim

# change the index of abundance and endmember
def arange_A_E(A_est, A_gt, M_est, M_gt):
    # A_est, A_gt: RxNrxNc
    # M_est, M_gt: BxR
    R = M_gt.shape[1]

    RMSE_matrix = np.zeros((R, R))
    SAD_matrix = np.zeros((R, R))
    RMSE_index = np.zeros(R).astype(int)
    SAD_index = np.zeros(R).astype(int)
    RMSE_abundance = np.zeros(R)
    SAD_endmember = np.zeros(R)

    for i in range(R):
        for j in range(R):
            RMSE_matrix[i, j] = AbundanceRMSE(A_gt[i, :, :], A_est[j, :, :])
            SAD_matrix[i, j] = EndmemberSAD(M_gt[:, i], M_est[:, j])
        RMSE_index[i] = np.argmin(RMSE_matrix[i, :])
        SAD_index[i] = np.argmin(SAD_matrix[i, :])
        RMSE_abundance[i] = np.min(RMSE_matrix[i, :])
        SAD_endmember[i] = np.min(SAD_matrix[i, :])
    
    A_est[np.arange(R), : ,:] = A_est[RMSE_index, :, :]
    M_est[:, np.arange(R)] = M_est[:, SAD_index]

    return A_est, M_est, RMSE_abundance, SAD_endmember

# plot abundance
def plot_abundance(A_est, A_gt):
    R = A_est.shape[0]
    plt.figure(num="Abundances | top: estimated | bottom: ground truth")
    for i in range(0, R):

        plt.subplot(2, R, i + 1)
        plt.imshow(A_est[i, :, :], cmap="jet")
        plt.axis("off")

        plt.subplot(2, R, R + i + 1)
        plt.imshow(A_gt[i, :, :], cmap="jet")
        plt.axis("off")
    plt.show()

# plot endmember
def plot_endmember(M_est, M_gt):
    R = M_est.shape[1]
    plt.figure(num="Endmembers")
    for i in range(0, R):
        plt.subplot(2, math.ceil(R / 2), i + 1)
        plt.plot(M_est[:, i], color="b", label = "M-est")
        plt.plot(M_gt[:, i], color="r", label = "M-gt")
        plt.legend(loc=1)
        plt.xlabel("bands")
        plt.ylabel("reflectance")
    plt.show()

def plot_detectionmap(ADmap, Map_gt):
    plt.figure(num="Detection maps | left: estimated | right: ground truth")
    plt.subplot(1, 2, 1)
    plt.imshow(ADmap, cmap='hot')
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(Map_gt, cmap='gray')
    plt.axis("off")
    plt.show()

# def pca(X, d):
#     N = np.shape(X)[1]
#     xMean = np.mean(X, axis=1, keepdims=True)
#     XZeroMean = X - xMean
#     [U, S, V] = np.linalg.svd((XZeroMean @ XZeroMean.T) / N)
#     Ud = U[:, 0:d]
#     return Ud

def pca(dataMat, topNfeat):
    # dataMat: N x B
    # topNfeat: D
    meanVals = np.mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals
    covMat = np.cov(meanRemoved, rowvar=False)
    eigVals, eigVets = np.linalg.eig(np.mat(covMat))
    eigValInd = np.argsort(eigVals)
    eigValInd = eigValInd[:-(topNfeat + 1):-1]
    redEigVects = eigVets[:, eigValInd]
    lowDDatMat = meanRemoved * redEigVects
    return lowDDatMat

def hyperVca(Y, R):
    '''
    M : [p,N]
    '''
    B, N = np.shape(Y)

    rMean = np.mean(Y, axis=1, keepdims=True)
    RZeroMean = Y - rMean
    U, S, V = np.linalg.svd(RZeroMean @ RZeroMean.T / N)
    Ud = U[:, 0:R]

    Rd = Ud.T @ RZeroMean
    P_R = np.sum(Y ** 2) / N
    P_Rp = np.sum(Rd ** 2) / N + rMean.T @ rMean
    SNR = np.abs(10 * np.log10((P_Rp - (R / B) * P_R) / (P_R - P_Rp)))
    snrEstimate = SNR
    # print('SNR estimate [dB]: %.4f' % SNR[0, 0])
    # Determine which projection to use.
    SNRth = 18 + 10 * np.log(R)

    if SNR > SNRth:
        d = R
        # [Ud, Sd, Vd] = svds((M * M.')/N, d);
        U, S, V = np.linalg.svd(Y @ Y.T / N)
        Ud = U[:, 0:d]
        Xd = Ud.T @ Y
        u = np.mean(Xd, axis=1, keepdims=True)
        # print(Xd.shape, u.shape, N, d)
        Y = Xd /  np.sum(Xd * u , axis=0, keepdims=True)

    else:
        d = R - 1
        r_bar = np.mean(Y.T, axis=0, keepdims=True).T
        Ud = pca(Y, d)

        R_zeroMean = Y - r_bar
        Xd = Ud.T @ R_zeroMean
        # Preallocate memory for speed.
        # c = np.zeros([N, 1])
        # for j in range(N):
        #     c[j] = np.linalg.norm(Xd[:, j], ord=2)
        c = [np.linalg.norm(Xd[:, j], ord=2) for j in range(N)]
        # print(type(c))
        c = np.array(c)
        c = np.max(c, axis=0, keepdims=True) @ np.ones([1, N])
        Y = np.concatenate([Xd, c.reshape(1, -1)])
    e_u = np.zeros([R, 1])
    # print('*',e_u)
    e_u[R - 1, 0] = 1
    A = np.zeros([R, R])
    # idg - Doesntmatch.
    # print (A[:, 0].shape)
    A[:, 0] = e_u[0]
    I = np.eye(R)
    k = np.zeros([N, 1])

    indicies = np.zeros([R, 1])
    for i in range(R):  # i=1:q
        w = np.random.random([R, 1])

        # idg - Oppurtunity for speed up here.
        tmpNumerator = (I - A @ np.linalg.pinv(A)) @ w
        # f = ((I - A * pinv(A)) * w) / (norm(tmpNumerator));
        f = tmpNumerator / np.linalg.norm(tmpNumerator)

        v = f.T @ Y
        k = np.abs(v)

        k = np.argmax(k)
        A[:, i] = Y[:, k]
        indicies[i] = k

    indicies = indicies.astype('int')
    # print(indicies.T)
    if (SNR > SNRth):
        U = Ud @ Xd[:, indicies.T[0]]
    else:
        U = Ud @ Xd[:, indicies.T[0]] + r_bar

    return U, indicies, snrEstimate
