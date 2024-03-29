import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import random
import time
import torchvision.transforms as transforms
from load_hsi import load_hsi
from net import UADNet
from gmm import GaussianMixtureModel
from utils import norm_abundance, norm_endmember, norm_energymap, arange_A_E, plot_abundance, plot_endmember, plot_detectionmap, pca
from post_processing import Post_processing

def SADloss(output, target):

    _, band, h, w = output.shape
    output = torch.reshape(output, (band, h * w))
    target = torch.reshape(target, (band, h * w))
    LossSAD = torch.acos(torch.cosine_similarity(output, target, dim=0))
    LossSAD = torch.mean(LossSAD)
    return LossSAD

def SADloss1(y, yhat):
    LossSAD = torch.acos(torch.cosine_similarity(y, yhat, dim=1))
    LossSAD = torch.mean(LossSAD)
    return LossSAD

def Sparseloss(a):
    _, R, Nr, Nc = a.shape
    a = torch.reshape(a, (R, Nr * Nc))
    LossSparse = torch.norm(a, p=2) / (Nr * Nc)
    return LossSparse

def EndmemberTVloss(M):
    B, R, _, _ = M.shape
    M_temp = torch.squeeze(M)
    M1 = M_temp[0:B-1, :]
    M2 = M_temp[1:B, :]
    M_TV = torch.abs(M1 - M2)
    LossTVM = torch.sum(torch.mean(M_TV, dim=0))
    return LossTVM

def AbundanceTVloss(A):
    print(A.shape)
    LossTVA = 0
    return LossTVA

MSE = torch.nn.MSELoss(size_average=True)

MAE = torch.nn.L1Loss(reduction = 'mean')

def Reconstructionloss(y, yhat, lambda_mse):
    LossRecon = SADloss1(y, yhat) + lambda_mse * MSE(y, yhat)
    return LossRecon

class load_data(torch.utils.data.Dataset):
    def __init__(self, img, pca_img, transform=None):
        self.img = img.float()
        self.pca_img = pca_img.float()
        self.transform = transform

    def __getitem__(self, idx):
        return self.img, self.pca_img

    def __len__(self):
        return 1

seed = 1
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

datalabels = [ 'hydice']
datalabel = datalabels[0]

Y, A_gt, M_gt, M_vca, Map_gt, Nr, Nc = load_hsi(datalabel)
N = Nr * Nc
B = Y.shape[0]
R = M_gt.shape[1]

K = 4
lambda_mse = 0.5
lambda_sparse = 0.1
lambda_energy = 0.01
lambda_cov = 0.0001

batchsize = 1

A_gt = torch.from_numpy(A_gt)  # abundance GT
nd = 2
Y_pca = pca(Y.T, nd)
Y_pca = Y_pca.T
Y = torch.from_numpy(Y)  # HSI
Y = torch.reshape(Y, (B, Nr, Nc))
Y_pca = torch.from_numpy(Y_pca)  # PCA HSI
Y_pca = torch.reshape(Y_pca, (nd, Nr, Nc))
A_gt = torch.reshape(A_gt, (R, Nr, Nc))
M_init = torch.from_numpy(M_vca).unsqueeze(2).unsqueeze(3).float()

gmm = GaussianMixtureModel(K)

def train():
    net = UADNet(R, B, K, drop_out=0.2, use_ALDR=1).cuda()

    nn.init.kaiming_normal_(net.Encoder[0].weight.data)
    nn.init.kaiming_normal_(net.Encoder[4].weight.data)
    nn.init.kaiming_normal_(net.Encoder[8].weight.data)
    nn.init.kaiming_normal_(net.Cluster[0].weight.data)
    nn.init.kaiming_normal_(net.Cluster[4].weight.data)
    nn.init.kaiming_normal_(net.Cluster[8].weight.data)
    net.Decoder.weight.data = M_init.cuda()

    train_y = load_data(img=Y, pca_img=Y_pca, transform=transforms.ToTensor())
    train_y = torch.utils.data.DataLoader(dataset=train_y, batch_size=batchsize, shuffle=False)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)

    print("Start training!")
    for epoch in range(1000):
        for i, (y, y_pca)in enumerate(train_y):
            y = y.cuda()
            y_pca = y_pca.cuda()
            a, a_aldr, y_hat, c = net(y, y_pca)

            d = a_aldr.shape[1]
            z = torch.squeeze(a_aldr)
            z = torch.reshape(z, (d, N))
            z = z.T

            c = torch.squeeze(c)
            c = torch.reshape(c, (K, N))
            c = c.T

            w, Mu, Sigma, L_Cholesky = gmm.gmm_Params(z, c)
            energy = gmm.Calculate_Energy(z, w, Mu, Sigma, L_Cholesky)
            e_sorted, indices = torch.sort(energy, descending=True)
            anomaly_index = indices[0:(N // 100)]
            nonanomaly_index = indices[(N // 100):N]

            y1 = torch.squeeze(y)
            y1 = torch.reshape(y1, (B, N))
            y1 = y1.T
            y_abnormal = y1[anomaly_index , :]
            y_normal = y1[nonanomaly_index , :]

            y_hat1 = torch.squeeze(y_hat)
            y_hat1 = torch.reshape(y_hat1, (B, N))
            y_hat1 = y_hat1.T
            yhat_abnormal = y_hat1[anomaly_index , :]
            yhat_normal = y_hat1[nonanomaly_index , :]

            M = net.Decoder.weight.data
            LossTVM = EndmemberTVloss(M)
            
            LossEnergy = torch.mean(energy)
            LossCov = gmm.Cov_Diag_Loss(Sigma)

            LossUnmixing = 0.9 * Reconstructionloss(y_normal, yhat_normal, lambda_mse) + 0.1 * Reconstructionloss(y_abnormal, yhat_abnormal, lambda_mse) + 0.1 * LossTVM
            LossClustering = (lambda_energy * LossEnergy + lambda_cov * LossCov) / lambda_energy
            Loss = LossUnmixing + lambda_energy * LossClustering

            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()
            scheduler.step()
        if epoch % 100 == 0:
            print("Epoch:", epoch, 
                  "| Loss: %.4f" % Loss.cpu().data.numpy(),
                  "| Unmixing Loss: %.4f" % LossUnmixing.cpu().data.numpy(),
                  "| Clustering Loss: %.4f" % LossClustering.cpu().data.numpy())
            
    torch.save(net.state_dict(), './weight/weights_hydice.pt')

# train
t1 = time.time()
train()
# test
print('Start test!')
net = UADNet(R, B, K, drop_out=0.2, use_ALDR=1).cuda()
net.load_state_dict(torch.load('./weight/weights_hydice.pt'))
net.eval()
with torch.no_grad():
    A_est, A_aldr, Y_hat, C_est = net(torch.unsqueeze(Y, 0).cuda(),
                                 torch.unsqueeze(Y_pca, 0).cuda().float())

    d = A_aldr.shape[1]
    Z_est = torch.squeeze(A_aldr)
    Z_est = torch.reshape(Z_est, (d, N))
    Z_est = Z_est.T

    C_est = torch.squeeze(C_est)
    C_est = torch.reshape(C_est, (K, N))
    C_est = C_est.T

    w, Mu, Sigma, L_Cholesky = gmm.gmm_Params(Z_est, C_est)
    E_est = gmm.Calculate_Energy(Z_est, w, Mu, Sigma, L_Cholesky)

M_est = net.state_dict()["Decoder.weight"].cpu().numpy()
M_est = np.mean(np.mean(M_est, -1), -1)
A_est, A_gt = norm_abundance(A_est, A_gt)
M_est, M_gt = norm_endmember(M_est, M_gt)
A_est, M_est, RMSE_abundance, SAD_endmember = arange_A_E(A_est, A_gt, M_est, M_gt)

E_est = norm_energymap(E_est)
E_est = E_est.cpu().numpy()

# Post-process
ADmap, AUC = Post_processing(E_est, Map_gt, Nr, Nc)
t2 = time.time()
print(t2-t1)

print("RMSE", RMSE_abundance)
print("mean_RMSE", RMSE_abundance.mean())
print("endmember_SAD", SAD_endmember)
print("mean_SAD", SAD_endmember.mean())
print("AUC", AUC)

sio.savemat('./result/' + 'results_hydice.mat',{'A_est':A_est,
                                                   'M_est':M_est,
                                                   'Map':Map_gt,
                                                   'ADmap':ADmap,
                                                   })

import matplotlib.pyplot as plt
plt.figure(num='Energy map')
plt.imshow(E_est.reshape(Nr, Nc), cmap='hot')
plt.axis("off")
plt.show()
plot_abundance(A_est, A_gt)
plot_endmember(M_est, M_gt)
plot_detectionmap(ADmap, Map_gt)