import torch

class GaussianMixtureModel:
    def __init__(self, Clusternum):
        self.Clusternum = Clusternum
        self.w = self.Mu = self.Sigma = None
    
    def gmm_Params(self, z, gamma):
        # Calculate w, Mu, Sigma
        gamma_sum = gamma.sum(axis=0)
        self.w = w = gamma.mean(axis=0)
        self.Mu = Mu = torch.einsum('nk,nr->kr', gamma, z) / gamma_sum[:, None]
        z_centered = torch.sqrt(gamma[:,:,None]) * (z[:,None,:] - Mu[None,:,:])
        self.Sigma = Sigma = torch.einsum('nkr,nkm->krm', z_centered, z_centered) / gamma_sum[:,None,None]
        # Cholesky decomposition
        min_vals =torch.diag(torch.ones(z.shape[1], dtype=torch.float32)).cuda() * 1e-6
        self.L_Cholesky = L_Cholesky =  torch.linalg.cholesky(Sigma + min_vals[None,:,:])
        return w, Mu, Sigma, L_Cholesky
    
    def Calculate_Energy(self, z, w, Mu, Sigma, L_Cholesky):
        z_centered = z[:,None,:] - Mu[None,:,:]
        v = torch.linalg.solve(L_Cholesky, torch.transpose(torch.transpose(z_centered,0,1),1,2))

        diag_L_Cholesky = torch.zeros(L_Cholesky.shape[0], L_Cholesky.shape[1]).cuda()
        for i in range(L_Cholesky.shape[0]):
            diag_L_Cholesky[i,:] = torch.diag(L_Cholesky[i,:,:])

        log_det_Sigma = 2.0 * torch.sum(torch.log(diag_L_Cholesky), axis=1)
        d = z.shape[1]
        logits = torch.log(w[:,None]) - 0.5 * (torch.sum(torch.square(v), axis=1) + d * torch.log(2.0 * torch.tensor(torch.pi).cuda()) + log_det_Sigma[:,None])
        energies = -torch.logsumexp(logits, axis=0)

        return energies
    
    def Cov_Diag_Loss(self, Sigma):
        diag_Sigma = torch.zeros(Sigma.shape[0], Sigma.shape[1]).cuda()
        for i in range(self.L_Cholesky.shape[0]):
            diag_Sigma[i,:] = torch.diag(Sigma[i,:,:])
        diag_loss = torch.sum(1 / diag_Sigma)
        return diag_loss
    