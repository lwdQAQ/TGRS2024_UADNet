import numpy as np
import scipy.io as sio

def load_hsi(datalabel):
    '''

    '''
    if datalabel == 'synthetic':
        data = sio.loadmat('./dataset/synthetic_dataset.mat')
        Nr = 100
        Nc = 100
        M_vca = data['M1']
        M_gt = data['M']
        Map_gt = data['map']
        Y = data['Y'].astype(np.float32)
        A_gt = data['A'].astype(np.float32)
        return Y, A_gt, M_gt, M_vca, Map_gt, Nr, Nc
    
    elif datalabel == 'hydice':
        data = sio.loadmat('./dataset/hydice_dataset.mat')
        Nr = 100
        Nc = 80
        M_vca = data['M1']
        M_gt = data['M']
        Map_gt = data['map']
        Y = data['Y'].astype(np.float32)
        A_gt = data['A'].astype(np.float32)
        return Y, A_gt, M_gt, M_vca, Map_gt, Nr, Nc
    
    elif datalabel == 'hydice2':
        data = sio.loadmat('./dataset/hydice2_dataset.mat')
        Nr = 100
        Nc = 80
        M_vca = data['M1']
        M_gt = data['M']
        Map_gt = data['map']
        Y = data['Y'].astype(np.float32)
        A_gt = data['A'].astype(np.float32)
        return Y, A_gt, M_gt, M_vca, Map_gt, Nr, Nc
    
    elif datalabel == 'coast':
        data = sio.loadmat('./dataset/coast_dataset.mat')
        Nr = 100
        Nc = 100
        M_vca = data['M1']
        Map_gt = data['map']
        Y = data['Y'].astype(np.float32)
        return Y, M_vca, Map_gt, Nr, Nc
    
    elif datalabel == 'sandiego':
        data = sio.loadmat('./dataset/sandiego_dataset.mat')
        Nr = 100
        Nc = 100
        M_vca = data['M1']
        Map_gt = data['map']
        Y = data['Y'].astype(np.float32)
        return Y, M_vca, Map_gt, Nr, Nc
    
    elif datalabel == 'pavia':
        data = sio.loadmat('./dataset/pavia_dataset.mat')
        Nr = 100
        Nc = 100
        M_vca = data['M1']
        Map_gt = data['map']
        Y = data['Y'].astype(np.float32)
        return Y, M_vca, Map_gt, Nr, Nc
    
    # Y = Y.astype(np.float32)
    # A_gt = A_gt.astype(np.float32)

if __name__=='__main__':
    datalabels = ['synthetic', 'hydice']
    datalabel = datalabels[0]

    Y, A_gt, M_gt, M_vca, Map_gt, Nr, Nc = load_hsi(datalabel)
    B = Y.shape[0]
    R = M_gt.shape[1]

    from matplotlib import pyplot as plt
    plt.plot(M_gt)
    plt.show()