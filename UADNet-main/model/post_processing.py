import matlab
import matlab.engine
import cv2
import numpy as np

def Post_processing(E, Map_gt, Nr, Nc):
    N = Nr * Nc
    ret, T = cv2.threshold(np.uint8(255*E),0,255,cv2.THRESH_OTSU)

    eng = matlab.engine.start_matlab()

    T = T.reshape(Nr, Nc)
    T = matlab.double(T.tolist())
    T = eng.im2bw(T)

    B = eng.bwareafilt(T, matlab.double([0, N/100])) # default : N/500

    E = E.reshape(Nr, Nc)

    E = matlab.double(E.tolist())
    S = E
    for i in range(Nr):
        for j in range(Nc):
            if B[i][j] == 0:
                S[i][j] = E[i][j] ** 5 # default : 5

    ADmap = eng.imguidedfilter(S)
    Map_gt = matlab.double(Map_gt.tolist())
    eng.cd(r'result', nargout=0)
    AUC = eng.ROC(ADmap, Map_gt, 0) # 1: show ROC curve 0: don't show ROC curves

    return ADmap, AUC