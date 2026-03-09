import numpy as np
def ece_score(y_true, y_pred, M=10):
    bins = np.linspace(0,1,M+1)
    ece = 0.0
    for i in range(M):
        idx = (y_pred>=bins[i]) & (y_pred<bins[i+1])
        if idx.sum()==0: continue
        ece += idx.mean() * abs(y_true[idx].mean() - y_pred[idx].mean())
    return float(ece)

def conformal_width(y_true, y_pred, alpha=0.1):
    resid = np.abs(y_true - y_pred)
    q = np.quantile(resid, 1-alpha)
    return float(2*q)