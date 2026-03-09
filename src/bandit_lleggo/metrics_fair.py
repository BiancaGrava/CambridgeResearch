import numpy as np
def tpr_gap(y_true, mask, groups):
    pos = (y_true == 1)
    tprs = []
    for g in np.unique(groups):
        gidx = (groups==g)
        tp = np.sum(pos & gidx & mask)
        pp = np.sum(pos & gidx)
        if pp == 0: continue
        tprs.append(tp / pp)
    return float(max(tprs)-min(tprs)) if len(tprs)>1 else 0.0