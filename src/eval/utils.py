import numpy as np
from sklearn.metrics import r2_score
import scipy.stats as stats
import torch

def compute_r2(y_trues, y_preds):
    return np.mean([r2_score(y_trues[i], y_preds[i]) for i in range(y_trues.shape[0])])

def compute_pearsonr(y_trues, y_preds):
    return np.mean([stats.pearsonr(y_trues[i, :], y_preds[i, :])[0] for i in range(y_trues.shape[0])])



### OLD CODE ###
# calculate r2 and spearman given y pred and y true
def compute_r2_spearman(y_true, y_pred):

    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)

    # r2 is easy we average across samples
    r2 = r2_score(y_true, y_pred, multioutput='uniform_average')

    # spearman we have to query each gene
    spearman = []
    for i in range(y_true.shape[1]):
        corr, _ = stats.spearmanr(y_true[:, i], y_pred[:, i])
        spearman.append(corr)
    mean_spearman = np.nanmean(spearman)

    return r2, mean_spearman

def l1_normalize(x, eps=1e-8):
    return x / (x.sum(dim=1, keepdim=True) + eps)

def zinb_loss(x, mean, dispersion, pi, eps=1e-8):
    t1 = torch.lgamma(dispersion + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + dispersion + eps)
    t2 = (dispersion + eps) * torch.log(1.0 + (mean / dispersion) + eps)
    t3 = x * (torch.log(dispersion + eps) - torch.log(mean + eps))
    nb_case = t1 + t2 + t3

    zero_case = -torch.log(pi + (1.0 - pi) * torch.exp(-nb_case) + eps)

    result = torch.where(x < 1e-8, zero_case, -torch.log(1.0 - pi + eps) + nb_case)

    return torch.mean(result)
