import glob
from models.fMRIcortical_MLP_new_multisubject import fMRI_decoding
from models.MEG_MLP_multisub import BrainEncoder
import torch.nn as nn
import numpy as np
import torch
from scipy.stats import pearsonr
import torch.nn.functional as F
from utils.extract_roi import extract_speech
import config
from utils.make_delay import make_delayed
from utils import clip_loss
from scipy.stats import zscore


def compute_rlne(y_pred, y_true):
    l2_norm_diff = torch.norm(y_pred - y_true, p=2)
    l2_norm_true = torch.norm(y_true, p=2)
    # 计算 RLNE
    rlne = l2_norm_diff / l2_norm_true
    return rlne


def cal_pearson(emb_pred,emb_gt):
    p_list = []
    mse_list = []
    for i in range(emb_pred.shape[0]):
        p_val = pearsonr(emb_pred[i],emb_gt[i])[0]
        p_val = np.nan_to_num(p_val, 0)
        p_list.append(p_val)
        mse_list.append(np.mean((emb_gt[i] - emb_pred[i]) ** 2) / np.mean((emb_gt[i] - np.mean(emb_gt[i])) ** 2))
    r = np.mean(p_list)
    mse = np.mean(mse_list)
    return r,mse

class NMSELoss(nn.Module):
    def __init__(self):
        super(NMSELoss, self).__init__()

    def forward(self, prediction, target):
        mse_loss = nn.MSELoss()(prediction, target)
        variance = torch.var(target, unbiased=False)
        nmse_loss = mse_loss / variance
        return nmse_loss

def eval_indicators(output,eb):
    criterion1 = NMSELoss()
    criterion2 = nn.MSELoss()
    criterion3 = clip_loss.ClipLoss()

    print('NMSE:')
    print(criterion1(output, eb).item())

    print('MSE:')
    print(criterion2(output, eb).item())

    print('RLNE:')
    rlne = compute_rlne(output, eb).item()
    print(rlne)

    print('pearson:')
    cos_sim = F.cosine_similarity(output, eb)
    print(cos_sim.mean().item())

    print('CLIP_loss:')
    cliploss = criterion3(output[:, None, :], eb[:, None, :])
    print(cliploss.item())



if __name__ == '__main__':
    path_rec = config.project_lfs_path + '/Result/largebatch/SMN4Lang_fMRI/story58/SMN4Lang_sub02_test_fMRI_20250404.pth'
    path_rec2 = config.project_lfs_path + '/Result/largebatch/SMN4Lang_MEG/story58/SMN4Lang_sub02_test_MEG_20250308.pth'
    path_rec3 = config.project_lfs_path + '/Result/largebatch/SMN4Lang_multi/story58/SMN4Lang_sub02_test_fusion2_20250404.pth'

    path_gt = config.project_lfs_path + '/SMN4Lang/dataset/fMRI/bloom1.1_dataclean_di20layer_split15.pth'

    rec = torch.load(path_rec).cpu()
    rec2 = torch.load(path_rec2).cpu()
    rec3 = torch.load(path_rec3).cpu()

    rec_fusion = rec2
    rec_fusion[:-4] = (rec + rec2[:-4]) / 2


    gt = torch.load(path_gt).cpu()
    story58 = 788
    story59 = 712
    story60 = 712

    # eval_indicators(rec2, gt[-(story58+story59+story60):-(story59+story60)])
    # eval_indicators(rec3, gt[-(story58+story59+story60):-(story59+story60)])
    eval_indicators(rec, gt[-(story58+story59+story60):-(story59+story60)][:-4])
    # eval_indicators(rec_fusion, gt[-(story58+story59+story60):-(story59+story60)])
