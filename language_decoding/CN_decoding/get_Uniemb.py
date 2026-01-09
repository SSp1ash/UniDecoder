import glob
from models.fMRIcortical_MLP_new_multisubject_v2 import fMRI_decoding
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

train_num = 49736
test_num = 712


def remove_zero_rows(input_tensor, corresponding_tensor):
    zero_streaks = torch.sum(input_tensor == 0, dim=1)
    non_zero_rows = torch.where(zero_streaks < 1000)[0]
    input_tensor = input_tensor[non_zero_rows]
    corresponding_tensor = corresponding_tensor[non_zero_rows]
    return input_tensor, corresponding_tensor

if __name__ == '__main__':
    with torch.no_grad():
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

        vox = 13201

        fMRI = torch.load('./CN001_test.pth').to(device)
        eb = torch.load('./eb_test.pth', map_location='cpu').to(device)

        model = fMRI_decoding(vox,1536,num_subjects=1)


        model.load_state_dict(torch.load('./checkpoint.pth', map_location='cpu'))
        model.eval().to(device)

        output = model(fMRI)
        torch.save(output.cpu(), './Decoded_story_run.pth')
        eval_indicators(output, eb)

