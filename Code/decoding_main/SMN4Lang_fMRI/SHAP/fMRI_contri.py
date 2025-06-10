import shap
import glob
from models.fMRIcortical_MLP_new_multisubject import fMRI_decoding
import torch.nn as nn
import numpy as np
import torch
from scipy.stats import pearsonr
import torch.nn.functional as F
from utils.extract_roi import extract_speech, roivox2para
import config
from utils.make_delay import make_delayed
from utils import clip_loss
from scipy.stats import zscore
import config


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

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    # sub_list = ['01','02','03','04','05','06','07','08','09','10','11','12']
    sub_list = ['01']

    for sub_num in sub_list:
        roi = extract_speech(config.project_lfs_path + r'/SMN4Lang/CN001.aparc.a2009s.32k_fs_LR.dlabel.nii')
        vox = roi.__len__() * 4

        # # fMRI load################################
        # fMRI = torch.load(config.project_lfs_path + f'/SMN4Lang/dataset/fMRI/fMRI_sub{sub_num}_roi.pth',
        #                   map_location='cpu')
        # eb = torch.load(config.project_lfs_path + '/SMN4Lang/dataset/fMRI/bloom1.1_dataclean_di20layer_split15.pth', map_location='cpu')
        #
        # sorted_files = sorted(glob.glob(
        #     config.project_lfs_path + "/SMN4Lang/dataset/bloom_word_times_dataclean/word_times_story*.npy"),
        #                       key=lambda x: int(x.split("word_times_story")[-1].split(".npy")[0]))
        # word_all_story = [np.load(f).shape[0] for f in sorted_files]
        #
        # fMRI = fMRI.numpy()
        # fMRI_list = []
        # idx = 0
        # for num in word_all_story:
        #     fMRI_it = make_delayed(fMRI[idx:idx + num], [-4, -3, -2, -1])
        #     idx = idx + num
        #     fMRI_list.append(fMRI_it)
        # fMRI = np.concatenate(fMRI_list)
        # fMRI = torch.from_numpy(fMRI)
        # # dataclean
        # fMRI, eb = remove_zero_rows(fMRI, eb)
        # # fMRI = fMRI.numpy()
        # fMRI = zscore(fMRI)
        #
        # word_all_story_new = [(i - 4) for i in word_all_story]
        # cumulative_lengths = [sum(word_all_story_new[:i]) for i in range(1, 61)]
        # start_index = cumulative_lengths[56]
        # end_index = cumulative_lengths[57]
        # fMRI_test, eb_test = fMRI[start_index:end_index].to(device).float()[:, None, ...], eb[start_index:end_index].to(device).float()
        #
        # # fMRI load################################
        #
        # fmri = fMRI_test
        # eb = eb_test
        # torch.save(fmri,'shap_test_fmri.pth')
        # torch.save(eb,'shape_test_eb.pth')
        fmri = torch.load('./shap_test_fmri.pth')
        eb = torch.load('./shape_test_eb.pth')

        model = fMRI_decoding(vox, 1536,num_subjects=1)

        key = f'*sub{sub_num}'

        model_path = glob.glob(config.project_lfs_path + f'/SMN4Lang/checkpoint/fMRI/bloom1.1/largebatch/{key}/epoch199*')[0]
        model.load_state_dict(torch.load(model_path, map_location='cpu'))


        model.eval().to(device)
        model_net1 = model.net1

        out = model(fmri)
        model = model.cpu()
        fmri = fmri.cpu()

        fmri = fmri[:10]


        class ModelWrapper(nn.Module):
            def __init__(self, base_model):
                super(ModelWrapper, self).__init__()
                self.base_model = base_model

            def forward(self, x):
                return self.base_model(x).mean(dim=-1, keepdim=True)

        wrapped_model = ModelWrapper(model_net1)

        # fmri_para = roivox2para(fmri)

        # 使用 SHAP 的 GradientExplainer 来计算贡献度
        explainer = shap.GradientExplainer(wrapped_model, fmri)

        # 计算 SHAP 值
        shap_values = explainer.shap_values(fmri)

        shap_values_para1 = roivox2para(shap_values[...,:13201])
        shap_values_para2 = roivox2para(shap_values[...,13201:13201*2])
        shap_values_para3 = roivox2para(shap_values[...,13201*2:13201*3])
        shap_values_para4 = roivox2para(shap_values[...,13201*3:])

        shap_values_para = shap_values_para1 + shap_values_para2 + shap_values_para3 + shap_values_para4


        # 可视化 SHAP 值
        shap.summary_plot(shap_values.squeeze(), torch.squeeze(fmri))