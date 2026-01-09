import torch
import config
import glob

if __name__ == '__main__':
    path = glob.glob(config.project_lfs_path + '/SMN4Lang/dataset/fMRI/*roi.pth')
    for p in path:
        data = torch.load(p)
        data = data.float()
        torch.save(data,p)
