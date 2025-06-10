import torch
import torch.nn as nn
import numpy as np


class fMRI_MAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=4096, bottleneck_dim=1536, mask_ratio=0.1):
        super(fMRI_MAE, self).__init__()

        # 基本参数
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.mask_ratio = mask_ratio

        # 编码器 (分层MLP架构: input_dim -> hidden_dim -> bottleneck_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.ReLU()
        )

        # 解码器 (分层MLP架构: bottleneck_dim -> hidden_dim -> input_dim)
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward_feature(self, x):
        feature = self.encoder(x)
        return feature

    def forward(self, x):
        # x: [B, 1, input_dim]
        B = x.shape[0]

        # 重塑为[B, input_dim]
        x_flat = x.view(B, -1)

        # 创建掩码 (1表示被掩码，0表示保留)
        mask = self.generate_mask(B, self.input_dim)

        # 应用掩码 (被掩码的位置设为0)
        x_masked = x_flat * (1 - mask)

        # 编码和解码
        encoded = self.encoder(x_masked)
        decoded = self.decoder(encoded)

        # 计算重建误差时只考虑被掩码的部分
        if self.training:
            # 在训练中，只关注重建被掩码的部分
            loss = ((decoded - x_flat) * mask).pow(2).mean()
            # 可以在此处返回loss作为第二个返回值

        # 恢复原始形状
        return decoded.view(B, 1, self.input_dim)

    def generate_mask(self, batch_size, seq_length):
        """
        生成随机掩码
        1表示被掩码的位置，0表示保留的位置
        """
        device = next(self.parameters()).device

        # 为每个样本创建掩码
        mask = torch.zeros(batch_size, seq_length, device=device)

        # 随机确定要掩码的位置
        for i in range(batch_size):
            # 随机选择要掩码的索引
            perm = torch.randperm(seq_length, device=device)
            idx = perm[:int(seq_length * self.mask_ratio)]
            mask[i, idx] = 1.0

        return mask


if __name__ == '__main__':
    x = torch.rand(32, 1, 13000)
    MAE = fMRI_MAE(input_dim=13000, hidden_dim=4096, bottleneck_dim=1536)
    parm = sum(p.numel() for p in MAE.parameters() if p.requires_grad)
    print(f"参数数量: {parm}")
    y = MAE(x)
    print(y.shape)