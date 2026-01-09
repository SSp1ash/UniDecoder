import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class BrainEncoder(nn.Module):
    def __init__(self, channel, time, output_dim=1536, dropout_rate=0.3, num_subjects=1):
        super(BrainEncoder, self).__init__()
        input_dim = channel * time
        self.net1 = RidgeRegression(input_dim, output_dim)
        self.net2 = BrainNetwork(h=output_dim, out_dim=output_dim, drop=dropout_rate)

        self.subject_wise_linear = nn.ModuleList(
            [nn.Linear(output_dim, output_dim) for _ in range(num_subjects)])

        self.layer_norm = nn.LayerNorm(output_dim)
        self.output_layer_norm = nn.LayerNorm(output_dim)


    def forward(self, x, sub=0):
        x = x.reshape([x.shape[0], -1])

        x = self.net1(x)

        x = self.subject_wise_linear[sub](x)
        x = self.layer_norm(x)

        x = self.net2(x)

        x = self.output_layer_norm(x)
        return x


class RidgeRegression(nn.Module):
    # Add weight_decay when initializing optimizer to enable regularization
    def __init__(self, input_size, out_feature):
        super(RidgeRegression, self).__init__()
        self.out_feature = out_feature
        self.linear = nn.Linear(input_size, out_feature)

    def forward(self, x):
        out = self.linear(x.squeeze(1))
        return out.unsqueeze(1)


class BrainNetwork(nn.Module):
    def __init__(self, h=4096, in_dim=15724, out_dim=768, seq_len=1, n_blocks=2, drop=0.3, clip_size=768):
        super().__init__()
        self.seq_len = seq_len
        self.h = h
        self.clip_size = clip_size

        # Mixer blocks - Reduced number of blocks to avoid overfitting
        self.mixer_blocks1 = nn.ModuleList([
            self.mixer_block1(h, drop) for _ in range(n_blocks)
        ])

        # Output linear layer
        self.backbone_linear = nn.Linear(h * seq_len, out_dim, bias=True)
        self.layer_norm = nn.LayerNorm(out_dim)  # Add LayerNorm for stabilization

    def mlp(self, in_dim, out_dim, drop):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(out_dim, out_dim),
        )

    def mixer_block1(self, h, drop):
        return nn.Sequential(
            nn.LayerNorm(h),
            self.mlp(h, h, drop),  # Token mixing
        )

    def forward(self, x):
        residual1 = x

        # Mixer blocks with residual connection and LayerNorm
        for block1 in self.mixer_blocks1:
            x = block1(x) + residual1
            x = F.gelu(x)
            residual1 = x

        x = x.reshape(x.size(0), -1)
        x = self.backbone_linear(x)
        x = self.layer_norm(x)  # Apply LayerNorm before output

        return x


if __name__ == '__main__':
    import time
    device = 'cuda:1'
    model = BrainEncoder(317, 124).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)

    x = torch.randn([30, 317, 124]).to(device)
    t1 = time.time()
    x = model(x).to(device)
    print(time.time() - t1)
    print(x.shape)
