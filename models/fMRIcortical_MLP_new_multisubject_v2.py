import torch
import torch.nn as nn
import torch.nn.functional as F


class fMRI_decoding(nn.Module):
    def __init__(self, input_dim, output_dim, num_subjects=1):
        super(fMRI_decoding, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )
        self.subject_wise_linear = nn.ModuleList(
            [nn.Linear(output_dim, output_dim) for _ in range(num_subjects)])
        self.net2 = BrainNetwork(h=output_dim, out_dim=output_dim, n_blocks=2,
                                 drop=0.3)  # Reduced blocks, increased dropout
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x, sub=0):

        x = self.encoder(x)
        x = self.subject_wise_linear[sub](x)
        x = self.layer_norm(x)
        x = self.net2(x)
        return x


class BrainNetwork(nn.Module):
    def __init__(self, h=4096, out_dim=768, seq_len=1, n_blocks=2, drop=0.3, clip_size=768):
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
    vox = 13000
    hidden = 1536  # Reduced hidden size to avoid overfitting
    model = fMRI_decoding(vox, hidden)
    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)
    x = torch.randn([2, 1, vox])
    x = model(x)
    print(x.shape)
