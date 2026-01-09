import torch
import torch.nn as nn
from functools import partial
from models.eegpt_pretrain import EEGTransformer


def reshape_tensor(tensor):
    # 原始形状
    original_shape = tensor.shape
    # 检查形状是否符合预期
    assert len(original_shape) == 3 and original_shape[2] == 256, "输入张量形状应为[x, 58, 256]"

    # 计算新的batch size
    old_batch = original_shape[0]
    new_batch = (old_batch + 3) // 4  # 向上取整，确保有足够空间

    # 计算需要填充的数量
    padding_count = (new_batch * 4) - old_batch

    # 创建新张量并初始化为0
    new_tensor = torch.zeros((new_batch, original_shape[1], 1024), dtype=tensor.dtype, device=tensor.device)

    # 填充数据
    for i in range(old_batch):
        batch_idx = i // 4  # 新的batch索引
        position = (i % 4) * 256  # 在1024维度上的起始位置
        new_tensor[batch_idx, :, position:position + 256] = tensor[i]

    return new_tensor, padding_count
class FeatureExtra(nn.Module):
    def __init__(self):
        super(FeatureExtra, self).__init__()

        # 创建EEGTransformer模型
        self.eegpt = EEGTransformer(
            img_size=[58, 256 * 4],
            patch_size=32 * 2,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_std=0.02,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            embed_dim=512,
            embed_num=4,
            depth=8,
            num_heads=8)

        # 加载预训练权重
        checkpoint_path = "/home/guoyi/llm_model/eegpt/eegpt_mcae_58chs_4s_large4E.ckpt"
        checkpoint = torch.load(checkpoint_path,map_location='cpu')

        # 获取state_dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # 只保留以"encoder."开头的参数，并去除前缀
        encoder_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('encoder.') or k.startswith('model.encoder.'):
                key = k.replace('encoder.', '').replace('model.', '')
                encoder_state_dict[key] = v

        # 加载权重
        missing_keys, unexpected_keys = self.eegpt.load_state_dict(encoder_state_dict, strict=False)

        print(f"加载encoder权重成功，共加载{len(encoder_state_dict) - len(unexpected_keys)}个参数")
        print(f"模型中未加载权重的参数数量: {len(missing_keys)}")
        print(f"未找到对应模型参数的权重数量: {len(unexpected_keys)}")

        # 冻结参数
        for param in self.eegpt.parameters():
            param.requires_grad = False

    def forward(self, x):
        x, padding_count = reshape_tensor(x)
        x = self.eegpt(x)
        x = torch.reshape(x,[x.shape[0]*4, 16, 512])


        x = x[:x.shape[0]-padding_count]
        return x


class RidgeRegression(nn.Module):
    def __init__(self, input_size, out_feature):
        super(RidgeRegression, self).__init__()
        self.out_feature = out_feature
        self.linear = nn.Linear(input_size, out_feature)

    def forward(self, x):
        out = self.linear(x.squeeze(1))
        return out.unsqueeze(1)


if __name__ == '__main__':
    import time
    device = 'cuda:5'

    model = FeatureExtra().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数数量: {total_params}")

    x = torch.randn([30, 58, 256]).to(device)
    t1 = time.time()
    x = model(x)
    inference_time = time.time() - t1
    print(f"推理时间: {inference_time:.4f}秒")
    print(f"输出形状: {x.shape}")