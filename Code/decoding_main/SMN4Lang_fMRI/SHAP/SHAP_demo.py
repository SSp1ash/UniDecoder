import torch
import torchvision.transforms as transforms
import shap
import numpy as np
from torchvision.models import resnet18  # 使用一个预训练模型作为示例

device = 'cuda:0'

# 加载预训练的CNN模型
model = resnet18(pretrained=True).to(device)
model.eval()  # 确保模型处于评估模式

# 假设你已经有预处理的测试数据
# 这里我们用一个随机生成的图像数据作为示例
# 假设输入数据是3通道的224x224图像
X_test = np.random.rand(10, 3, 224, 224).astype(np.float32)

# 将numpy数组转换为torch tensor
X_test_tensor = torch.tensor(X_test, requires_grad=True).to(device)

# 使用GradientExplainer
background = X_test_tensor[:5]  # 使用部分数据作为背景
explainer = shap.GradientExplainer(model, background)

# SHAP 解释值计算
shap_values, indexes = explainer.shap_values(X_test_tensor, ranked_outputs=1)

# 使用适合图像数据的SHAP绘图
shap.image_plot([np.swapaxes(np.swapaxes(sv, 1, -1), 1, 2) for sv in shap_values],
                X_test_tensor.cpu().detach().numpy())
