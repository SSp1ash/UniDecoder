import torch
def pearson_correlation(x, y):
    # 确保输入是 1D 向量
    if x.dim() != 1 or y.dim() != 1:
        raise ValueError("Input tensors must be 1D vectors.")

    # 计算均值
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)

    # 计算去均值后的向量
    xm = x - mean_x
    ym = y - mean_y

    # 计算皮尔逊相关系数
    r_num = torch.sum(xm * ym)
    r_den = torch.sqrt(torch.sum(xm ** 2) * torch.sum(ym ** 2))

    return r_num / r_den