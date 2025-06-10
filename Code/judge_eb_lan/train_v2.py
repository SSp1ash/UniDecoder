from models.judge_lan import *
import config
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

if __name__ == '__main__':
    device = 'cuda:4'

    # Load data
    # LPPC-fMRI
    eb_CN1 = torch.load(config.project_lfs_path + '/LPPC-fMRI/dataset/fMRI/eb_bloom1.1_CN_split15_20layer.pth', map_location='cpu')
    eb_FR = torch.load(config.project_lfs_path + '/LPPC-fMRI/dataset/fMRI/eb_bloom1.1_FR_split15_20layer.pth', map_location='cpu')
    eb_EN1 = torch.load(config.project_lfs_path + '/LPPC-fMRI/dataset/fMRI/eb_bloom1.1_FR_split15_20layer.pth', map_location='cpu')
    # SMN4Lang
    eb_CN2 = torch.load(config.project_lfs_path + '/SMN4Lang/dataset/fMRI/bloom1.1_dataclean_di20layer_split15.pth', map_location='cpu')
    # Broderick
    eb_EN2 = torch.load(config.project_lfs_path + '/Broderick2018/dataset/EEG/eb_bloom1.1_EN_split15_20layer.pth', map_location='cpu')

    # eb_NL = torch.load(config.project_lfs_path + '/SparrKULee/dataset/EEG/qwen_dataclean_di20layer_split15.pth', map_location='cpu')
    eb_NL = torch.load(config.project_lfs_path + '/SparrKULee/dataset/EEG/bloom_dataclean_di20layer_split15.pth', map_location='cpu')

    def split_data(eb_data):
        train_size = int(0.6 * len(eb_data))
        train_data = eb_data[:train_size]
        test_data = eb_data[train_size:]
        return train_data, test_data


    eb_CN1_train, eb_CN1_test = split_data(eb_CN1)
    eb_FR_train, eb_FR_test = split_data(eb_FR)
    eb_EN1_train, eb_EN1_test = split_data(eb_EN1)
    eb_CN2_train, eb_CN2_test = split_data(eb_CN2)
    eb_EN2_train, eb_EN2_test = split_data(eb_EN2)
    eb_NL_train, eb_NL_test = split_data(eb_NL)

    inputs_train = [eb_CN1_train, eb_FR_train, eb_EN1_train, eb_CN2_train, eb_EN2_train, eb_NL_train]
    labels_train = [torch.ones(eb_CN1_train.shape[0], 1), torch.full((eb_FR_train.shape[0], 1), 2), torch.zeros(eb_EN1_train.shape[0], 1), torch.ones(eb_CN2_train.shape[0], 1), torch.zeros(eb_EN2_train.shape[0], 1), torch.full((eb_NL_train.shape[0], 1), 3)]

    inputs_test = [eb_CN1_test, eb_FR_test, eb_EN1_test, eb_CN2_test, eb_EN2_test, eb_NL_test]
    labels_test = [torch.ones(eb_CN1_test.shape[0], 1), torch.full((eb_FR_test.shape[0], 1), 2), torch.zeros(eb_EN1_test.shape[0], 1), torch.ones(eb_CN2_test.shape[0], 1), torch.zeros(eb_EN2_test.shape[0], 1), torch.full((eb_NL_test.shape[0], 1), 3)]

    all_inputs_train = torch.cat(inputs_train, dim=0).to(device) # 拼接所有输入数据
    all_labels_train = torch.cat(labels_train, dim=0).to(device).to(torch.int64)  # 拼接所有标签数据

    all_inputs_test = torch.cat(inputs_test, dim=0).to(device)  # 拼接所有输入数据
    all_labels_test = torch.cat(labels_test, dim=0).to(device).to(torch.int64)  # 拼接所有标签数据


    # 将数据和标签封装成一个TensorDataset
    dataset_train = TensorDataset(all_inputs_train, all_labels_train)
    dataset_test = TensorDataset(all_inputs_test, all_labels_test)

    # 使用DataLoader打乱数据
    dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=False)

    # Train
    embedding_dim = 1536
    num_languages = 4
    # model = LanguageClassifier(input_dim=embedding_dim, num_classes=num_languages).to(device)
    model = LanguageClassifier2(input_dim=embedding_dim, num_classes=num_languages).to(device)
    criterion = nn.CrossEntropyLoss()  # 多分类交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    # 使用 StepLR 学习率调度器，每 50 个 epochs 将学习率减少为原来的一半
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    epochs = 40
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        running_loss = 0.0

        for i, data in tqdm(enumerate(dataloader_train), total=len(dataloader_train), desc="Processing Batches"):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # 清空梯度
            outputs = model(inputs)  # 得到预测的概率分布
            loss = criterion(outputs, labels.squeeze())

            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新模型参数

            running_loss += loss.item()

        # Step the scheduler after each epoch
        scheduler.step()

        # 打印每个epoch的损失和当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(dataloader_train):.4f}, Learning Rate: {current_lr:.6f}')

    torch.save(model.state_dict(), './judge_lan.model.real_20250123_v2')


