import torch
import torch.nn as nn
import torch.optim as optim


class LanguageClassifier2(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LanguageClassifier2, self).__init__()

        # 第一层隐藏层：从输入维度到1024
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)  # 第二层隐藏层：从1024到512
        self.fc3 = nn.Linear(512, num_classes)  # 输出层：从512到类别数量

        # 对残差连接进行维度匹配的线性层
        self.res_fc1 = nn.Linear(input_dim, 1024)  # 用于对fc1的残差连接进行维度匹配
        self.res_fc2 = nn.Linear(1024, 512)  # 用于对fc2的残差连接进行维度匹配

        # 激活函数
        self.relu = nn.ReLU()

        # Dropout层，防止过拟合
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 第一层：加入残差连接
        residual1 = self.res_fc1(x)  # 残差部分
        x = self.relu(self.fc1(x))  # 通过fc1
        x = x + residual1  # 加入残差连接（skip connection）

        # Dropout层
        x = self.dropout(x)

        # 第二层：加入残差连接
        residual2 = self.res_fc2(x)  # 残差部分
        x = self.relu(self.fc2(x))  # 通过fc2
        x = x + residual2  # 加入残差连接（skip connection）

        # Dropout层
        x = self.dropout(x)

        # 输出层
        x = self.fc3(x)  # 最后一层输出

        return x  # 输出未经Softmax的logits


class LanguageClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LanguageClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)  # 输入维度到1024的隐藏层
        self.fc2 = nn.Linear(1024, 512)  # 1024到512的隐藏层
        self.fc3 = nn.Linear(512, num_classes)  # 512到语言类别的输出层
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # 使用Softmax得到每个语言的概率分布

    def forward(self, x):
        x = self.relu(self.fc1(x))  # 第一个隐藏层
        x = self.relu(self.fc2(x))  # 第二个隐藏层
        x = self.fc3(x)  # 输出层，直接给出logits
        return self.softmax(x)  # 使用softmax得到概率分布

if __name__ == '__main__':
    # 假设我们有3个语言类别：英语、中文、法语、荷兰语、其他
    num_languages = 5
    embedding_dim = 1536  # 输入向量的维度
    batch_size = 100  # 假设一次处理100个样本
    # 初始化模型、损失函数和优化器
    model = LanguageClassifier(input_dim=embedding_dim, num_classes=num_languages)
    criterion = nn.CrossEntropyLoss()  # 多分类交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 假设你有数据加载器
    # train_loader = DataLoader(...)

    # 训练过程示例
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        # 假设你有一个数据加载器返回(batch_size, 1536)的输入向量和标签
        # for inputs, labels in train_loader:
        # 这里我们用随机数据来示范
        inputs = torch.randn(batch_size, embedding_dim)  # 随机生成输入数据
        labels = torch.randint(0, num_languages, (batch_size,))  # 随机生成标签

        # 前向传播
        outputs = model(inputs)  # 得到预测的概率分布

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新模型参数

        # 打印损失
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 假设你有测试数据
    # test_loader = DataLoader(...)

    # 测试过程示例
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():
        # 假设你有一个数据加载器返回(batch_size, 1536)的输入向量
        # for inputs in test_loader:
        inputs = torch.randn(batch_size, embedding_dim)  # 随机生成输入数据
        # 预测
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)  # 获取最大概率对应的语言类别
        print(f'Predicted Languages: {predicted}')
