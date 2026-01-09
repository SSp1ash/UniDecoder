import torch
import torch.nn as nn
import torch.optim as optim


class LanguageClassifier2(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LanguageClassifier2, self).__init__()


        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)  #
        self.fc3 = nn.Linear(512, num_classes)  #


        self.res_fc1 = nn.Linear(input_dim, 1024)  #
        self.res_fc2 = nn.Linear(1024, 512)  #


        self.relu = nn.ReLU()


        self.dropout = nn.Dropout(0.5)

    def forward(self, x):

        residual1 = self.res_fc1(x)
        x = self.relu(self.fc1(x))
        x = x + residual1

        # Dropout
        x = self.dropout(x)


        residual2 = self.res_fc2(x)
        x = self.relu(self.fc2(x))
        x = x + residual2


        x = self.dropout(x)


        x = self.fc3(x)

        return x


class LanguageClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LanguageClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)

if __name__ == '__main__':

    num_languages = 5
    embedding_dim = 1536
    batch_size = 100

    model = LanguageClassifier(input_dim=embedding_dim, num_classes=num_languages)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()

        # for inputs, labels in train_loader:

        inputs = torch.randn(batch_size, embedding_dim)
        labels = torch.randint(0, num_languages, (batch_size,))

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 打印损失
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    model.eval()
    with torch.no_grad():


        inputs = torch.randn(batch_size, embedding_dim)

        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        print(f'Predicted Languages: {predicted}')
