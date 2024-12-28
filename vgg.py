import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
import numpy as np

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 自定义 PyTorch Dataset 类
class IndianPinesDataset(Dataset):
    def __init__(self, data_path, target_path, transform=None):
        self.data = loadmat(data_path)['indian_pines_corrected']
        self.targets = loadmat(target_path)['indian_pines_gt']
        self.transform = transform

    def __len__(self):
        return self.data.shape[0] * self.data.shape[1]

    def __getitem__(self, idx):
        row = idx // self.data.shape[1]
        col = idx % self.data.shape[1]

        sample = {'input': self.data[row, col], 'label': self.targets[row, col]}

        if self.transform:
            sample = self.transform(sample)

        return sample

# 定义数据预处理
class ToTensor(object):
    def __call__(self, sample):
        input, label = sample['input'], sample['label']
        input = torch.from_numpy(input.astype(np.float32))
        label = torch.tensor(label, dtype=torch.int64)  # 使用 int64 类型表示标签
        return {'input': input, 'label': label}

# 定义数据集路径
data_path = "Indian_pines_corrected.mat"
target_path = "Indian_pines_gt.mat"
# 创建数据集实例
dataset = IndianPinesDataset(data_path, target_path, transform=ToTensor())
# 划分训练集和验证集
train_indices, val_indices = train_test_split(np.arange(len(dataset)), test_size=0.5, random_state=42)

# 创建 DataLoader 实例
train_loader = DataLoader(dataset, batch_size=64, sampler=torch.utils.data.SubsetRandomSampler(train_indices))
val_loader = DataLoader(dataset, batch_size=64, sampler=torch.utils.data.SubsetRandomSampler(val_indices))
data_loader = DataLoader(dataset,batch_size=64,shuffle=False)

#搭建vgg11网络
class VGG11(nn.Module):
    def __init__(self, num_classes=17):
        super(VGG11, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        ).to(device)
        # 计算全连接层输入尺寸
        self.num_flat_features = 512 * 12
        self.classifier = nn.Sequential(
            nn.Linear(self.num_flat_features, 4096),  # 修改线性层输入尺寸
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        ).to(device)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 将特征张量展平
        x = self.classifier(x)
        return x



# 初始化模型
model = VGG11(num_classes=17)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 训练模型
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        inputs, labels = batch['input'].to(device), batch['label'].to(device)
        # imputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs.unsqueeze(1))  # 添加通道维度
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}")

    # 在验证集上评估模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch['input'].to(device), batch['label'].to(device)
            # imputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs.unsqueeze(1)).to(device)  # 添加通道维度
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_acc = correct / total
    print(f"Validation Accuracy: {val_acc:.4f}")

    # 在整体数据上评估模型
    model.eval()
    correct = 0
    total = 0
    pret = np.zeros((145,145))
    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch['input'].to(device), batch['label'].to(device)
            # imputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs.unsqueeze(1)).to(device)  # 添加通道维度
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pret = outputs.tolist()
    total_acc = correct / total
    print(f"Total Accuracy: {total_acc:.4f}")



print("Training finished.")