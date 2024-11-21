import torch
import torch.nn as nn
import torch.nn.functional as F
class Cov1D(nn.Module):
    def __init__(self):
        super(Cov1D, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=22, out_channels=32, kernel_size=1)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=1)

        self.fc1 = nn.Linear(64*10, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        output = self.fc3(x)
        return output

class Cov2D(nn.Module):
    def __init__(self, num_atoms=10):
        super(Cov2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32) 
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)  
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.fc1 = nn.Linear(64 * 5 * 10 * num_atoms, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        batch_size = x.size(0)
        num_atoms = x.size(1)
        x = x.view(batch_size * num_atoms, 1, 10, 20)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        output = self.fc3(x)
        return output

class Cov3D(nn.Module):
    def __init__(self, num_atoms=10):
        super(Cov3D, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(16) 
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(32)  
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(32 * num_atoms, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        batch_size = x.size(0)
        num_atoms = x.size(1)

        x = x.view(batch_size * num_atoms, 1, 8, 8, 8)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(batch_size, num_atoms, -1)
        x = x.view(batch_size, -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

