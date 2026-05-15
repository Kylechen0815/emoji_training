import torch
import torch.nn as nn
import torch.nn.functional as F

class FacialLandmarkCNN(nn.Module):
    def __init__(self):
        super(FacialLandmarkCNN, self).__init__()
        
        # 卷積層：提取特徵 (Input: 3, 128, 128)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # 池化層：壓縮空間維度
        self.pool = nn.MaxPool2d(2, 2)
        
        # 全連接層：將特徵轉換為座標 (128*8*8 -> 512 -> 10)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10) # 輸出 5 個點的 (x, y)

    def forward(self, x):
        # 數據流向：Conv -> ReLU -> Pool
        x = self.pool(F.relu(self.conv1(x))) # 64x64
        x = self.pool(F.relu(self.conv2(x))) # 32x32
        x = self.pool(F.relu(self.conv3(x))) # 16x16
        x = self.pool(F.relu(self.conv4(x))) # 8x8
        
        # 展平 (Flatten)
        x = x.view(-1, 128 * 8 * 8)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # 輸出 10 個座標值
        return x