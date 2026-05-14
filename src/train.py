import torch                    # PyTorch 的核心庫，處理張量 (Tensor) 運算
import torch.nn as nn             # 包含神經網路層（如 Conv2d, Linear）與損失函數
import torch.optim as optim       # 包含優化算法（如 Adam, SGD），負責更新參數
from model import FacialLandmarkCNN # 載入你剛寫好的「大腦」模型架構
from dataset import CelebADataset   # 載入你剛寫好的「數據讀取」邏輯
from torchvision import transforms  # 官方提供的影像處理工具（如縮放、轉張量）
from torch.utils.data import DataLoader # 負責將數據打包成 Batch 的管理員


# 偵測是否有 NVIDIA GPU (CUDA)，有的話就用顯卡跑，沒有就用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 學習率：決定模型更新權重的「步幅」。太大會跑過頭，太小會跑太慢
LR = 0.001

# 批次大小：一次餵給模型幾張圖。32 張圖算一組，跑完這組才更新一次權重
BATCH_SIZE = 32

# 將多個處理動作串聯成一個 Pipeline
transform = transforms.Compose([
    transforms.Resize((128, 128)), # 強制將所有圖片縮放到 128x128，確保輸入維度一致
    transforms.ToTensor(),         # 將 0-255 的像素值縮放到 0-1 區間，並轉為 PyTorch 張量
])

# 實例化模型，並將它搬到指定硬體 (GPU 或 CPU) 的記憶體中
model = FacialLandmarkCNN().to(device)

# 定義損失函數：使用均方誤差 (MSE)，計算預測座標與真實座標的距離平方
criterion = nn.MSELoss()

# 定義優化器：使用 Adam 算法。
# 它會拿到 Loss 給出的回饋，並自動計算「梯度」，去微調 model.parameters() 裡的權重
optimizer = optim.Adam(model.parameters(), lr=LR)