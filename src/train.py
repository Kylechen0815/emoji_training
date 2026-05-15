import torch                    # PyTorch 的核心庫，處理張量 (Tensor) 運算
import torch.nn as nn             # 包含神經網路層（如 Conv2d, Linear）與損失函數
import torch.optim as optim       # 包含優化算法（如 Adam, SGD），負責更新參數
from model import FacialLandmarkCNN # 載入你剛寫好的「大腦」模型架構
from dataset import CelebADataset   # 載入你剛寫好的「數據讀取」邏輯
from tqdm import tqdm          # 進度條工具，讓你在訓練時看到進度和損失變化
from torchvision import transforms  # 官方提供的影像處理工具（如縮放、轉張量）
from torch.utils.data import DataLoader # 負責將數據打包成 Batch 的管理員
import os                     # 處理檔案路徑和檔案操作


# 1. 初始化與超參數
CHECKPOINT_PATH = "checkpoint.pth" # 存檔點檔名
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 1         # 總共要讀完幾次數據集
BATCH_SIZE = 32     # 一次處理多少張圖
LR = 0.001          # 學習率

# 2. 資料準備
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

dataset = CelebADataset(
    csv_file='data/annotations/list_landmarks_align_celeba.csv',
    img_dir='data/raw/img_align_celeba',
    transform=transform
)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)




# 3. 初始化模型、考官(Loss)與教練(Optimizer)
model = FacialLandmarkCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# 3'. 【核心邏輯】檢查是否有舊存檔，有則載入
start_epoch = 0
if os.path.exists(CHECKPOINT_PATH):
    print(f"發現存檔點 {CHECKPOINT_PATH}，正在恢復進度...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])   # 恢復大腦記憶
    optimizer.load_state_dict(checkpoint['optimizer_state_dict']) # 恢復訓練手感
    start_epoch = checkpoint['epoch'] + 1                 # 從下一輪開始
    print(f"成功從 Epoch {start_epoch} 恢復！")


# 4. 正式開始訓練迴圈
print(f"正在 {device} 上啟動訓練...")

for epoch in range(start_epoch, EPOCHS):
    model.train()  # 切換為訓練模式
    running_loss = 0.0
    
    # 使用 tqdm 顯示進度條
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for images, landmarks in pbar:
        # 將數據搬移到指定的硬體 (GPU/CPU)
        images = images.to(device)
        landmarks = landmarks.to(device)
        
        # --- 訓練五部曲 ---
        # 1. 清空舊的梯度 (就像歸零計數器)
        optimizer.zero_grad()
        
        # 2. 前向傳播 (Forward Pass)：模型做出預測
        outputs = model(images)
        
        # 3. 計算誤差 (Loss)：對答案
        loss = criterion(outputs, landmarks)
        
        # 4. 反向傳播 (Backward Pass)：計算每個權重該修正多少
        loss.backward()
        
        # 5. 更新權重 (Optimizer Step)：正式調整大腦參數
        optimizer.step()
        
        # 累積誤差並更新進度條文字
        running_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})

    # 每跑完一個 Epoch，印出平均誤差
    print(f"Epoch {epoch+1} 完成, 平均 Loss: {running_loss/len(train_loader):.6f}")

    # --- 【核心邏輯】每輪結束後儲存當前狀態 ---
    checkpoint_data = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': running_loss / len(train_loader)
        }
    torch.save(checkpoint_data, CHECKPOINT_PATH)

# 5. 儲存訓練好的「大腦」模型檔
torch.save(model.state_dict(), "facial_landmarks.pth")
print("訓練完成，模型已儲存為 facial_landmarks.pth")