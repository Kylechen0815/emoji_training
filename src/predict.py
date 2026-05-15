import torch
import cv2
import numpy as np
from model import FacialLandmarkCNN
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# 1. 設定環境與載入模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FacialLandmarkCNN().to(device)
model.load_state_dict(torch.load("facial_landmarks.pth", map_location=device))
model.eval() # 切換為評估模式

# 2. 影像預處理 (必須跟訓練時一模一樣)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

def predict_landmarks(image_path):
    # 讀取並轉換圖片
    img_raw = Image.open(image_path).convert('RGB')
    img_tensor = transform(img_raw).unsqueeze(0).to(device) # 增加 Batch 維度 [1, 3, 128, 128]

    with torch.no_grad(): # 預測時不需要計算梯度，省記憶體
        output = model(img_tensor)
        landmarks = output.cpu().numpy().flatten()

    # 將正規化後的座標 (0~1) 轉回像素座標 (0~128)
    landmarks = landmarks * 128
    
    # 畫圖呈現
    img_plot = np.array(img_raw.resize((128, 128)))
    plt.imshow(img_plot)
    plt.scatter(landmarks[0::2], landmarks[1::2], c='red', marker='x', s=20)
    plt.title("AI Prediction")
    plt.show()

# 3. 隨便從資料夾挑一張圖來試試看
test_image = "data/raw/img_align_celeba/000003.jpg" # 換成你硬碟裡有的檔名
predict_landmarks(test_image)   