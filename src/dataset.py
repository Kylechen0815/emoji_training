import pandas as pd
import torch
from torch.utils.data import Dataset  # <--- 就是少了這一行！
from PIL import Image
import os
import numpy as np


class CelebADataset(Dataset): # 繼承自 PyTorch 官方類別，必須實作 __len__ 和 __getitem__
    def __init__(self, csv_file, img_dir, transform=None):
        # 使用 Pandas 讀取標註檔 (CSV)，這就像是你的 File Allocation Table
        self.data_frame = pd.read_csv(csv_file) 
        
        # 存放圖片的資料夾路徑
        self.img_dir = img_dir 
        
        # 預處理動作 (如 Resize, ToTensor)
        self.transform = transform

    def __len__(self):
        # 回傳 CSV 總行數，即圖片總張數
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        # 1. 取得圖片路徑：從 CSV 第一欄拿檔名，並與路徑字串拼接
        img_name = os.path.join(self.img_dir, self.data_frame.iloc[idx, 0])
        
        # 2. 開啟圖片：使用 PIL 庫打開，並強制轉為 RGB 三通道 (防止有些圖是灰階或帶 Alpha 通道)
        image = Image.open(img_name).convert('RGB')
        
        # 3. 取得標註：從 CSV 第二欄以後拿到 10 個座標數字 (x1, y1, ..., x5, y5)
        landmarks = self.data_frame.iloc[idx, 1:].values.astype('float32')
    
        # 座標正規化：將原始像素座標 (0~128) 除以 128.0
        landmarks = landmarks / 128.0

        # 4. 套用預處理：如果初始化時有傳入 transform，就對圖片進行縮放與轉張量
        if self.transform:
            image = self.transform(image)
            
        # 回傳一個 Tuple：(影像張量, 座標張量)
        return image, torch.tensor(landmarks)
    
    