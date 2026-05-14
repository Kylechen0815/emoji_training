import pandas as pd
import cv2
import os
import numpy as np

# 這裡使用相對於「根目錄」的路徑
CSV_PATH = 'data/annotations/list_landmarks_align_celeba.csv'
IMG_DIR = 'data/raw/img_align_celeba'

if not os.path.exists(CSV_PATH):
    print(f"錯誤：找不到 CSV！請檢查執行路徑。目前路徑為: {os.getcwd()}")
else:
    df = pd.read_csv(CSV_PATH)
    row = df.iloc[0]
    img_name = row[0]
    # 這裡要注意欄位索引，CelebA CSV 第一欄通常是檔名，後續是座標
    landmarks = row[1:].values.astype('int')

    img_path = os.path.join(IMG_DIR, img_name)
    image = cv2.imread(img_path)

    if image is not None:
        for i in range(0, len(landmarks), 2):
            x, y = landmarks[i], landmarks[i+1]
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        
        # 存成圖片，不要用 imshow
        cv2.imwrite('check_landmarks.jpg', image)
        print("成功！請查看根目錄下的 check_landmarks.jpg")
    else:
        print(f"找不到圖片: {img_path}")