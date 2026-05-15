import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_training_results(image, landmarks, pred_landmarks=None):
    """
    視覺化圖片與點位，檢查模型預測得準不準
    """
    img = image.permute(1, 2, 0).cpu().numpy() # Tensor 轉為 numpy 格式
    
    plt.imshow(img)
    
    # 真實點位 (綠色)
    plt.scatter(landmarks[0::2] * 128, landmarks[1::2] * 128, s=10, c='g', label='True')
    
    # 預測點位 (紅色)
    if pred_landmarks is not None:
        plt.scatter(pred_landmarks[0::2] * 128, pred_landmarks[1::2] * 128, s=10, c='r', label='Pred')
    
    plt.legend()
    plt.show()

def save_model(model, path="facial_landmarks.pth"):
    """儲存模型權重"""
    torch.save(model.state_dict(), path)
    print(f"模型已儲存至 {path}")