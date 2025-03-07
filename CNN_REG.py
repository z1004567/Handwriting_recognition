import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model

# 檢查 CNN 模型是否存在
model_file = 'cnn_model.h5'
if not os.path.exists(model_file):
    print(f"❌ 錯誤: 找不到 {model_file}，請先訓練 CNN 模型")
    exit()

# 載入 CNN 模型
model = load_model(model_file)

# 初始化變數
input_number, image_number, image_result = [None] * 10, [None] * 10, [None] * 10
test_number, result, result_str = [None] * 10, [None] * 10, [None] * 10

# 設定測試圖片名稱（確保副檔名正確）
input_number = [f"digit_{i}.png" for i in range(10)]  # 確保測試圖片名稱正確

# 預測
for i in range(10):
    image_path = f"output_digits/{input_number[i]}"
    image_number[i] = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image_number[i] is None:
        print(f"❌ 無法讀取影像: {image_path}")
        continue

    # **🔹 影像前處理**
    _, binary = cv2.threshold(image_number[i], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # **🔹 找出數字輪廓**
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # ✅ 選擇面積最大的輪廓（確保選中數字）
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        x, y, w, h = cv2.boundingRect(contours[0])  
        digit = binary[y:y+h, x:x+w]  # 裁剪數字區域
        digit = cv2.resize(digit, (20, 20), interpolation=cv2.INTER_NEAREST)  # 重新調整大小
    else:
        digit = binary  # 如果沒找到數字，直接使用原圖

    # **✅ 儲存處理後的影像**
    image_number[i] = binary  # 🚀 確保每張影像獨立更新






    # 轉換測試影像為 CNN 需要的格式 (20, 20, 1)
    test_number[i] = binary.reshape(1, 20, 20, 1).astype(np.float32) / 255.0  # 正規化

    # 使用 CNN 進行預測
    predictions = model.predict(test_number[i])
    result[i] = np.argmax(predictions, axis=1)[0]  # 取得預測結果

    # 建立 64x64 白色背景結果影像
    image_result[i] = np.ones((64, 64, 3), np.uint8) * 255

    # 轉換預測結果為整數
    result_str[i] = str(result[i])

    # 在影像上顯示預測結果
    text_color = (0, 255, 0) if result[i] == i else (255, 0, 0)
    cv2.putText(image_result[i], result_str[i], (15, 52), cv2.FONT_HERSHEY_SIMPLEX, 2, text_color, 3)

# **🔍 顯示影像（使用 Matplotlib）**
fig, axes = plt.subplots(3, 10, figsize=(12, 6))

for i in range(10):
    if image_number[i] is None:
        continue  # 略過讀取失敗的影像
    
    # **顯示原始影像**
    axes[0, i].imshow(cv2.imread(f"output_digits/{input_number[i]}", cv2.IMREAD_GRAYSCALE), cmap='gray')
    axes[0, i].set_title(f"原始 {i}")
    axes[0, i].axis("off")

    # **顯示處理後影像**
    axes[1, i].imshow(image_number[i], cmap='gray')  # ✅ 確保是 image_number[i]
    axes[1, i].set_title(f"處理後 {i}")
    axes[1, i].axis("off")

    # **顯示 CNN 預測結果**
    axes[2, i].imshow(image_result[i])
    axes[2, i].set_title(f"預測 {result[i]}")
    axes[2, i].axis("off")

plt.tight_layout()
plt.show()
