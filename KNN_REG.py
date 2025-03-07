import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

# 檢查訓練數據是否存在
data_file = 'knn_data.npz'
if not os.path.exists(data_file):
    print(f"❌ 錯誤: 找不到 {data_file}，請先執行 knn_ocr_sample.py 來生成訓練數據")
    exit()

# 載入 KNN 訓練數據
with np.load(data_file) as data:
    train = data['train'].astype(np.float32)
    train_labels = data['train_labels'].astype(np.float32)

# 初始化 KNN 分類器
knn = cv2.ml.KNearest_create()
knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)

# 初始化變數
input_number, image_number, processed_number, image_result = [None] * 10, [None] * 10, [None] * 10, [None] * 10
test_number, result, result_str = [None] * 10, [None] * 10, [None] * 10

# 設定測試圖片名稱
input_number = [f"digit_{i}.png" for i in range(10)]

# 預測
for i in range(10):
    image_path = f"output_digits/{input_number[i]}"
    
    # 讀取測試影像
    image_number[i] = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image_number[i] is None:
        print(f"❌ 無法讀取影像: {image_path}")
        continue

    # **🔹 影像前處理**
    # 1️⃣ OTSU 二值化 + 反轉 (黑底白字)
    _, binary = cv2.threshold(image_number[i], 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2️⃣ 找數字輪廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # ✅ 選擇面積最大的輪廓
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        x, y, w, h = cv2.boundingRect(contours[0])  
        digit = binary[y:y+h, x:x+w]  # 裁剪數字區域

        # **確保影像保持比例縮放到 20x20**
        aspect_ratio = w / h
        if aspect_ratio > 1:  # 寬度較大
            new_w = 20
            new_h = int(20 / aspect_ratio)
        else:  # 高度較大
            new_h = 20
            new_w = int(20 * aspect_ratio)

        digit_resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # **填充成 20x20**
        digit_final = np.ones((20, 20), dtype=np.uint8) * 0  # **確保黑底**
        y_offset = (20 - new_h) // 2
        x_offset = (20 - new_w) // 2
        digit_final[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = digit_resized
    else:
        digit_final = binary  # 如果沒找到數字，直接使用原圖

    # **⚠️ 確保黑底白字**
    digit_final = cv2.bitwise_not(digit_final)

    # **✅ 儲存處理後的影像**
    processed_number[i] = digit_final  

    # 轉換測試影像為 400 維向量（20x20 展平成 1D）
    test_number[i] = processed_number[i].reshape(-1, 400).astype(np.float32)

    # 使用 KNN 進行預測
    ret, result[i], neighbours, dist = knn.findNearest(test_number[i], k=5)

    # 建立 64x64 白色背景結果影像
    image_result[i] = np.ones((64, 64, 3), np.uint8) * 255

    # 轉換預測結果為整數
    result_str[i] = str(int(result[i][0][0]))

    # 在影像上顯示預測結果
    text_color = (0, 255, 0) if int(result[i][0][0]) == i else (255, 0, 0)
    cv2.putText(image_result[i], result_str[i], (15, 52), cv2.FONT_HERSHEY_SIMPLEX, 2, text_color, 3)

# **🔍 顯示影像（使用 Matplotlib）**
fig, axes = plt.subplots(3, 10, figsize=(12, 6))

for i in range(10):
    if image_number[i] is None:
        continue  # 略過讀取失敗的影像
    
    # **顯示原始影像**
    axes[0, i].imshow(image_number[i], cmap='gray')
    axes[0, i].set_title(f"原始 {i}")
    axes[0, i].axis("off")

    # **顯示處理後影像**
    axes[1, i].imshow(processed_number[i], cmap='gray')  # ✅ 確保是 processed_number[i]
    axes[1, i].set_title(f"處理後 {i}")
    axes[1, i].axis("off")

    # **顯示 KNN 預測結果**
    axes[2, i].imshow(image_result[i])
    axes[2, i].set_title(f"預測 {result[i]}")
    axes[2, i].axis("off")

plt.tight_layout()
plt.show()
