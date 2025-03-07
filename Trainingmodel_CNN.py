import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # ✅【1】加入數據增強工具

# 設定影像路徑
image_path = "C:\\Users\\Chih-Hang.LIU\\Downloads\\digits1.png"

# 檢查影像是否存在
if not os.path.exists(image_path):
    print(f"❌ 錯誤: 找不到影像 '{image_path}'，請確認檔案是否存在")
    exit()

# 讀取影像（確保影像存在，避免 None 錯誤）
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    print("❌ 無法讀取影像，請確認影像格式是否正確")
    exit()

# 將影像切割成 50 行 × 100 列的數字（每個數字 20x20）
cells = [np.hsplit(row, 100) for row in np.vsplit(img, 50)]
x = np.array(cells)  # 轉換為 NumPy 陣列

# 分割訓練集和測試集
train = x[:, :50].reshape(-1, 20, 20, 1).astype(np.float32) / 255.0  # 正規化
test = x[:, 50:100].reshape(-1, 20, 20, 1).astype(np.float32) / 255.0  # 正規化

# 建立標籤（每個數字 0-9 各 250 張）
k = np.arange(10)  # 數字 0-9
train_labels = np.repeat(k, 250)  # 變成 2500 個標籤
test_labels = train_labels.copy()  # 測試標籤與訓練標籤相同

# 轉換為 One-hot Encoding 格式
train_labels_cnn = to_categorical(train_labels, 10)
test_labels_cnn = to_categorical(test_labels, 10)

# ✅【2】建立數據增強器
datagen = ImageDataGenerator(
    rotation_range=10,     # 隨機旋轉 ±10 度
    width_shift_range=0.1, # 水平平移 ±10%
    height_shift_range=0.1,# 垂直平移 ±10%
    shear_range=0.1,       # 斜切變形
    zoom_range=0.1         # 縮放 ±10%
)
datagen.fit(train)  # ✅【3】讓增強器學習訓練數據的分佈

# **CNN 模型結構**
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(20, 20, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')  # 10 類輸出
])

# 編譯模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ✅【4】使用增強數據進行訓練
model.fit(datagen.flow(train, train_labels_cnn, batch_size=32),
          epochs=30, 
          validation_data=(test, test_labels_cnn),
          steps_per_epoch=len(train) // 32)

# 測試準確度
test_loss, test_acc = model.evaluate(test, test_labels_cnn, verbose=0)
print(f"✅ CNN 模型測試準確度: {test_acc * 100:.2f}%")

# **儲存 CNN 模型**
model.save("cnn_model.h5")
print("✅ CNN 模型已成功儲存為 'cnn_model.h5'")

# **測試載入模型**
model = load_model("cnn_model.h5")
print("✅ 成功載入 CNN ！")
