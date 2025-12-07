# 使用者手冊

**版本**: 2.0
**更新日期**: 2025-12-07


---

## 目錄

1. [系統概述](#系統概述)
2. [安裝與快速開始](#安裝與快速開始)
3. [📝 標註頁籤](#標註頁籤-annotation-tab)
4. [🎯 訓練頁籤](#訓練頁籤-training-tab)
5. [🔍 檢測頁籤](#檢測頁籤-detection-tab)
6. [常見問題](#常見問題)
7. [進階功能](#進階功能)

---

## 系統概述

此專案是一套整合式的物體標註、訓練與檢測系統，專為旋轉邊界框 (Oriented Bounding Box, OBB) 標註設計。

### 主要功能

- **智慧標註**: 使用 SAM (Segment Anything Model) 自動生成精準遮罩
- **旋轉邊界框**: PCA 演算法自動計算 OBB 參數
- **批次處理**: 一鍵處理大量圖片
- **YOLO OBB 訓練**: 整合 YOLOv10n-obb 模型訓練
- **資料強化**: 可調整的訓練資料強化參數
- **6D 姿態估計**: 支援 Simple 與 Full 模式的物體姿態檢測
- **RealSense 整合**: 支援 Intel RealSense D415 深度相機
- **模型管理**: 靈活的模型切換與管理

### 系統架構

```
tt_yolov10_grasp/
├── src/
│   ├── ui/              # UI 元件 (標註、訓練、檢測頁籤)
│   ├── services/        # 核心服務 (SAM, YOLO, RealSense)
│   ├── workers/         # 背景工作執行緒
│   ├── core/            # 核心邏輯 (Canvas, Config)
│   └── utils/           # 工具函式
├── runs/                # 訓練結果輸出
├── models/              # 模型檔案
└── data/                # 資料集
```

---

## 安裝與快速開始

### 系統需求

請在安裝與執行前確認系統符合最低需求：

- **作業系統**: Windows 10/11 或 Ubuntu 20.04+（本手冊以 Windows PowerShell 範例為主）
- **Python**: 3.10.x（建議使用 conda 管理環境）
- **GPU（可選）**: NVIDIA GPU 支援 CUDA 12.8 時，可使用 `torch==2.8.0+cu128`；若無 GPU，請使用 CPU 版本的 PyTorch
- **NVIDIA 驅動**: 與 CUDA 12.8 相容的驅動（請參考 NVIDIA 官方說明）
- **RealSense（可選）**: Intel RealSense D400 系列，建議安裝 RealSense SDK 與 `pyrealsense2==2.56.x`
- **其他工具**: `git`（用於安裝某些 pip 套件）、Visual C++ Build Tools（Windows，如需編譯原始碼）

### 安裝步驟

#### 使用 Conda（含 GPU 支援）

1. 在專案根目錄開啟 PowerShell，建立並啟用環境：

```powershell
conda env create -f environment.yaml -n tt_yolov10_grasp
conda activate tt_yolov10_grasp
```

2. 若 `environment.yaml` 中仍有需要用 `pip` 安裝的套件，執行：

```powershell
pip install -r requirements.txt
```

3. 若要使用 GPU（CUDA 12.8 / PyTorch +cu128），可手動安裝對應的 PyTorch wheel：

```powershell
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0+cu128 torchvision==0.23.0+cu128
```



### 啟動應用程式

```powershell
# 使用 GPU
python app.py --image-dir "C:\path\to\images" --output-dir "C:\path\to\out" --checkpoint "C:\path\to\sam_vit_b_01ec64.pth" --device cuda

# 使用 CPU
python app.py --image-dir "C:\path\to\images" --output-dir "C:\path\to\out" --device cpu
```

常見 CLI 參數：`--image-dir`, `--output-dir`, `--checkpoint`, `--device` (cpu|cuda), `--iou-threshold`

### 快速驗證（10 張圖片測試）

1. 準備測試資料：

```powershell
mkdir C:\data\demo_images
# 把 10 張 JPG/PNG 放到 C:\data\demo_images
```

2. 啟動應用（建議先用 CPU 測試）：

```powershell
python app.py --image-dir "C:\data\demo_images" --output-dir "C:\data\demo_out" --device cpu
```

3. **標註步驟**：
   - 切換到 `📝 標註` 頁籤
   - 點選 `選擇資料夾` 載入圖片
   - 點 `🔍 執行 SAM` 生成遮罩
   - 選取遮罩後點 `匯出所有標註`

4. **訓練步驟**：
   - 切到 `🎯 訓練` 頁籤
   - 選擇標註輸出資料夾
   - 點 `配置訓練/驗證集分割`（80/20）
   - 設定 `Epochs=5, Batch Size=4`
   - 點 `▶️ 開始訓練`

5. **檢測步驟**：
   - 訓練完成後切到 `🔍 檢測` 頁籤
   - 選擇單張圖片或資料夾
   - 點 `🔍 開始檢測`

---

## 📝 標註頁籤 (Annotation Tab)

### 功能概述

使用 SAM 模型進行智慧標註，自動生成 OBB 格式的標註檔案。

### UI 區塊說明

#### 圖片管理區
- **選擇資料夾**: 載入待標註圖片資料夾（支援 `.png`, `.jpg`, `.jpeg`, `.bmp`）
- **圖片導航**: 查看當前圖片數量，上/下一張圖片

#### ROI 設定區
- **設定 ROI**: 在圖片上拖曳繪製矩形區域，限制 SAM 標註範圍以提升處理速度
- **確認 ROI**: 儲存 ROI 設定

#### SAM 控制區
- **🔍 執行 SAM**: 對當前圖片或 ROI 區域運行 SAM 生成遮罩
- **Ctrl 多選**: 在結果中按 Ctrl 選擇多個遮罩
- **儲存當前圖片**: 保存當前圖片的標註結果
- **匯出所有標註**: 批次輸出所有圖片的標註檔案

### 操作流程

#### 單張標註

1. 點擊 `選擇資料夾` 載入圖片
2. 點擊 `🔍 執行 SAM` 生成遮罩
3. SAM 會自動偵測候選遮罩，使用 Ctrl 多選要匯出的遮罩
4. 點 `儲存當前圖片` 保存標註結果

**OBB 格式** (YOLO OBB):
```
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

#### 批次標註

1. 確認已選擇圖片資料夾
2. 可選：點 `設定 ROI` 限制處理範圍
3. 點 `開始批次標註` 自動處理所有圖片
4. 進度對話框會顯示處理狀態

#### 輸出格式

```
output_dir/
├── labels/
│   ├── image1.txt
│   ├── image2.txt
│   └── ...
└── images/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

---

## 🎯 訓練頁籤 (Training Tab)

### 功能概述

使用 YOLOv10n-obb 模型訓練 OBB 檢測器。

### UI 區塊說明

#### 資料集設定區
- **瀏覽**: 選擇標註輸出目錄（包含 `labels/` 和 `images/` 的資料夾）
- **配置訓練/驗證集分割**: 調整分割比例（預設 80/20）並生成 `dataset.yaml`

#### 訓練參數區
- **訓練輪數 (Epochs)**: 建議 100-300（測試時可設 5）
- **Batch Size**: 根據 GPU 記憶體調整（建議 8-16，測試時可設 4）

#### 資料強化區
- **⚙️ 資料強化設定**: 開啟對話框進行資料強化設定
  - 提供 3 種預設：**輕度**、**中度**（推薦）、**重度**
  - 支援自訂參數並保存預設組合

#### 訓練控制區
- **▶️ 開始訓練**: 啟動訓練流程
- **訓練日誌**: 實時顯示訓練進度和指標
- **訓練曲線**: 即時繪製 Loss、mAP 等指標曲線

### 操作流程

#### 1. 準備資料集

1. 點擊 `瀏覽` 選擇標註輸出資料夾
2. 點擊 `配置訓練/驗證集分割`
3. 調整分割比例（預設 80/20）
4. 點擊 `確定` 生成 `dataset.yaml`

#### 2. 設定訓練參數

1. 輸入 **訓練輪數**（Epochs）：建議 100-300
2. 設定 **Batch Size**：根據 GPU 記憶體調整（建議 8-16）

#### 3. 設定資料強化（可選）

1. 點擊 `⚙️ 資料強化設定` 開啟對話框
2. 選擇預設組合（輕度/中度/重度）或自訂參數
3. 點擊 `確定` 套用設定

資料強化可以幫助模型學習更多變化，提升泛化能力：
- **輕度強化**: 適用於資料充足、物體變化小的情況
- **中度強化**: 一般應用推薦
- **重度強化**: 適用於資料不足、物體變化大的情況

#### 4. 開始訓練

1. 點擊 `▶️ 開始訓練`
2. 訓練日誌顯示於右側面板
3. 訓練曲線即時更新
4. 訓練完成後自動評估模型

#### 訓練結果

訓練完成後，結果儲存於：

```
runs/obb/train/
├── weights/
│   ├── best.pt      # 最佳模型
│   └── last.pt      # 最後一輪模型
├── results.png      # 訓練曲線圖
└── confusion_matrix.png
```

---

## 🔍 檢測頁籤 (Detection Tab)

### 功能概述

使用訓練好的模型進行物體檢測與 6D 姿態估計。支援圖片、資料夾批次檢測和 RealSense 實時檢測。

### UI 區塊說明

#### 輸入來源選擇區
- **圖片模式**: 點 `瀏覽` 選擇單張圖片進行檢測
- **資料夾模式**: 點 `瀏覽` 選擇資料夾進行批次檢測
- **相機模式**: 點 `啟動相機` 使用 RealSense 進行實時檢測

#### 姿態模式選擇區
- **Simple Mode**: 僅估計 Z 軸旋轉（2D 平面旋轉），速度快
- **Full Mode**: 完整 6D 姿態估計（位置 + 旋轉），精度更高

#### 模型設定區 🆕
- **模型路徑設定**: 點擊按鈕選擇不同的 YOLO OBB 模型檔案（`.pt` 或 `.pth`）
- **模型狀態**: 顯示當前載入的模型名稱和狀態
- **自動保存**: 模型路徑自動保存，下次啟動時自動載入

#### 相機內參設定區
- **相機內參設定**: 設定 Image Mode 使用的相機內部參數
  - 支援從 RealSense 自動載入
  - 支援自訂值並持久化

#### 視覺化選項區
- **顯示 3D 視覺化**: 勾選後開啟 Open3D 視窗查看點雲和姿態

#### 儲存選項區
- **自動儲存 JSON**: 勾選後檢測完自動保存結果至輸出目錄（用於機器手臂）
- **手動儲存 JSON**: 點擊按鈕選擇位置手動保存檢測結果
- **儲存 TXT**: 以文字格式保存檢測結果
- **儲存圖片**: 保存標註後的檢測結果圖片

### 操作流程

#### 圖片/資料夾模式檢測

1. 點擊 `瀏覽` 選擇圖片或資料夾
2. 選擇 **姿態模式**（Simple/Full）
3. （可選）點 `模型路徑設定` 切換不同模型
4. （可選）點 `相機內參設定` 調整相機參數
5. 點 `🔍 開始檢測` 執行檢測
6. 檢測完成後：
   - 結果自動儲存（如已勾選自動儲存）
   - 或點按鈕手動儲存 JSON/TXT/圖片

#### 相機模式檢測

1. 確保 RealSense 相機已連接
2. 點 `啟動相機` 開啟實時檢測
3. 相機會自動讀取內參並進行檢測
4. 結果實時顯示於畫面
5. 點 `停止相機` 結束檢測

#### 結果說明

檢測結果包含：
- **transform_matrix**: 4x4 變換矩陣（位置和旋轉）
- **position**: 物體在相機坐標系中的 3D 位置（公尺）
- **rotation_euler**: 歐拉角表示的旋轉（弧度）
- **obb**: 旋轉邊界框信息

---



