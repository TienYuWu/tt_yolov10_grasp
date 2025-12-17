# TT_Yolov10_PileGrasp

整合式物體標註、訓練與 6D 姿態估計系統。

- 使用者手冊：[USER_GUIDE.md](./USER_GUIDE.md)
- 快速參考：[QUICK_REFERENCE.md](./QUICK_REFERENCE.md)

**核心功能**
- 智慧標註：SAM 產生遮罩，支援 OBB 標註
- 模型訓練：YOLOv10n-obb，含資料強化與訓練監控
- 姿態檢測：Simple 與 Full 模式，支援圖片與 RealSense
- 機器手臂整合：輸出標準 JSON（4x4 變換矩陣 + 歐拉角）

**快速開始（Windows/Conda）**

```powershell
conda create -n tt_yolov10_pilegrasp python=3.10
conda activate tt_yolov10_pilegrasp
pip install -r requirements.txt
pip install ultralytics
# GPU（可選，CUDA 12.8）
pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 --index-url https://download.pytorch.org/whl/cu128
# RealSense（可選）
pip install pyrealsense2
```

**啟動範例**

```powershell
python app.py --image-dir "C:\data\images" --output-dir "C:\data\out" --device cpu
```

**典型流程**
- 標註：載入圖片 → 執行 SAM → 匯出 OBB
- 訓練：設定資料集分割與參數 → 開始訓練
- 檢測：設定模型路徑與姿態模式 → 🚀 開始檢測 → 匯出 JSON




