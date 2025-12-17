# 快速參考

快速參考頁（便於技術支援與快速驗收）

## 專案資訊
- 專案名稱: TT_Yolov10_PileGrasp
- 環境名稱: tt_yolov10_pilegrasp
- 設定目錄: .tt_yolov10_pilegrasp/

## 常用 CLI 參數
- `--image-dir <path>`: 指定輸入圖片資料夾
- `--output-dir <path>`: 指定匯出/結果資料夾
- `--checkpoint <path>`: SAM 權重檔 (.pth)
- `--device <cpu|cuda>`: 選擇執行裝置
- `--iou-threshold <float>`: NMS IoU 閾值

**範例（Windows PowerShell）**:

```powershell
python app.py --image-dir "C:\data\demo_images" --output-dir "C:\data\demo_out" --device cpu
python app.py --image-dir "C:\data\images" --output-dir "C:\data\out" --checkpoint "C:\models\sam_vit_b_01ec64.pth" --device cuda
```

## 主要 UI 按鈕對照

| 功能 | UI 按鈕文字 |
|------|-----------|
| 選擇圖片 | `瀏覽` |
| 執行 SAM | `🔍 執行 SAM` |
| 儲存標註（單張） | `儲存當前圖片` |
| 匯出所有標註 | `匯出所有標註` |
| 開始批次標註 | `開始批次標註` |
| 開始訓練 | `▶️ 開始訓練` |
| 模型設定 | `模型路徑設定` |
| 修改內參 | `修改內參` |
| 開始檢測 | `🚀 開始檢測` |
| 啟動相機 | `啟動相機` |

## 重要路徑
- 專案根目錄: tt_yolov10_grasp/
- 訓練輸出: runs/obb/train/
- 檢測結果: {output_dir}/detections/json/, {output_dir}/detections/txt/
- SAM 權重建議放置: models/sam_vit_b_01ec64.pth

## 快速檢查命令

```powershell
# Python 版本
python --version

# PyTorch + CUDA
python -c "import torch; print(torch.__version__, 'cuda_available=', torch.cuda.is_available())"

# SAM 權重檔
Test-Path "C:\models\sam_vit_b_01ec64.pth"

# RealSense
python -c "import pyrealsense2 as rs; print('pyrealsense2', rs.__version__)"
```

## 問題回報要點
- 作業系統與 Python 版本
- 是否使用 GPU（CUDA 版本與 NVIDIA 驅動）
- 問題截圖與完整終端錯誤（stderr）
- requirements.txt（或 environment.yaml, 若使用 conda）
