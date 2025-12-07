# Quick Reference — Smart Label TT

快速參考頁（便於技術支援與快速驗收）

- ## 常用 CLI 參數

- `--image-dir <path>`: 指定輸入圖片資料夾
- `--output-dir <path>`: 指定匯出/結果資料夾
- `--checkpoint <path>`: SAM 權重檔 (.pth)
- `--device <cpu|cuda>`: 選擇執行裝置
- `--iou-threshold <float>`: NMS IoU 閾值

範例（Windows PowerShell）:

```powershell
python app.py --image-dir "C:\data\demo_images" --output-dir "C:\data\demo_out" --device cpu
python app.py --image-dir "C:\data\images" --output-dir "C:\data\out" --checkpoint "C:\models\sam_vit_b_01ec64.pth" --device cuda
```

- ## 主要 UI 按鈕（文件 -> UI）

- 選擇圖片資料夾: `選擇資料夾`
- 設定 ROI: `設定 ROI`
- 執行 SAM: `🔍 執行 SAM`
- 儲存標註（單張）: `儲存當前圖片`
- 匯出所有標註: `匯出所有標註`
- 開始批次標註: `開始批次標註`
- Training: `▶️ 開始訓練` / `⏹️ 停止訓練`
- Detection: `瀏覽`（選檔） / `啟動相機` / `🎯 執行檢測`

- ## 重要路徑

- 專案根目錄: `Smart_Label_TT/`
- 模型預設（訓練 tab）: `src/yolov10n-obb.pt`
- SAM 權重建議放置: `models/`（例如 `models/sam_vit_b_01ec64.pth`）
- 訓練輸出: `runs/obb/train/`
- 快速測試資料: `C:\data\demo_images`（示例）

- ## 快速檢查命令（PowerShell / Python）

- 檢查 Python 版本:

```powershell
python --version
```

- 檢查 PyTorch 與 CUDA 可用性:

```powershell
python -c "import torch; print(torch.__version__, 'cuda_available=', torch.cuda.is_available())"
```

- 檢查是否有 SAM 權重檔:

```powershell
Test-Path "C:\models\sam_vit_b_01ec64.pth"
```

- 檢查 RealSense (pyrealsense2):

```powershell
python -c "import pyrealsense2 as rs; print('pyrealsense2', rs.__version__)"
```

- ## 問題回報要點

- 提供以下資訊能加速除錯：
- 作業系統與 Python 版本
- 是否使用 GPU（CUDA 版本與 NVIDIA 驅動）
- 問題截圖與完整終端錯誤（stderr）
- `requirements.txt` 或 `smart_label.yaml`（如使用 conda）

---

（此文件為草稿，可依需求擴充為單張 PDF 或放在 README 的快速參考區）
