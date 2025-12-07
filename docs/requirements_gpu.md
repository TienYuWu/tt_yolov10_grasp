# GPU 安裝說明（CUDA 12.8 / PyTorch +cu128）

此說明針對想在 Windows / Linux 上啟用 GPU 的使用者，重點列出安裝步驟與常見錯誤處理。

## 前提

- 有支援 CUDA 的 NVIDIA GPU
- 安裝相容的 NVIDIA driver（請參考 NVIDIA 官方驅動說明）
- 使用者具有管理員權限以安裝驅動/套件

## 建議 CUDA / 驅動版本

- 本專案推薦使用 CUDA 12.8 對應 `torch==2.8.0+cu128`。
- 若使用不同 CUDA 版本，請前往 PyTorch 官網選擇相對應的 wheel 安裝指令。

## Windows 安裝步驟（簡要）

1. 從 NVIDIA 官網下載並安裝相容的 NVIDIA Driver（選擇對應到 CUDA 12.8 的驅動）。
2. 重新啟動系統。
3. 建議使用 conda 建環境後再用 pip 安裝 PyTorch：

```powershell
conda create -n smart_label_tt python=3.10 -y
conda activate smart_label_tt
# 安裝其他 conda 套件 (可選): conda install -c conda-forge opencv numpy -y
# 安裝 PyTorch + cu128 wheel
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0+cu128 torchvision==0.23.0+cu128
```

1. 安裝 `requirements.txt` 中其他套件：

```powershell
pip install -r requirements.txt
```

## 驗證 GPU 是否可用

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

若回傳 `False`，請檢查：

- NVIDIA driver 是否安裝正確
- CUDA 與 driver 是否相容
- PATH/環境變數未指向錯誤的 CUDA 目錄

## 常見錯誤與處理

- pip 安裝 wheel 時出現二進位相依錯誤：先安裝 `visualstudio build tools` 或使用 conda 的二進位套件
- conda env create 遇到編碼/plugin 錯誤（Windows cp950）：可將 pip 部分抽出為 `requirements.txt`，先建立 conda env 再執行 `pip install -r requirements.txt`
- segment-anything git+ 安裝失敗：請先安裝 `git` 並確認能夠從 GitHub clone

## 備註

- 若無 GPU，可跳過上述 wheel 直接使用 `pip install -r requirements.txt`（但某些功能/效能會降低）
- 若在公司網路或代理下安裝失敗，請使用可訪問的 pip index 或將 wheel 下載到本機再安裝

---

此檔為草稿，可根據你實際的 `smart_label.yaml` 與 `requirements.txt` 調整與精緻化。
