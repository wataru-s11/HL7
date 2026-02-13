# 生体情報モニター画像認識システム

Raspberry Pi 5 + PCIe AI Kit + HDMI/CSIキャプチャを使用した生体情報モニター自動読み取りシステム

## システム概要

本システムは、生体情報モニター(患者モニター)の画面をHDMIキャプチャまたはカメラで撮影し、OCR技術により各バイタルサイン(心拍数、血圧、SpO2など)を自動的に数値化します。

### 主な機能

- **リアルタイムキャプチャ**: HDMI出力またはカメラ経由での画面取得
- **OCR認識**: PaddleOCR/Tesseractによる数値認識
- **ROI自動検出**: 画面上の各バイタル表示位置を自動判定
- **データ検証**: 正常範囲チェックと異常値フィルタリング
- **CSV出力**: 時系列データの記録
- **Hailo AI加速**: PCIe AI Kitによる高速推論(オプション)

## ハードウェア要件

### 必須
- Raspberry Pi 5 (4GB以上推奨)
- microSD カード (32GB以上)
- 電源アダプタ (5V/5A, USB-C)

### 画像入力 (いずれか)
- **方法A**: HQ Camera Module (IMX477) + C/CSマウントレンズ
  - モニター画面を直接撮影
  - 照明条件の影響を受けやすい
  
- **方法B**: CSI-HDMI変換キャプチャボード
  - HDMI出力を直接取得
  - 高画質・安定
  - 例: Geekworm HDMI to CSI-2 Adapter

### オプション
- PCIe AI Kit (Hailo-8L) - OCR高速化用
- アクティブ冷却ファン

## ソフトウェア要件

### OS
- Raspberry Pi OS (64-bit) Bookworm以降

### Python
- Python 3.9以上

### 依存パッケージ
```bash
# システムパッケージ
sudo apt-get install python3-opencv tesseract-ocr v4l-utils

# Pythonパッケージ
pip3 install opencv-python numpy pytesseract paddleocr
```

## セットアップ

### 1. 自動セットアップ
```bash
chmod +x setup.sh
./setup.sh
```

### 2. 手動セットアップ

#### システムパッケージ
```bash
sudo apt-get update
sudo apt-get install -y \
    python3-pip python3-opencv \
    tesseract-ocr libtesseract-dev \
    v4l-utils libv4l-dev
```

#### Pythonパッケージ
```bash
pip3 install opencv-python numpy pytesseract
pip3 install paddlepaddle paddleocr  # 推奨
```

#### Hailo SDK (オプション)
```bash
# https://hailo.ai/developer-zone/ からSDKをダウンロード
pip3 install hailort-*.whl
```

## 使用方法

### Step 1: デバイス確認
```bash
# カメラ/キャプチャデバイスを確認
v4l2-ctl --list-devices
ls /dev/video*
```

通常、`/dev/video0`または`/dev/video1`が対象デバイスです。

### Step 2: テスト撮影
```bash
# 静止画キャプチャ
ffmpeg -f v4l2 -i /dev/video0 -frames 1 test_capture.png

# または
python3 -c "import cv2; \
cap = cv2.VideoCapture(0); \
ret, frame = cap.read(); \
cv2.imwrite('test_capture.png', frame); \
cap.release()"
```

### Step 3: ROI設定

モニター画面の各バイタル表示位置を設定します。

```bash
# 対話的キャリブレーション
python3 roi_calibrator.py test_capture.png --output roi_config_custom.json
```

操作方法:
- マウスドラッグでROI領域を選択
- SPACEキーで現在の項目をスキップ
- 'r'キーでリセット
- 's'キーで保存して終了

### Step 4: 動作確認
```bash
# デバッグモードで実行 (ROI表示付き)
python3 vital_monitor_capture.py \
    --device 0 \
    --config roi_config_custom.json \
    --debug
```

### Step 5: 本番運用
```bash
# CSV出力付きで実行
python3 vital_monitor_capture.py \
    --device 0 \
    --config roi_config_custom.json \
    --output vitals_data.csv
```

## ROI設定ファイル

`roi_config.json`の例:
```json
{
  "HR": {
    "roi": [100, 100, 200, 80],
    "unit": "bpm",
    "range": [30, 200]
  },
  "SpO2": {
    "roi": [100, 200, 150, 80],
    "unit": "%",
    "range": [70, 100]
  }
}
```

パラメータ:
- `roi`: [x, y, width, height] - ROI座標
- `unit`: 単位
- `range`: [最小値, 最大値] - 有効範囲

## Hailo AI Kit使用

### モデル準備

OCRモデル(ONNXフォーマット)をHEF形式に変換:

```bash
# Hailo Dataflow Compilerで変換
hailo compile \
  --hw-arch hailo8l \
  --input-model ocr_model.onnx \
  --output-hef ocr_model.hef
```

### 実行
```bash
python3 hailo_accelerated.py --benchmark
```

## トラブルシューティング

### カメラが認識されない
```bash
# デバイス確認
ls -la /dev/video*
v4l2-ctl --list-devices

# 権限確認
groups  # videoグループに所属しているか確認
sudo usermod -a -G video $USER  # 追加
```

### OCR精度が低い
1. **画像品質を確認**
   - 解像度: 1080p以上推奨
   - フォーカス: ピントが合っているか
   - 照明: 均一な明るさ

2. **前処理パラメータ調整**
   - `VitalMonitorOCR.preprocess_roi()`のCLAHEパラメータ
   - 二値化閾値の調整

3. **ROI位置の再設定**
   - 数字のみが含まれるように領域を調整
   - 余白を最小化

### Hailo SDKエラー
```bash
# ファームウェア確認
hailortcli fw-control identify

# デバイス確認
lspci | grep Hailo

# ドライバ再読み込み
sudo modprobe -r hailo_pci
sudo modprobe hailo_pci
```

## システム構成

```
vital_monitor_capture.py    # メインプログラム
├── HDMICapture             # 画像キャプチャ
├── VitalMonitorOCR         # OCRエンジン
└── VitalMonitorAnalyzer    # 解析・判定

roi_calibrator.py           # ROI設定ツール
hailo_accelerated.py        # Hailo加速版
roi_config.json             # 設定ファイル
```

## パフォーマンス

### CPU版 (Raspberry Pi 5)
- 処理速度: 約1-2 FPS
- CPU使用率: 60-80%
- メモリ: 約500MB

### Hailo加速版
- 処理速度: 約10-15 FPS
- CPU使用率: 20-30%
- 推論レイテンシ: <10ms

## 臨床使用における注意事項

**本システムは研究・開発用途のプロトタイプです。**

医療機器としての認証は取得しておらず、臨床使用には以下の対応が必要です:

1. **医療機器認証**: PMDA等の認証取得
2. **データ検証**: 実測値との一致率検証
3. **冗長化**: バックアップシステムの構築
4. **アラート連携**: 異常値検出時の通知機能
5. **ログ管理**: 改ざん防止とトレーサビリティ

## 応用例

- ICU/手術室での自動記録
- テレメディシン用データ送信
- 研究データ収集
- モニター異常検知

## ライセンス

MIT License

## 開発者向け情報

### コード拡張

カスタムOCRエンジンの追加:
```python
class CustomOCR(VitalMonitorOCR):
    def recognize_text(self, roi_image):
        # カスタム実装
        return text, confidence
```

データベース連携:
```python
import sqlite3
# VitalSign保存処理を追加
```

### デバッグ

```python
# デバッグログ有効化
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 参考資料

- [Raspberry Pi Documentation](https://www.raspberrypi.com/documentation/)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [Hailo Developer Zone](https://hailo.ai/developer-zone/)
- [OpenCV Documentation](https://docs.opencv.org/)

## 更新履歴

- v1.0.0 (2025-02-02): 初版リリース
  - 基本機能実装
  - Hailo対応
  - ROIキャリブレーションツール

## お問い合わせ

Issue報告やプルリクエストを歓迎します。
