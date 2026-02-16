# HL7 仮想モニタ統合システム

院内ネットワーク接続前の事前検証向けに、以下3コンポーネントで構成したシステムです。

- `generator.py`（仮想セントラル）
- `hl7_receiver.py`（TCP/MLLP受信 + ベッド単位集約）
- `monitor.py`（固定レイアウトGUI表示）

要件対応:
- HL7 v2.x `ORU^R01`
- TCP/MLLP通信
- ベッド識別は `PV1-3`
- GUI固定レイアウト（白背景・黒文字・桁固定）
- Python 3.11 / Windows想定

---

## アーキテクチャ（実機置換を意識）

`generator.py` はあくまで**テスト用の送信元**で、受信側は送信元に依存しません。
本番では generator を停止し、実機セントラルから同じ ORU^R01 を送ればそのまま動作します。

1. `generator.py` が6ベッド分の ORU^R01 を送信
2. `hl7_receiver.py` がMLLP受信し、`PV1-3` のベッドIDで最新データを集約
3. `monitor.py` が `monitor_cache.json` を定期再読込して表示

---

## モニタ表示仕様（OCR前提）

- 画面サイズは **1920x1080固定**（スクロールなし・はみ出しなし）
- 6ベッド（`BED01`〜`BED06`）を **2列×3行** で固定表示
- 1ベッドあたり **4列×5行 = 20セル** の固定配置（位置が変動しない）
- 表示項目（固定順）:
  `HR, ART_S, ART_D, ART_M, CVP_M, RAP_M, SpO2, TSKIN, TRECT, rRESP, EtCO2, RR, VTe, VTi, Ppeak, PEEP, O2conc, NO, BSR1, BSR2`
- 白背景・黒文字
- 値フォントは40px（Consolas等幅）、ラベルは16px
- `--refresh-ms` でJSON再読込周期を指定
- `--stale-sec`（互換: `--stale-seconds`）秒以上更新がないベッドは `NA` 表示（既定30秒、桁幅固定）
- JSONファイルが欠損/壊れ/更新途中でも表示は維持（直前の正常表示を継続）
- `hl7_receiver.py` のJSON出力は `monitor_cache.json.tmp` へ一時書込後に `os.replace` で原子的に差し替え
- 各ベッドに `last: HH:MM:SS`（最終更新時刻）を小さく表示

---

## セットアップ（Windows / Python 3.11）

```powershell
cd C:\path\to\HL7
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
```

> GUIは標準の `tkinter` を利用します（通常のWindows Pythonに同梱）。

---

## 実行手順

### 1) 受信サービス起動（必須）

```powershell
python hl7_receiver.py --mode service --host 0.0.0.0 --port 2575 --cache monitor_cache.json
```

### 2) GUIモニタ起動（必須）

別ターミナルで:

```powershell
python monitor.py --cache monitor_cache.json --refresh-ms 1000 --stale-sec 30
```

### 3) 仮想セントラル起動（検証時のみ）

別ターミナルで:

```powershell
python generator.py --host 127.0.0.1 --port 2575 --interval 10 --count 3
```

---

## generatorの本番無効化

実機セントラルに切替えるときは `generator.py` を起動しないか、明示的に無効化します。

```powershell
python generator.py --enabled false
```

---

## 主要ファイル

- `generator.py`: 6ベッドのランダムバイタル ORU^R01 を生成・送信
- `hl7_receiver.py`: MLLP受信、HL7パース、ベッド単位JSONキャッシュ作成
- `hl7_parser.py`: ORU^R01、`PV1-3`、OBXバイタル抽出
- `monitor.py`: OCR前提の固定レイアウト表示（6ベッド×20項目）


## 動作確認例（generator 10秒更新）

```powershell
# terminal-1
python hl7_receiver.py --mode service --host 0.0.0.0 --port 2575 --cache monitor_cache.json

# terminal-2
python monitor.py --cache monitor_cache.json --refresh-ms 1000 --stale-sec 30

# terminal-3
python generator.py --host 127.0.0.1 --port 2575 --interval 10
```

確認ポイント:
- 6ベッドが FullHD 1画面内に収まる
- 各ベッドは 4x5 の固定セルで値更新時も位置不変
- 10秒ごとに値が更新されても `NA` がチラつかない

---

## OCRキャプチャ学習データ作成アプリ

`ocr_capture_app.py` は表示中の monitor 画面を定期キャプチャして EasyOCR で数値抽出し、JSON Lines へ追記保存します。

### 追加インストール

```powershell
python -m pip install -r requirements.txt
```

### モニター選択の考え方（mss index）

- 起動時に `mss.monitors` の一覧（`index / left / top / width / height`）を必ずログ表示します。
- `--mss-monitor-index` を指定すると、その index のモニター全域をキャプチャします（`window_title` 検索は無視）。
- Windows の「ディスプレイ番号」と `mss` index が一致しない場合があるため、上記ログを見て index を選んでください。
- `--mss-monitor-index` 未指定時は従来互換（`window_title` を優先、見つからなければ `--monitor-index`）で動作します。

### 実行例

```powershell
# monitor.py を起動せず、mss monitor index=3 を10秒間隔でOCR（GPU有効）
python ocr_capture_app.py --cache monitor_cache.json --config ocr_capture_config.json --outdir dataset --interval-ms 10000 --no-launch-monitor true --mss-monitor-index 3 --gpu true

# debug ROI画像を保存（bed/vitalラベル付き）
python ocr_capture_app.py --cache monitor_cache.json --debug-roi true --save-images true

# validator 単体実行（最新50件）
python validator.py --ocr-results dataset/20260216/ocr_results.jsonl --monitor-cache monitor_cache.json --validator-config validator_config.json --last 50
```

### 主なオプション

- `--no-launch-monitor true` : `monitor.py` を起動せず、既に表示されている画面だけをキャプチャ（デフォルト）
- `--mss-monitor-index 3` : `sct.monitors[3]` を使ってモニター全域キャプチャ
- `--save-images false` : 画像保存を無効化（`ocr_results.jsonl` のみ追記）
- `--debug-roi true` : ROI枠 + `BEDxx:vital` ラベル付きデバッグ画像を保存
- `--gpu true/false` : EasyOCR のGPU利用を指定（CUDA不可時は自動CPUフォールバック）
- `--run-validator true` : キャプチャごとに `validator.py` を実行（重いので既定は false）

### 出力

- `dataset/<YYYYMMDD>/images/*.png` : フレーム画像
- `dataset/<YYYYMMDD>/ocr_results.jsonl` : 1フレーム=1JSON（append）
- `dataset/<YYYYMMDD>/validation_results.jsonl` : validator比較結果

