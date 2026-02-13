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

1. `generator.py` が6ベッド分の ORU^R01 を1分毎に送信
2. `hl7_receiver.py` がMLLP受信し、`PV1-3` のベッドIDで最新データを集約
3. `monitor.py` が `monitor_cache.json` を読み取り表示

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
python monitor.py --cache monitor_cache.json --refresh-ms 1000
```

### 3) 仮想セントラル起動（検証時のみ）

別ターミナルで:

```powershell
python generator.py --host 127.0.0.1 --port 2575 --interval 60 --enabled true
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
- `monitor.py`: OCR前提の固定レイアウト表示

