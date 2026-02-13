# HL7 仮想セントラルモニタ統合システム

6ベッド(BED01〜BED06)のバイタルを10秒ごとに生成し、HL7 ORU^R01 (TCP/MLLP) で送信、受信側で最新値を `latest.json` に集約し、GUIモニタで固定表示します。

## 構成
- `hl7_sender.py`: 仮想セントラルモニタ（generator + sender）
- `hl7_receiver.py`: HL7受信 + ベッド別最新値を `latest.json` 出力
- `hl7_parser.py`: HL7 ORUパーサ（PV1-3のベッドID、OBX-3コードマッピング対応）
- `monitor.py`: PySide6 GUI（白背景/黒文字、固定レイアウト）

## 20項目
`HR, ART_S, ART_D, ART_M, CVP_M, RAP_M, SpO2, TSKIN, TRECT, rRESP, EtCO2, RR, VTe, VTi, Ppeak, PEEP, O2conc, NO, BSR1, BSR2`

## セットアップ
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 起動手順（3プロセス同時）
### 1) 受信・集約開始
```bash
python3 hl7_receiver.py --host 0.0.0.0 --port 2575 --latest-file latest.json
```

### 2) 仮想セントラルモニタ送信開始
```bash
python3 hl7_sender.py --host 127.0.0.1 --port 2575 --interval 10
```

### 3) GUIモニタ起動
```bash
python3 monitor.py latest.json
```

## 補足
- ベッドIDは `PV1-3` に `ICU^BED01` 形式で格納。
- 受信JSONの形式:
```json
{
  "BED01": {
    "HR": {"value": 92.0, "unit": "bpm", "time": "2026-01-01T00:00:00"}
  }
}
```
- データが20秒以上更新されない値はGUI上でグレー表示（NA含む）。
