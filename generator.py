#!/usr/bin/env python3
"""
仮想セントラル: 6ベッド分のORU^R01を1分ごとにMLLP/TCP送信する。
本番環境では --enabled false で無効化できる。
"""

from __future__ import annotations

import argparse
import logging
import random
import socket
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Tuple


logger = logging.getLogger(__name__)


START_BLOCK = b"\x0b"
END_BLOCK = b"\x1c"
CARRIAGE_RETURN = b"\x0d"


@dataclass(frozen=True)
class BedProfile:
    bed_id: str
    patient_id: str
    patient_name: str
    sex: str


BEDS = [
    BedProfile("BED01", "SIM001", "YAMADA^TARO", "M"),
    BedProfile("BED02", "SIM002", "SATO^HANAKO", "F"),
    BedProfile("BED03", "SIM003", "SUZUKI^JIRO", "M"),
    BedProfile("BED04", "SIM004", "TAKAHASHI^AI", "F"),
    BedProfile("BED05", "SIM005", "TANAKA^KEN", "M"),
    BedProfile("BED06", "SIM006", "KATO^MIO", "F"),
]


def wrap_mllp(message: str) -> bytes:
    return START_BLOCK + message.encode("utf-8") + END_BLOCK + CARRIAGE_RETURN


TEMP_VITALS = {"TSKIN", "TRECT"}


VITAL_SPECS: Dict[str, Tuple[str, str, str]] = {
    "HR": ("bpm", "60-160", "int"),
    "ART_S": ("mmHg", "70-140", "int"),
    "ART_D": ("mmHg", "40-90", "int"),
    "ART_M": ("mmHg", "50-110", "int"),
    "CVP_M": ("mmHg", "0-20", "int"),
    "RAP_M": ("mmHg", "0-20", "int"),
    "SpO2": ("%", "85-100", "int"),
    "TSKIN": ("Cel", "34.0-39.5", "float"),
    "TRECT": ("Cel", "34.0-39.5", "float"),
    "rRESP": ("/min", "5-60", "int"),
    "EtCO2": ("mmHg", "20-60", "int"),
    "RR": ("/min", "5-60", "int"),
    "VTe": ("mL", "50-800", "int"),
    "VTi": ("mL", "50-800", "int"),
    "Ppeak": ("cmH2O", "10-40", "int"),
    "PEEP": ("cmH2O", "3-12", "int"),
    "O2conc": ("%", "21-100", "int"),
    "NO": ("ppm", "0-40", "int"),
    "BSR1": ("%", "0-100", "int"),
    "BSR2": ("%", "0-100", "int"),
}


def make_vitals() -> Dict[str, float]:
    return {
        "HR": random.randint(60, 160),
        "ART_S": random.randint(70, 140),
        "ART_D": random.randint(40, 90),
        "ART_M": random.randint(50, 110),
        "CVP_M": random.randint(0, 20),
        "RAP_M": random.randint(0, 20),
        "SpO2": random.randint(85, 100),
        "TSKIN": round(random.uniform(34.0, 39.5), 1),
        "TRECT": round(random.uniform(34.0, 39.5), 1),
        "rRESP": random.randint(5, 60),
        "EtCO2": random.randint(20, 60),
        "RR": random.randint(5, 60),
        "VTe": random.randint(50, 800),
        "VTi": random.randint(50, 800),
        "Ppeak": random.randint(10, 40),
        "PEEP": random.randint(3, 12),
        "O2conc": random.randint(21, 100),
        "NO": random.randint(0, 40),
        "BSR1": random.randint(0, 100),
        "BSR2": random.randint(0, 100),
    }


def make_oru_r01(profile: BedProfile) -> str:
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    msg_id = f"GEN{profile.bed_id}{ts}"
    v = make_vitals()

    obx_segments = []
    for idx, key in enumerate(VITAL_SPECS.keys(), start=1):
        unit, ref_range, _value_type = VITAL_SPECS[key]
        value = v[key]
        value_text = f"{float(value):.1f}" if key in TEMP_VITALS else str(int(value))
        obx_segments.append(
            f"OBX|{idx}|NM|{key}^{key}||{value_text}|{unit}|{ref_range}|N|||F|{ts}||"
        )

    return f"""MSH|^~\\&|VIRTUAL_CENTRAL|SIMHOSP|MONITOR|WARD|{ts}||ORU^R01|{msg_id}|P|2.5
PID|1||{profile.patient_id}^^^SIMMRN||{profile.patient_name}||19800101|{profile.sex}|||
PV1|1|I|ICU^01^{profile.bed_id}
OBR|1||ORD{ts}|VITAL^Vital Signs|||{ts}
{chr(10).join(obx_segments)}"""


def send_message(host: str, port: int, message: str) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(5)
            sock.connect((host, port))
            sock.sendall(wrap_mllp(message))
            ack = sock.recv(4096)
            return b"MSA|AA" in ack
    except OSError as exc:
        logger.error("送信エラー: %s", exc)
        return False


def str_to_bool(v: str) -> bool:
    return v.lower() in {"1", "true", "yes", "on"}


def main() -> None:
    parser = argparse.ArgumentParser(description="HL7 virtual central generator")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2575)
    parser.add_argument("--interval", type=int, default=10, help="送信周期(秒)")
    parser.add_argument("--enabled", default="true", help="falseで無効化")
    parser.add_argument("--count", type=int, default=-1, help="送信ループ回数(-1:無限)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if not str_to_bool(args.enabled):
        logger.info("generatorは無効化されています。終了します。")
        return

    loop = 0
    while args.count < 0 or loop < args.count:
        logger.info("%d回目の送信を開始", loop + 1)
        for bed in BEDS:
            ok = send_message(args.host, args.port, make_oru_r01(bed))
            logger.info("bed=%s send=%s", bed.bed_id, "OK" if ok else "NG")
        loop += 1
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
