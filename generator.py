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
from typing import Dict


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


def make_vitals() -> Dict[str, float]:
    return {
        "HR": random.randint(55, 110),
        "SpO2": random.randint(93, 100),
        "NIBP_SYS": random.randint(95, 145),
        "NIBP_DIA": random.randint(55, 95),
        "RR": random.randint(10, 28),
        "TEMP": round(random.uniform(35.8, 38.2), 1),
    }


def make_oru_r01(profile: BedProfile) -> str:
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    msg_id = f"GEN{profile.bed_id}{ts}"
    v = make_vitals()

    return f"""MSH|^~\\&|VIRTUAL_CENTRAL|SIMHOSP|MONITOR|WARD|{ts}||ORU^R01|{msg_id}|P|2.5
PID|1||{profile.patient_id}^^^SIMMRN||{profile.patient_name}||19800101|{profile.sex}|||
PV1|1|I|ICU^01^{profile.bed_id}
OBR|1||ORD{ts}|VITAL^Vital Signs|||{ts}
OBX|1|NM|8867-4^Heart Rate^LN||{v['HR']}|bpm|60-100|N|||F|{ts}||
OBX|2|NM|2708-6^Oxygen Saturation^LN||{v['SpO2']}|%|95-100|N|||F|{ts}||
OBX|3|NM|8480-6^Systolic BP^LN||{v['NIBP_SYS']}|mmHg|90-140|N|||F|{ts}||
OBX|4|NM|8462-4^Diastolic BP^LN||{v['NIBP_DIA']}|mmHg|60-90|N|||F|{ts}||
OBX|5|NM|9279-1^Respiratory Rate^LN||{v['RR']}|/min|12-20|N|||F|{ts}||
OBX|6|NM|8310-5^Body Temperature^LN||{v['TEMP']}|Cel|36.0-37.5|N|||F|{ts}||"""


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
    parser.add_argument("--interval", type=int, default=60, help="送信周期(秒)")
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
