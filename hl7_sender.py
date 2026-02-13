#!/usr/bin/env python3
"""仮想セントラルモニタ (generator + sender)。"""

import argparse
import logging
import random
import socket
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


VITAL_DEFS = [
    {"name": "HR", "code": "ICU_HR", "unit": "bpm", "min": 60, "max": 160, "step": 4, "fmt": "{value:03.0f}"},
    {"name": "ART_S", "code": "ICU_ART_S", "unit": "mmHg", "min": 60, "max": 140, "step": 6, "fmt": "{value:03.0f}"},
    {"name": "ART_D", "code": "ICU_ART_D", "unit": "mmHg", "min": 30, "max": 90, "step": 5, "fmt": "{value:03.0f}"},
    {"name": "ART_M", "code": "ICU_ART_M", "unit": "mmHg", "min": 40, "max": 110, "step": 5, "fmt": "{value:03.0f}"},
    {"name": "CVP_M", "code": "ICU_CVP_M", "unit": "mmHg", "min": 0, "max": 20, "step": 2, "fmt": "{value:02.0f}"},
    {"name": "RAP_M", "code": "ICU_RAP_M", "unit": "mmHg", "min": 0, "max": 20, "step": 2, "fmt": "{value:02.0f}"},
    {"name": "SpO2", "code": "ICU_SPO2", "unit": "%", "min": 80, "max": 100, "step": 1, "fmt": "{value:03.0f}"},
    {"name": "TSKIN", "code": "ICU_TSKIN", "unit": "Cel", "min": 34.0, "max": 39.5, "step": 0.2, "fmt": "{value:04.1f}"},
    {"name": "TRECT", "code": "ICU_TRECT", "unit": "Cel", "min": 34.0, "max": 39.5, "step": 0.2, "fmt": "{value:04.1f}"},
    {"name": "rRESP", "code": "ICU_RRESP", "unit": "/min", "min": 5, "max": 60, "step": 3, "fmt": "{value:03.0f}"},
    {"name": "EtCO2", "code": "ICU_ETCO2", "unit": "mmHg", "min": 10, "max": 60, "step": 3, "fmt": "{value:03.0f}"},
    {"name": "RR", "code": "ICU_RR", "unit": "/min", "min": 5, "max": 60, "step": 3, "fmt": "{value:03.0f}"},
    {"name": "VTe", "code": "ICU_VTE", "unit": "mL", "min": 0, "max": 800, "step": 40, "fmt": "{value:03.0f}"},
    {"name": "VTi", "code": "ICU_VTI", "unit": "mL", "min": 0, "max": 800, "step": 40, "fmt": "{value:03.0f}"},
    {"name": "Ppeak", "code": "ICU_PPEAK", "unit": "cmH2O", "min": 5, "max": 40, "step": 2, "fmt": "{value:02.0f}"},
    {"name": "PEEP", "code": "ICU_PEEP", "unit": "cmH2O", "min": 0, "max": 15, "step": 1, "fmt": "{value:02.0f}"},
    {"name": "O2conc", "code": "ICU_O2CONC", "unit": "%", "min": 21, "max": 100, "step": 3, "fmt": "{value:03.0f}"},
    {"name": "NO", "code": "ICU_NO", "unit": "ppm", "min": 0, "max": 40, "step": 2, "fmt": "{value:02.0f}"},
    {"name": "BSR1", "code": "ICU_BSR1", "unit": "", "min": 0, "max": 100, "step": 4, "fmt": "{value:03.0f}"},
    {"name": "BSR2", "code": "ICU_BSR2", "unit": "", "min": 0, "max": 100, "step": 4, "fmt": "{value:03.0f}"},
]


@dataclass
class BedState:
    bed_id: str
    patient_id: str
    patient_name: str
    values: Dict[str, float]


class HL7Sender:
    START_BLOCK = b'\x0b'
    END_BLOCK = b'\x1c'
    CARRIAGE_RETURN = b'\x0d'

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

    def send(self, message: str) -> bool:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(5.0)
                sock.connect((self.host, self.port))
                wrapped = self.START_BLOCK + message.encode("utf-8") + self.END_BLOCK + self.CARRIAGE_RETURN
                sock.sendall(wrapped)
                response = sock.recv(4096)
                return b"MSA|AA" in response
        except Exception as exc:
            logger.error("send failed: %s", exc)
            return False


def random_walk(current: float, vdef: Dict[str, float]) -> float:
    delta = random.uniform(-vdef["step"], vdef["step"])
    new_value = max(vdef["min"], min(vdef["max"], current + delta))
    if isinstance(vdef["min"], int) and isinstance(vdef["max"], int):
        return round(new_value)
    return round(new_value, 1)


def build_oru_message(bed: BedState) -> str:
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    msg_id = f"{bed.bed_id}_{ts}"
    lines: List[str] = [
        f"MSH|^~\\&|VirtualCentral|ICU|Monitor|ICU|{ts}||ORU^R01|{msg_id}|P|2.5",
        f"PID|1||{bed.patient_id}^^^MRN||{bed.patient_name}||19800101|U|||",
        f"PV1|1|I|ICU^{bed.bed_id}",
        f"OBR|1||{bed.bed_id}_{ts}|ICU_PANEL^ICU Vital Panel|||{ts}",
    ]
    for idx, vdef in enumerate(VITAL_DEFS, start=1):
        value = bed.values[vdef["name"]]
        ref = f"{vdef['min']}-{vdef['max']}"
        lines.append(
            f"OBX|{idx}|NM|{vdef['code']}^{vdef['name']}^99ICU||{value}|{vdef['unit']}|{ref}|N|||F|||{ts}|"
        )
    return "\n".join(lines)


def create_beds() -> List[BedState]:
    beds = []
    for i in range(1, 7):
        bed_id = f"BED{i:02d}"
        values = {}
        for v in VITAL_DEFS:
            if isinstance(v["min"], int) and isinstance(v["max"], int):
                values[v["name"]] = random.randint(v["min"], v["max"])
            else:
                values[v["name"]] = round(random.uniform(v["min"], v["max"]), 1)
        beds.append(BedState(bed_id=bed_id, patient_id=f"PT{i:04d}", patient_name=f"PATIENT^{bed_id}", values=values))
    return beds


def run_generator(host: str, port: int, interval: int, cycles: int):
    sender = HL7Sender(host, port)
    beds = create_beds()
    cycle = 0

    while cycles < 0 or cycle < cycles:
        cycle += 1
        logger.info("cycle %s start", cycle)
        for bed in beds:
            for v in VITAL_DEFS:
                bed.values[v["name"]] = random_walk(bed.values[v["name"]], v)
            msg = build_oru_message(bed)
            ok = sender.send(msg)
            logger.info("%s -> %s", bed.bed_id, "ACK" if ok else "NACK/ERR")
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Virtual central monitor HL7 sender")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2575)
    parser.add_argument("--interval", type=int, default=10, help="seconds between cycles")
    parser.add_argument("--cycles", type=int, default=-1, help="-1 for infinite")
    args = parser.parse_args()
    run_generator(args.host, args.port, args.interval, args.cycles)


if __name__ == "__main__":
    main()
