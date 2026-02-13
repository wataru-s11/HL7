#!/usr/bin/env python3
"""HL7 TCP/MLLP receiver + latest.json aggregator."""

import argparse
import json
import logging
import queue
import socket
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional

from hl7_parser import HL7Message, HL7Parser

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class MLLPProtocol:
    START_BLOCK = b'\x0b'
    END_BLOCK = b'\x1c'
    CARRIAGE_RETURN = b'\x0d'

    @staticmethod
    def wrap(message: str) -> bytes:
        return MLLPProtocol.START_BLOCK + message.encode("utf-8") + MLLPProtocol.END_BLOCK + MLLPProtocol.CARRIAGE_RETURN

    @staticmethod
    def unwrap(data: bytes) -> Optional[str]:
        start_idx = data.find(MLLPProtocol.START_BLOCK)
        end_idx = data.find(MLLPProtocol.END_BLOCK, start_idx + 1)
        if start_idx == -1 or end_idx == -1:
            return None
        return data[start_idx + 1:end_idx].decode("utf-8", errors="ignore")


class LatestStore:
    def __init__(self, latest_file: str = "latest.json"):
        self.latest_file = Path(latest_file)
        self.latest: Dict[str, Dict[str, Dict[str, object]]] = {}
        self.lock = threading.Lock()

    def update(self, hl7_message: HL7Message):
        bed_id = hl7_message.bed_id or hl7_message.patient_id or "UNKNOWN"
        event_time = hl7_message.message_datetime or datetime.now()
        with self.lock:
            self.latest.setdefault(bed_id, {})
            for vital_name, vital in hl7_message.vitals.items():
                ts = vital.observation_time or event_time
                self.latest[bed_id][vital_name] = {
                    "value": vital.value,
                    "unit": vital.unit,
                    "time": ts.isoformat(),
                }
            self.latest_file.write_text(json.dumps(self.latest, ensure_ascii=False, indent=2), encoding="utf-8")


class HL7TCPReceiver:
    def __init__(self, host: str = "0.0.0.0", port: int = 2575, callback: Optional[Callable[[HL7Message], None]] = None):
        self.host = host
        self.port = port
        self.callback = callback
        self.parser = HL7Parser()
        self.server_socket: Optional[socket.socket] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.message_queue: "queue.Queue[HL7Message]" = queue.Queue()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.thread.start()
        logger.info("receiver started on %s:%s", self.host, self.port)

    def stop(self):
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        if self.thread:
            self.thread.join(timeout=3)

    def _run_server(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(8)
        self.server_socket.settimeout(1.0)

        while self.running:
            try:
                client_socket, _ = self.server_socket.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=self._handle_client, args=(client_socket,), daemon=True).start()

    def _handle_client(self, client_socket: socket.socket):
        with client_socket:
            buffer = b""
            while self.running:
                data = client_socket.recv(4096)
                if not data:
                    return
                buffer += data
                if MLLPProtocol.END_BLOCK in buffer and MLLPProtocol.CARRIAGE_RETURN in buffer:
                    msg_raw = MLLPProtocol.unwrap(buffer)
                    buffer = b""
                    if not msg_raw:
                        client_socket.sendall(MLLPProtocol.wrap(self._create_nack()))
                        continue
                    hl7_message = self.parser.parse(msg_raw)
                    if not hl7_message:
                        client_socket.sendall(MLLPProtocol.wrap(self._create_nack()))
                        continue
                    self.message_queue.put(hl7_message)
                    if self.callback:
                        self.callback(hl7_message)
                    client_socket.sendall(MLLPProtocol.wrap(self._create_ack(hl7_message)))

    def _create_ack(self, hl7_message: HL7Message) -> str:
        ts = time.strftime("%Y%m%d%H%M%S")
        return f"MSH|^~\\&|RECEIVER|ICU|SENDER|ICU|{ts}||ACK^R01|ACK{ts}|P|2.5\nMSA|AA|{hl7_message.message_type}|OK|Message accepted"

    def _create_nack(self) -> str:
        ts = time.strftime("%Y%m%d%H%M%S")
        return f"MSH|^~\\&|RECEIVER|ICU|SENDER|ICU|{ts}||ACK|NACK{ts}|P|2.5\nMSA|AE|UNKNOWN||Message rejected"


def run_server(host: str, port: int, latest_file: str):
    store = LatestStore(latest_file=latest_file)

    def on_message(msg: HL7Message):
        store.update(msg)
        logger.info("updated bed=%s vitals=%s", msg.bed_id, len(msg.vitals))

    receiver = HL7TCPReceiver(host=host, port=port, callback=on_message)
    receiver.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        receiver.stop()


def main():
    parser = argparse.ArgumentParser(description="HL7 receiver with latest.json export")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=2575)
    parser.add_argument("--latest-file", default="latest.json")
    args = parser.parse_args()
    run_server(args.host, args.port, args.latest_file)


if __name__ == "__main__":
    main()
