#!/usr/bin/env python3
"""
HL7送信テストツール
統合システムのテスト用にHL7メッセージを送信
"""

import socket
import time
import random
from datetime import datetime
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HL7Sender:
    """HL7メッセージ送信クラス"""
    
    # MLLP Protocol
    START_BLOCK = b'\x0b'
    END_BLOCK = b'\x1c'
    CARRIAGE_RETURN = b'\x0d'
    
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
    
    def wrap_mllp(self, message: str) -> bytes:
        """MLLPでラップ"""
        msg_bytes = message.encode('utf-8')
        return self.START_BLOCK + msg_bytes + self.END_BLOCK + self.CARRIAGE_RETURN
    
    def send(self, message: str) -> bool:
        """
        HL7メッセージを送信
        
        Args:
            message: HL7メッセージ
        
        Returns:
            成功時True
        """
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            
            logger.info(f"Connecting to {self.host}:{self.port}...")
            sock.connect((self.host, self.port))
            
            # 送信
            wrapped = self.wrap_mllp(message)
            sock.sendall(wrapped)
            logger.info("Message sent")
            
            # ACK受信
            response = sock.recv(4096)
            if b'MSA|AA' in response:
                logger.info("ACK received")
                return True
            else:
                logger.warning("NACK or unexpected response")
                return False
        
        except Exception as e:
            logger.error(f"Send failed: {e}")
            return False
        
        finally:
            sock.close()
    
    def send_file(self, filepath: str) -> bool:
        """ファイルからHL7メッセージを送信"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                message = f.read()
            
            logger.info(f"Sending file: {filepath}")
            return self.send(message)
        
        except Exception as e:
            logger.error(f"Failed to read file: {e}")
            return False


def generate_vital_signs_message(patient_id: str = "TEST001",
                                 patient_name: str = "TEST^PATIENT",
                                 with_variation: bool = False) -> str:
    """
    バイタルサインHL7メッセージを生成
    
    Args:
        patient_id: 患者ID
        patient_name: 患者氏名
        with_variation: バイタルサインに変動を加える
    
    Returns:
        HL7メッセージ
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    msg_id = f"MSG{timestamp}"
    
    # 基準値
    base_values = {
        "HR": 75,
        "SpO2": 98,
        "NIBP_SYS": 120,
        "NIBP_DIA": 80,
        "RR": 16,
        "TEMP": 36.8,
    }
    
    # 変動を加える
    if with_variation:
        vitals = {
            "HR": base_values["HR"] + random.randint(-5, 5),
            "SpO2": base_values["SpO2"] + random.randint(-1, 1),
            "NIBP_SYS": base_values["NIBP_SYS"] + random.randint(-10, 10),
            "NIBP_DIA": base_values["NIBP_DIA"] + random.randint(-8, 8),
            "RR": base_values["RR"] + random.randint(-2, 2),
            "TEMP": round(base_values["TEMP"] + random.uniform(-0.3, 0.3), 1),
        }
    else:
        vitals = base_values
    
    message = f"""MSH|^~\\&|Monitor|ICU|HIS|Hospital|{timestamp}||ORU^R01|{msg_id}|P|2.5
PID|1||{patient_id}^^^MRN||{patient_name}||19800101|M|||
OBR|1||ORDER{timestamp}|VITAL^Vital Signs|||{timestamp}
OBX|1|NM|8867-4^Heart Rate^LN||{vitals['HR']}|bpm|60-100|N|||F|{timestamp}||
OBX|2|NM|2708-6^Oxygen Saturation^LN||{vitals['SpO2']}|%|95-100|N|||F|{timestamp}||
OBX|3|NM|8480-6^Systolic BP^LN||{vitals['NIBP_SYS']}|mmHg|90-140|N|||F|{timestamp}||
OBX|4|NM|8462-4^Diastolic BP^LN||{vitals['NIBP_DIA']}|mmHg|60-90|N|||F|{timestamp}||
OBX|5|NM|9279-1^Respiratory Rate^LN||{vitals['RR']}|/min|12-20|N|||F|{timestamp}||
OBX|6|NM|8310-5^Body Temperature^LN||{vitals['TEMP']}|Cel|36.0-37.5|N|||F|{timestamp}||"""
    
    return message


def continuous_send(host: str, port: int, interval: float = 5.0, count: int = 10):
    """
    継続的にHL7メッセージを送信
    
    Args:
        host: 送信先ホスト
        port: 送信先ポート
        interval: 送信間隔（秒）
        count: 送信回数（-1で無限）
    """
    sender = HL7Sender(host, port)
    
    sent = 0
    try:
        while count == -1 or sent < count:
            # メッセージ生成（変動あり）
            message = generate_vital_signs_message(
                patient_id=f"TEST{sent:03d}",
                with_variation=True
            )
            
            # 送信
            success = sender.send(message)
            
            if success:
                sent += 1
                logger.info(f"Progress: {sent}/{count if count > 0 else '∞'}")
            else:
                logger.warning("Send failed, retrying...")
            
            time.sleep(interval)
    
    except KeyboardInterrupt:
        logger.info(f"\nStopped. Total sent: {sent}")


def save_sample_message(filepath: str):
    """サンプルメッセージをファイルに保存"""
    message = generate_vital_signs_message()
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(message)
    
    logger.info(f"Sample message saved: {filepath}")


def main():
    parser = argparse.ArgumentParser(description='HL7 Sender Test Tool')
    
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    # send コマンド
    send_parser = subparsers.add_parser('send', help='Send HL7 message')
    send_parser.add_argument('--host', type=str, default='localhost',
                            help='Destination host')
    send_parser.add_argument('--port', type=int, default=2575,
                            help='Destination port')
    send_parser.add_argument('--file', type=str,
                            help='HL7 file to send')
    
    # continuous コマンド
    cont_parser = subparsers.add_parser('continuous', 
                                       help='Send messages continuously')
    cont_parser.add_argument('--host', type=str, default='localhost',
                            help='Destination host')
    cont_parser.add_argument('--port', type=int, default=2575,
                            help='Destination port')
    cont_parser.add_argument('--interval', type=float, default=5.0,
                            help='Interval between messages (seconds)')
    cont_parser.add_argument('--count', type=int, default=10,
                            help='Number of messages (-1 for infinite)')
    
    # generate コマンド
    gen_parser = subparsers.add_parser('generate', help='Generate sample message')
    gen_parser.add_argument('--output', type=str, default='sample_hl7.hl7',
                           help='Output file')
    
    args = parser.parse_args()
    
    if args.command == 'send':
        sender = HL7Sender(args.host, args.port)
        
        if args.file:
            sender.send_file(args.file)
        else:
            # デフォルトメッセージ送信
            message = generate_vital_signs_message()
            sender.send(message)
    
    elif args.command == 'continuous':
        continuous_send(args.host, args.port, args.interval, args.count)
    
    elif args.command == 'generate':
        save_sample_message(args.output)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
