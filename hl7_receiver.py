#!/usr/bin/env python3
"""
HL7メッセージ受信システム
MLLP (Minimal Lower Layer Protocol) over TCP/IP対応
"""

import socket
import threading
import queue
import time
from typing import Optional, Callable
from pathlib import Path
import logging

from hl7_parser import HL7Parser, HL7Message

logger = logging.getLogger(__name__)


class MLLPProtocol:
    """
    MLLP (Minimal Lower Layer Protocol)
    
    HL7メッセージの送受信に使用される低レベルプロトコル
    フォーマット: <VT>HL7_MESSAGE<FS><CR>
    - VT (Vertical Tab): 0x0B (開始マーカー)
    - FS (File Separator): 0x1C (終了マーカー)
    - CR (Carriage Return): 0x0D
    """
    
    START_BLOCK = b'\x0b'  # VT
    END_BLOCK = b'\x1c'    # FS
    CARRIAGE_RETURN = b'\x0d'  # CR
    
    @staticmethod
    def wrap(message: str) -> bytes:
        """HL7メッセージをMLLPでラップ"""
        msg_bytes = message.encode('utf-8')
        return MLLPProtocol.START_BLOCK + msg_bytes + MLLPProtocol.END_BLOCK + MLLPProtocol.CARRIAGE_RETURN
    
    @staticmethod
    def unwrap(data: bytes) -> Optional[str]:
        """MLLPフレームからHL7メッセージを抽出"""
        if not data:
            return None
        
        # 開始マーカーを探す
        start_idx = data.find(MLLPProtocol.START_BLOCK)
        if start_idx == -1:
            return None
        
        # 終了マーカーを探す
        end_idx = data.find(MLLPProtocol.END_BLOCK, start_idx)
        if end_idx == -1:
            return None
        
        # メッセージ抽出
        message = data[start_idx + 1:end_idx]
        return message.decode('utf-8', errors='ignore')


class HL7TCPReceiver:
    """
    HL7 TCP/IPレシーバー
    MLLP over TCP/IP でHL7メッセージを受信
    """
    
    def __init__(self, 
                 host: str = '0.0.0.0',
                 port: int = 2575,
                 callback: Optional[Callable[[HL7Message], None]] = None):
        """
        Args:
            host: バインドするホスト
            port: ポート番号（デフォルト: 2575はHL7標準ポート）
            callback: メッセージ受信時のコールバック関数
        """
        self.host = host
        self.port = port
        self.callback = callback
        self.parser = HL7Parser()
        
        self.server_socket = None
        self.running = False
        self.thread = None
        
        self.message_queue = queue.Queue()
    
    def start(self):
        """受信開始"""
        if self.running:
            logger.warning("Receiver already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.thread.start()
        logger.info(f"HL7 TCP Receiver started on {self.host}:{self.port}")
    
    def stop(self):
        """受信停止"""
        self.running = False
        
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        
        if self.thread:
            self.thread.join(timeout=5)
        
        logger.info("HL7 TCP Receiver stopped")
    
    def _run_server(self):
        """サーバースレッド"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.server_socket.settimeout(1.0)  # タイムアウト設定
            
            logger.info(f"Listening for HL7 messages on {self.host}:{self.port}")
            
            while self.running:
                try:
                    client_socket, client_address = self.server_socket.accept()
                    logger.info(f"Connection from {client_address}")
                    
                    # クライアント処理を別スレッドで
                    client_thread = threading.Thread(
                        target=self._handle_client,
                        args=(client_socket, client_address),
                        daemon=True
                    )
                    client_thread.start()
                
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        logger.error(f"Error accepting connection: {e}")
        
        except Exception as e:
            logger.error(f"Server error: {e}")
        
        finally:
            if self.server_socket:
                self.server_socket.close()
    
    def _handle_client(self, client_socket: socket.socket, client_address):
        """クライアント接続処理"""
        try:
            buffer = b''
            
            while self.running:
                try:
                    data = client_socket.recv(4096)
                    if not data:
                        break
                    
                    buffer += data
                    
                    # メッセージの完全性チェック
                    if MLLPProtocol.END_BLOCK in buffer and MLLPProtocol.CARRIAGE_RETURN in buffer:
                        # メッセージ抽出
                        message_str = MLLPProtocol.unwrap(buffer)
                        
                        if message_str:
                            logger.info(f"Received HL7 message from {client_address}")
                            logger.debug(f"Message:\n{message_str}")
                            
                            # HL7パース
                            hl7_message = self.parser.parse(message_str)
                            
                            if hl7_message:
                                # キューに追加
                                self.message_queue.put(hl7_message)
                                
                                # コールバック実行
                                if self.callback:
                                    try:
                                        self.callback(hl7_message)
                                    except Exception as e:
                                        logger.error(f"Callback error: {e}")
                                
                                # ACK応答送信
                                ack = self._create_ack(hl7_message)
                                ack_wrapped = MLLPProtocol.wrap(ack)
                                client_socket.sendall(ack_wrapped)
                                logger.debug("ACK sent")
                            else:
                                logger.error("Failed to parse HL7 message")
                                # NACK送信
                                nack = self._create_nack()
                                client_socket.sendall(MLLPProtocol.wrap(nack))
                        
                        # バッファクリア
                        buffer = b''
                
                except socket.timeout:
                    continue
                except Exception as e:
                    logger.error(f"Error handling client: {e}")
                    break
        
        finally:
            client_socket.close()
            logger.info(f"Connection closed: {client_address}")
    
    def _create_ack(self, hl7_message: HL7Message) -> str:
        """ACK（肯定応答）メッセージ作成"""
        # 簡易ACK（実際のシステムではより詳細な実装が必要）
        timestamp = time.strftime("%Y%m%d%H%M%S")
        
        ack = f"""MSH|^~\\&|RECEIVER|HOSPITAL|SENDER|MONITOR|{timestamp}||ACK^R01|ACK{timestamp}|P|2.5
MSA|AA|{hl7_message.message_type}||Message accepted"""
        
        return ack
    
    def _create_nack(self) -> str:
        """NACK（否定応答）メッセージ作成"""
        timestamp = time.strftime("%Y%m%d%H%M%S")
        
        nack = f"""MSH|^~\\&|RECEIVER|HOSPITAL|SENDER|MONITOR|{timestamp}||ACK|NACK{timestamp}|P|2.5
MSA|AE|UNKNOWN||Message rejected - parsing error"""
        
        return nack
    
    def get_message(self, timeout: Optional[float] = None) -> Optional[HL7Message]:
        """
        キューからメッセージを取得
        
        Args:
            timeout: タイムアウト（秒）
        
        Returns:
            HL7Message、またはタイムアウト時None
        """
        try:
            return self.message_queue.get(timeout=timeout)
        except queue.Empty:
            return None


class HL7FileWatcher:
    """
    HL7ファイル監視
    指定ディレクトリに配置されたHL7ファイルを読み込む
    """
    
    def __init__(self,
                 watch_dir: str,
                 callback: Optional[Callable[[HL7Message], None]] = None,
                 poll_interval: float = 1.0):
        """
        Args:
            watch_dir: 監視ディレクトリ
            callback: ファイル検出時のコールバック
            poll_interval: ポーリング間隔（秒）
        """
        self.watch_dir = Path(watch_dir)
        self.callback = callback
        self.poll_interval = poll_interval
        self.parser = HL7Parser()
        
        self.processed_files = set()
        self.running = False
        self.thread = None
        
        # ディレクトリ作成
        self.watch_dir.mkdir(parents=True, exist_ok=True)
    
    def start(self):
        """監視開始"""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._watch_loop, daemon=True)
        self.thread.start()
        logger.info(f"HL7 File Watcher started: {self.watch_dir}")
    
    def stop(self):
        """監視停止"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("HL7 File Watcher stopped")
    
    def _watch_loop(self):
        """監視ループ"""
        while self.running:
            try:
                # .hl7ファイルを検索
                hl7_files = list(self.watch_dir.glob("*.hl7"))
                
                for hl7_file in hl7_files:
                    if hl7_file in self.processed_files:
                        continue
                    
                    try:
                        # ファイル読み込み
                        with open(hl7_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        logger.info(f"Processing file: {hl7_file.name}")
                        
                        # HL7パース
                        hl7_message = self.parser.parse(content)
                        
                        if hl7_message:
                            # コールバック実行
                            if self.callback:
                                try:
                                    self.callback(hl7_message)
                                except Exception as e:
                                    logger.error(f"Callback error: {e}")
                        else:
                            logger.error(f"Failed to parse: {hl7_file.name}")
                        
                        # 処理済みマーク
                        self.processed_files.add(hl7_file)
                        
                        # ファイルを処理済みディレクトリに移動（オプション）
                        # processed_dir = self.watch_dir / "processed"
                        # processed_dir.mkdir(exist_ok=True)
                        # hl7_file.rename(processed_dir / hl7_file.name)
                    
                    except Exception as e:
                        logger.error(f"Error processing {hl7_file.name}: {e}")
            
            except Exception as e:
                logger.error(f"Watch loop error: {e}")
            
            time.sleep(self.poll_interval)


def test_tcp_receiver():
    """TCPレシーバーのテスト"""
    received_messages = []
    
    def on_message(hl7_msg: HL7Message):
        print(f"\n[CALLBACK] Received HL7 message:")
        print(f"  Type: {hl7_msg.message_type}")
        print(f"  Patient: {hl7_msg.patient_name} ({hl7_msg.patient_id})")
        print(f"  Vitals: {len(hl7_msg.vitals)}")
        for name, vital in hl7_msg.vitals.items():
            print(f"    {name}: {vital.value} {vital.unit}")
        received_messages.append(hl7_msg)
    
    # レシーバー起動
    receiver = HL7TCPReceiver(host='0.0.0.0', port=2575, callback=on_message)
    receiver.start()
    
    print(f"HL7 TCP Receiver listening on port 2575")
    print("Send HL7 messages using:")
    print("  python3 hl7_sender.py --host localhost --port 2575 --file sample.hl7")
    print("\nPress Ctrl+C to stop...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping receiver...")
        receiver.stop()


def test_file_watcher():
    """ファイル監視のテスト"""
    def on_file(hl7_msg: HL7Message):
        print(f"\n[FILE] Detected HL7 file:")
        print(f"  Patient: {hl7_msg.patient_name}")
        print(f"  Vitals: {len(hl7_msg.vitals)}")
    
    watcher = HL7FileWatcher(watch_dir="./hl7_inbox", callback=on_file)
    watcher.start()
    
    print("Watching ./hl7_inbox for .hl7 files")
    print("Place .hl7 files in that directory to test")
    print("\nPress Ctrl+C to stop...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping watcher...")
        watcher.stop()


if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description='HL7 Receiver Test')
    parser.add_argument('--mode', choices=['tcp', 'file'], default='tcp',
                       help='Receiver mode')
    args = parser.parse_args()
    
    if args.mode == 'tcp':
        test_tcp_receiver()
    else:
        test_file_watcher()
