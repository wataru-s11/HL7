#!/usr/bin/env python3
"""
HL7 v2.x メッセージパーサー
生体情報モニターからのバイタルサイン情報抽出
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class HL7VitalSign:
    """HL7から抽出したバイタルサイン"""
    observation_id: str          # LOINC code等
    observation_name: str         # HR, SpO2など
    value: Optional[float]        # 測定値
    unit: str                     # 単位
    reference_range: str = ""     # 基準範囲
    abnormal_flag: str = ""       # N(正常), H(高値), L(低値)等
    observation_time: Optional[datetime] = None
    status: str = "F"             # F(Final), P(Preliminary)等
    
    def is_abnormal(self) -> bool:
        """異常値フラグ判定"""
        return self.abnormal_flag in ["H", "HH", "L", "LL", "A"]
    
    def __repr__(self):
        time_str = self.observation_time.strftime("%H:%M:%S") if self.observation_time else "N/A"
        return (f"HL7VitalSign({self.observation_name}={self.value} {self.unit}, "
                f"flag={self.abnormal_flag}, time={time_str})")


@dataclass
class HL7Message:
    """HL7メッセージ全体"""
    message_type: str              # ORU^R01等
    message_datetime: datetime
    patient_id: str = ""
    patient_name: str = ""
    vitals: Dict[str, HL7VitalSign] = field(default_factory=dict)
    raw_message: str = ""
    
    def get_vital(self, name: str) -> Optional[HL7VitalSign]:
        """バイタルサイン取得"""
        return self.vitals.get(name)


class HL7Parser:
    """
    HL7 v2.x メッセージパーサー
    
    対応メッセージタイプ:
    - ORU^R01: 観察結果報告（バイタルサイン）
    
    生体情報モニターで使用される主なLOINC/観察ID:
    - 8867-4: Heart Rate (HR)
    - 2708-6: Oxygen Saturation (SpO2)
    - 8480-6: Systolic Blood Pressure
    - 8462-4: Diastolic Blood Pressure
    - 8867-4: Pulse Rate
    - 9279-1: Respiratory Rate
    - 8310-5: Body Temperature
    - など
    """
    
    # LOINC/観察IDとバイタル名のマッピング
    OBSERVATION_MAPPING = {
        "8867-4": "HR",           # Heart Rate
        "8889-8": "HR",           # Heart Rate (alternative)
        "2708-6": "SpO2",         # Oxygen Saturation
        "59408-5": "SpO2",        # SpO2 (alternative)
        "8480-6": "NIBP_SYS",     # Systolic BP
        "8462-4": "NIBP_DIA",     # Diastolic BP
        "8478-0": "NIBP_MEAN",    # Mean BP
        "9279-1": "RR",           # Respiratory Rate
        "8310-5": "TEMP",         # Body Temperature
        "8328-7": "TEMP",         # Axillary Temperature
        "8331-1": "TEMP",         # Oral Temperature
        "19935-6": "EtCO2",       # End Tidal CO2
        "60985-9": "CVP",         # Central Venous Pressure
    }
    
    # メーカー独自コードのマッピング（例：Philips, GE, Nihon Kohden）
    VENDOR_MAPPING = {
        "MDC_PULS_OXIM_SAT_O2": "SpO2",
        "MDC_ECG_HEART_RATE": "HR",
        "MDC_PRESS_BLD_NONINV_SYS": "NIBP_SYS",
        "MDC_PRESS_BLD_NONINV_DIA": "NIBP_DIA",
        "MDC_PRESS_BLD_NONINV_MEAN": "NIBP_MEAN",
        "MDC_RESP_RATE": "RR",
        "MDC_TEMP": "TEMP",
        "MDC_AWAY_CO2_ET": "EtCO2",
        "MDC_PRESS_BLD_VEN_CENT": "CVP",
    }
    
    def __init__(self):
        self.field_separator = "|"
        self.component_separator = "^"
        self.repetition_separator = "~"
        self.escape_char = "\\"
        self.subcomponent_separator = "&"
    
    def parse_encoding_characters(self, msh_segment: str):
        """MSHセグメントからエンコーディング文字を抽出（堅牢版）"""
        if not msh_segment.startswith("MSH") or len(msh_segment) < 4:
            return

        # MSH-1: field separator（4文字目）
        self.field_separator = msh_segment[3]

        # MSH-2: encoding characters（フィールドとして取得）
        parts = msh_segment.split(self.field_separator)
        if len(parts) < 2:
            return

        encoding = parts[1]  # 例: "^~\\&"
        if len(encoding) < 4:
            return

        self.component_separator = encoding[0]      # '^'
        self.repetition_separator = encoding[1]     # '~'
        self.escape_char = encoding[2]              # '\\'
        self.subcomponent_separator = encoding[3]   # '&'
 
    def split_segment(self, segment: str) -> List[str]:
        """セグメントをフィールドに分割"""
        return segment.split(self.field_separator)
    
    def split_component(self, field: str) -> List[str]:
        """フィールドをコンポーネントに分割"""
        return field.split(self.component_separator)
    
    def parse_datetime(self, hl7_datetime: str) -> Optional[datetime]:
        """
        HL7日時フォーマットをパース
        例: 20250202153045 → 2025-02-02 15:30:45
        """
        if not hl7_datetime:
            return None
        
        # HL7形式: YYYYMMDDHHmmss[.SSSS][+/-ZZZZ]
        try:
            # タイムゾーンを除去
            dt_str = re.sub(r'[+\-]\d{4}$', '', hl7_datetime)
            
            # 桁数に応じて解析
            if len(dt_str) >= 14:
                return datetime.strptime(dt_str[:14], "%Y%m%d%H%M%S")
            elif len(dt_str) >= 12:
                return datetime.strptime(dt_str[:12], "%Y%m%d%H%M")
            elif len(dt_str) >= 8:
                return datetime.strptime(dt_str[:8], "%Y%m%d")
            else:
                return None
        except ValueError as e:
            logger.warning(f"Failed to parse datetime '{hl7_datetime}': {e}")
            return None
    
    def parse_msh(self, fields: List[str]) -> Tuple[str, datetime]:
        """
        MSHセグメント解析
        
        MSH|^~\&|SendApp|SendFac|RecvApp|RecvFac|20250202153045||ORU^R01|...
        """
        message_type = ""
        message_datetime = datetime.now()
        
        if len(fields) > 8:
            # フィールド9: メッセージタイプ
            msg_type_field = fields[8]
            components = self.split_component(msg_type_field)
            if components:
                message_type = components[0]
        
        if len(fields) > 6:
            # フィールド7: メッセージ日時
            dt = self.parse_datetime(fields[6])
            if dt:
                message_datetime = dt
        
        return message_type, message_datetime
    
    def parse_pid(self, fields: List[str]) -> Tuple[str, str]:
        """
        PIDセグメント解析
        
        PID|1||PATIENT123^^^MRN||YAMADA^TARO||19800101|M|||...
        """
        patient_id = ""
        patient_name = ""
        
        if len(fields) > 3:
            # フィールド3: 患者ID
            pid_field = fields[3]
            components = self.split_component(pid_field)
            if components:
                patient_id = components[0]
        
        if len(fields) > 5:
            # フィールド5: 患者氏名
            name_field = fields[5]
            components = self.split_component(name_field)
            if len(components) >= 2:
                # 姓^名
                patient_name = f"{components[0]} {components[1]}"
            elif components:
                patient_name = components[0]
        
        return patient_id, patient_name
    
    def parse_obx(self, fields: List[str]) -> Optional[HL7VitalSign]:
        """
        OBXセグメント解析（観察結果）
        
        OBX|1|NM|8867-4^Heart Rate^LN||75|bpm|60-100|N|||F|20250202153045||
        
        フィールド構成:
        1: Set ID
        2: Value Type (NM=数値, ST=文字列, etc)
        3: Observation Identifier (LOINC code等)
        4: Observation Sub-ID
        5: Observation Value
        6: Units
        7: Reference Range
        8: Abnormal Flags
        11: Observation Result Status
        14: Observation DateTime
        """
        if len(fields) < 6:
            return None
        
        # フィールド2: Value Type
        value_type = fields[2]
        if value_type not in ["NM", "SN"]:  # 数値型のみ
            return None
        
        # フィールド3: Observation Identifier
        obs_id_field = fields[3]
        obs_components = self.split_component(obs_id_field)
        
        observation_id = obs_components[0] if obs_components else ""
        observation_name_raw = obs_components[1] if len(obs_components) > 1 else ""
        
        # 標準化されたバイタル名を取得
        observation_name = self.OBSERVATION_MAPPING.get(observation_id)
        if not observation_name:
            # ベンダー独自コードをチェック
            observation_name = self.VENDOR_MAPPING.get(observation_id)
        
        if not observation_name:
            # マッピングにない場合は生の名前を使用
            observation_name = observation_name_raw
            logger.debug(f"Unknown observation ID: {observation_id} ({observation_name_raw})")
        
        # フィールド5: Observation Value
        value_str = fields[5]
        value = None
        try:
            if value_str:
                # 数値以外の文字を除去
                value_cleaned = re.sub(r'[^\d.\-+]', '', value_str)
                if value_cleaned:
                    value = float(value_cleaned)
        except ValueError:
            logger.warning(f"Failed to parse value '{value_str}'")
        
        # フィールド6: Units
        unit = fields[6] if len(fields) > 6 else ""
        
        # フィールド7: Reference Range
        reference_range = fields[7] if len(fields) > 7 else ""
        
        # フィールド8: Abnormal Flags
        abnormal_flag = fields[8] if len(fields) > 8 else ""
        
        # フィールド11: Result Status
        status = fields[11] if len(fields) > 11 else "F"
        
        # フィールド14: Observation DateTime
        obs_time = None
        if len(fields) > 14:
            obs_time = self.parse_datetime(fields[14])
        
        return HL7VitalSign(
            observation_id=observation_id,
            observation_name=observation_name,
            value=value,
            unit=unit,
            reference_range=reference_range,
            abnormal_flag=abnormal_flag,
            observation_time=obs_time,
            status=status
        )
    
    def parse(self, hl7_message: str) -> Optional[HL7Message]:
        """
        HL7メッセージ全体を解析
        
        Args:
            hl7_message: HL7メッセージ（改行で区切られたセグメント）
        
        Returns:
            解析されたHL7Message、または解析失敗時None
        """
        if not hl7_message:
            return None
        
        # 改行で分割
        segments = [s.strip() for s in hl7_message.splitlines() if s.strip()]
        
        if not segments:
            return None
        
        # MSHセグメント必須
        msh_segment = segments[0]
        if not msh_segment.startswith("MSH"):
            logger.error("Invalid HL7 message: missing MSH segment")
            return None
        
        # エンコーディング文字を抽出
        self.parse_encoding_characters(msh_segment)
        
        # 初期化
        message_type = ""
        message_datetime = datetime.now()
        patient_id = ""
        patient_name = ""
        vitals = {}
        
        # 各セグメントを解析
        for segment in segments:
            segment_type = segment[:3]
            fields = self.split_segment(segment)
            
            if segment_type == "MSH":
                message_type, message_datetime = self.parse_msh(fields)
            
            elif segment_type == "PID":
                patient_id, patient_name = self.parse_pid(fields)
            
            elif segment_type == "OBX":
                print("DEBUG OBX segment:", segment)
                vital = self.parse_obx(fields)
                print("DEBUG vital:", vital)
                if vital and vital.observation_name:
                    vitals[vital.observation_name] = vital

        
        return HL7Message(
            message_type=message_type,
            message_datetime=message_datetime,
            patient_id=patient_id,
            patient_name=patient_name,
            vitals=vitals,
            raw_message=hl7_message
        )


def test_hl7_parser():
    """HL7パーサーのテスト"""
    # サンプルHL7メッセージ（ORU^R01）
    sample_message = """MSH|^~\\&|Monitor|ICU|HIS|Hospital|20250202153045||ORU^R01|MSG123456|P|2.5
PID|1||PATIENT123^^^MRN||YAMADA^TARO||19800101|M|||
OBR|1||ORDER123|VITAL^Vital Signs|||20250202153045
OBX|1|NM|8867-4^Heart Rate^LN||75|bpm|60-100|N|||F|20250202153045||
OBX|2|NM|2708-6^Oxygen Saturation^LN||98|%|95-100|N|||F|20250202153045||
OBX|3|NM|8480-6^Systolic BP^LN||120|mmHg|90-140|N|||F|20250202153045||
OBX|4|NM|8462-4^Diastolic BP^LN||80|mmHg|60-90|N|||F|20250202153045||
OBX|5|NM|9279-1^Respiratory Rate^LN||16|/min|12-20|N|||F|20250202153045||
OBX|6|NM|8310-5^Body Temperature^LN||36.8|Cel|36.0-37.5|N|||F|20250202153045||"""
    
    parser = HL7Parser()
    result = parser.parse(sample_message)
    
    if result:
        print("=== HL7 Message Parsed ===")
        print(f"Message Type: {result.message_type}")
        print(f"Message DateTime: {result.message_datetime}")
        print(f"Patient ID: {result.patient_id}")
        print(f"Patient Name: {result.patient_name}")
        print(f"\nVital Signs ({len(result.vitals)}):")
        for name, vital in result.vitals.items():
            print(f"  {vital}")
    else:
        print("Failed to parse HL7 message")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_hl7_parser()
