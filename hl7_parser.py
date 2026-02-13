#!/usr/bin/env python3
"""HL7 v2.x メッセージパーサー。"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class HL7VitalSign:
    observation_id: str
    observation_name: str
    value: Optional[float]
    unit: str
    reference_range: str = ""
    abnormal_flag: str = ""
    observation_time: Optional[datetime] = None
    status: str = "F"


@dataclass
class HL7Message:
    message_type: str
    message_datetime: datetime
    patient_id: str = ""
    patient_name: str = ""
    bed_id: str = ""
    vitals: Dict[str, HL7VitalSign] = field(default_factory=dict)
    raw_message: str = ""


class HL7Parser:
    OBSERVATION_MAPPING = {
        "8867-4": "HR",
        "2708-6": "SpO2",
        "8480-6": "NIBP_SYS",
        "8462-4": "NIBP_DIA",
        "8478-0": "NIBP_MEAN",
        "9279-1": "RR",
        "8310-5": "TEMP",
        "19935-6": "EtCO2",
        "60985-9": "CVP",
        # custom ICU mapping
        "ICU_HR": "HR",
        "ICU_ART_S": "ART_S",
        "ICU_ART_D": "ART_D",
        "ICU_ART_M": "ART_M",
        "ICU_CVP_M": "CVP_M",
        "ICU_RAP_M": "RAP_M",
        "ICU_SPO2": "SpO2",
        "ICU_TSKIN": "TSKIN",
        "ICU_TRECT": "TRECT",
        "ICU_RRESP": "rRESP",
        "ICU_ETCO2": "EtCO2",
        "ICU_RR": "RR",
        "ICU_VTE": "VTe",
        "ICU_VTI": "VTi",
        "ICU_PPEAK": "Ppeak",
        "ICU_PEEP": "PEEP",
        "ICU_O2CONC": "O2conc",
        "ICU_NO": "NO",
        "ICU_BSR1": "BSR1",
        "ICU_BSR2": "BSR2",
    }

    def __init__(self):
        self.field_separator = "|"
        self.component_separator = "^"
        self.repetition_separator = "~"
        self.escape_char = "\\"
        self.subcomponent_separator = "&"

    def parse_encoding_characters(self, msh_segment: str):
        if not msh_segment.startswith("MSH") or len(msh_segment) < 4:
            return
        self.field_separator = msh_segment[3]
        parts = msh_segment.split(self.field_separator)
        if len(parts) < 2:
            return
        encoding = parts[1]
        if len(encoding) < 4:
            return
        self.component_separator = encoding[0]
        self.repetition_separator = encoding[1]
        self.escape_char = encoding[2]
        self.subcomponent_separator = encoding[3]

    def split_segment(self, segment: str) -> List[str]:
        return segment.split(self.field_separator)

    def split_component(self, field: str) -> List[str]:
        return field.split(self.component_separator)

    def parse_datetime(self, hl7_datetime: str) -> Optional[datetime]:
        if not hl7_datetime:
            return None
        try:
            dt_str = re.sub(r'[+\-]\d{4}$', '', hl7_datetime)
            if len(dt_str) >= 14:
                return datetime.strptime(dt_str[:14], "%Y%m%d%H%M%S")
            if len(dt_str) >= 12:
                return datetime.strptime(dt_str[:12], "%Y%m%d%H%M")
            if len(dt_str) >= 8:
                return datetime.strptime(dt_str[:8], "%Y%m%d")
            return None
        except ValueError:
            return None

    def parse_msh(self, fields: List[str]) -> Tuple[str, datetime]:
        message_type = ""
        message_datetime = datetime.now()
        if len(fields) > 8:
            message_type = fields[8]
        if len(fields) > 6:
            dt = self.parse_datetime(fields[6])
            if dt:
                message_datetime = dt
        return message_type, message_datetime

    def parse_pid(self, fields: List[str]) -> Tuple[str, str]:
        patient_id = ""
        patient_name = ""
        if len(fields) > 3:
            components = self.split_component(fields[3])
            patient_id = components[0] if components else ""
        if len(fields) > 5:
            components = self.split_component(fields[5])
            if len(components) >= 2:
                patient_name = f"{components[0]} {components[1]}"
            elif components:
                patient_name = components[0]
        return patient_id, patient_name

    def parse_pv1_bed(self, fields: List[str]) -> str:
        if len(fields) <= 3:
            return ""
        location = self.split_component(fields[3])
        if len(location) >= 2:
            return location[1]
        return location[0] if location else ""

    def parse_obx(self, fields: List[str]) -> Optional[HL7VitalSign]:
        if len(fields) < 6:
            return None
        if fields[2] not in ["NM", "SN"]:
            return None
        obs_components = self.split_component(fields[3])
        observation_id = obs_components[0] if obs_components else ""
        observation_name_raw = obs_components[1] if len(obs_components) > 1 else ""
        observation_name = self.OBSERVATION_MAPPING.get(observation_id, observation_name_raw)

        value = None
        value_str = fields[5]
        if value_str:
            cleaned = re.sub(r'[^\d.\-+]', '', value_str)
            if cleaned:
                try:
                    value = float(cleaned)
                except ValueError:
                    value = None

        unit = fields[6] if len(fields) > 6 else ""
        reference_range = fields[7] if len(fields) > 7 else ""
        abnormal_flag = fields[8] if len(fields) > 8 else ""
        status = fields[11] if len(fields) > 11 else "F"
        obs_time = self.parse_datetime(fields[14]) if len(fields) > 14 else None

        return HL7VitalSign(
            observation_id=observation_id,
            observation_name=observation_name,
            value=value,
            unit=unit,
            reference_range=reference_range,
            abnormal_flag=abnormal_flag,
            observation_time=obs_time,
            status=status,
        )

    def parse(self, hl7_message: str) -> Optional[HL7Message]:
        if not hl7_message:
            return None
        segments = [s.strip() for s in hl7_message.splitlines() if s.strip()]
        if not segments or not segments[0].startswith("MSH"):
            return None

        self.parse_encoding_characters(segments[0])
        message_type = ""
        message_datetime = datetime.now()
        patient_id = ""
        patient_name = ""
        bed_id = ""
        vitals = {}

        for segment in segments:
            segment_type = segment[:3]
            fields = self.split_segment(segment)
            if segment_type == "MSH":
                message_type, message_datetime = self.parse_msh(fields)
            elif segment_type == "PID":
                patient_id, patient_name = self.parse_pid(fields)
            elif segment_type == "PV1":
                bed_id = self.parse_pv1_bed(fields)
            elif segment_type == "OBX":
                vital = self.parse_obx(fields)
                if vital and vital.observation_name:
                    vitals[vital.observation_name] = vital

        return HL7Message(
            message_type=message_type,
            message_datetime=message_datetime,
            patient_id=patient_id,
            patient_name=patient_name,
            bed_id=bed_id,
            vitals=vitals,
            raw_message=hl7_message,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
