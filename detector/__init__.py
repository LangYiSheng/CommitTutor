from detector.demo import DEMODefectDetector
from detector.lstm_v1 import LSTMV1Detector
from detector.lstm_v2 import LSTMV2Detector
from detector.transformer_v1 import TransformerV1Detector
from detector.transformer_v2 import TransformerV2Detector

from detector.manager import DetectorManager
from detector.registry import get_detector_class, list_detectors, register_detector

__all__ = [
    "DetectorManager",
    "get_detector_class",
    "list_detectors",
    "register_detector",
    "DEMODefectDetector",
    "LSTMV1Detector",
    "LSTMV2Detector",
    "TransformerV1Detector",
    "TransformerV2Detector"
]


