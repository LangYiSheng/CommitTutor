from detector.model import DefectDetector
from detector.registry import register_detector


@register_detector("LSTM")
class LSTMDefectDetector(DefectDetector):
    def load(self):
        # TODO: Load LSTM weights/resources.
        self._loaded = True

    def analyze(self, commit_info):
        # TODO: Replace with real LSTM scoring logic.
        self._ensure_loaded()
        return 0.72


__all__ = ["LSTMDefectDetector"]
