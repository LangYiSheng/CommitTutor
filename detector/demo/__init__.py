from detector.model import DefectDetector
from git_utils.models import CommitData
from detector.registry import register_detector


@register_detector("MODEL_DEMO")
class DEMODefectDetector(DefectDetector):
    def load(self):
        # TODO: Load LSTM weights/resources.
        self._loaded = True

    def analyze(self, commit_info: CommitData):
        # TODO: Replace with real LSTM scoring logic.
        self._ensure_loaded()
        return 0.72


__all__ = ["DEMODefectDetector"]
