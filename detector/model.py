class DefectDetector:
    def __init__(self):
        self._loaded = False

    def load(self):
        # TODO: Load model weights or other assets.
        self._loaded = True

    def unload(self):
        # TODO: Release heavy resources if needed.
        self._loaded = False

    def _ensure_loaded(self):
        if not self._loaded:
            self.load()

    def analyze(self, commit_info):
        raise NotImplementedError("DefectDetector subclasses must implement analyze().")


__all__ = ["DefectDetector"]
