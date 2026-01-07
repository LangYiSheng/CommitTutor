from detector.registry import get_detector_class


class DetectorManager:
    def __init__(self):
        self._instances = {}
        self._current_name = None

    def set_current(self, name):
        if name == self._current_name:
            return self._instances.get(name)

        if self._current_name:
            current = self._instances.get(self._current_name)
            if current:
                current.unload()

        detector = self._instances.get(name)
        if detector is None:
            detector_class = get_detector_class(name)
            if detector_class is None:
                return None
            detector = detector_class()
            self._instances[name] = detector

        self._current_name = name
        return detector

    def get_current(self):
        if not self._current_name:
            return None
        return self._instances.get(self._current_name)

    def shutdown(self):
        for detector in self._instances.values():
            detector.unload()
        self._instances.clear()
        self._current_name = None


__all__ = ["DetectorManager"]
