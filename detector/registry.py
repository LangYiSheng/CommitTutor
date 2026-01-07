_REGISTRY = {}


def register_detector(name):
    def decorator(cls):
        _REGISTRY[name] = cls
        cls.registry_name = name
        return cls

    return decorator


def list_detectors():
    return sorted(_REGISTRY.keys())


def get_detector_class(name):
    return _REGISTRY.get(name)


__all__ = ["register_detector", "list_detectors", "get_detector_class"]
