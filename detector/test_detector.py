from detector import get_detector_class


def run_comparison():
    # TODO: Add comparison logic across multiple detectors.
    detector_class = get_detector_class("LSTM")
    detector = detector_class()
    print("Sample score:", detector.analyze(None))


if __name__ == "__main__":
    run_comparison()
