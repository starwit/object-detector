import pytest

def test_detector_import():
    try:
        from objectdetector.detector import Detector
    except ImportError as e:
        pytest.fail(f"ObjectDetector import failed: {e}")
        
        