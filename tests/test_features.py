import numpy as np

from infernotech.events import PseudoEvents
from infernotech.features import FEATURE_NAMES, extract_region_features, zero_crossing_rate


def test_zero_crossing_rate_bounds():
    assert 0.0 <= zero_crossing_rate(np.array([-1, 1, -1, 1], dtype=np.float32)) <= 1.0
    assert zero_crossing_rate(np.zeros(4, dtype=np.float32)) == 0.0


def test_feature_extraction_shape():
    shape = (16, 16)
    batches = []
    for i in range(6):
        x = np.array([1, 2 + i], dtype=np.int32)
        y = np.array([1, 2], dtype=np.int32)
        polarity = np.array([1, -1], dtype=np.int8)
        timestamp = np.full(2, i * 0.033, dtype=np.float64)
        batches.append(PseudoEvents(x, y, polarity, timestamp, shape))
    features = extract_region_features(batches, shape, grid=(2, 2))
    assert features.shape == (4, len(FEATURE_NAMES))
    assert features.dtype == np.float32
    assert np.isfinite(features).all()
