import numpy as np

from infernotech.events import PseudoEventGenerator


def test_event_generation_shapes_and_polarity():
    generator = PseudoEventGenerator(threshold=0.08)
    first = np.zeros((8, 8), dtype=np.uint8)
    second = first.copy()
    second[2:4, 3:5] = 255
    assert len(generator.generate(first)) == 0
    events = generator.generate(second)
    assert events.x.shape == events.y.shape == events.polarity.shape == events.timestamp.shape
    assert events.shape == (8, 8)
    assert len(events) == 4
    assert np.all(events.polarity == 1)
