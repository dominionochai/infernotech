"""Pseudo-event generation from ordinary video frames."""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class PseudoEvents:
    """A compact batch of events generated from one frame interval.

    Coordinates are image coordinates (x, y); polarity is +1 for an increase
    and -1 for a decrease in log-like grayscale intensity.
    """

    x: np.ndarray
    y: np.ndarray
    polarity: np.ndarray
    timestamp: np.ndarray
    shape: Tuple[int, int]

    def __len__(self) -> int:
        return int(self.x.size)

    @classmethod
    def empty(cls, shape: Tuple[int, int], timestamp: float = 0.0) -> "PseudoEvents":
        empty_i = np.empty(0, dtype=np.int32)
        empty_p = np.empty(0, dtype=np.int8)
        empty_t = np.empty(0, dtype=np.float64)
        return cls(empty_i, empty_i.copy(), empty_p, empty_t, shape)


class PseudoEventGenerator:
    """Generate positive and negative pseudo-events by frame differencing."""

    def __init__(self, threshold: float = 0.08) -> None:
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        self.threshold = float(threshold)
        self._previous: Optional[np.ndarray] = None

    @staticmethod
    def _gray_float32(frame: np.ndarray) -> np.ndarray:
        if frame is None:
            raise ValueError("frame must not be None")
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif frame.ndim == 2:
            gray = frame
        else:
            raise ValueError("frame must be a 2-D grayscale or 3-D BGR image")
        gray = gray.astype(np.float32, copy=False)
        if gray.size and float(gray.max()) > 1.0:
            gray = gray / 255.0
        return gray

    def generate(self, frame: np.ndarray, timestamp: float = 0.0) -> PseudoEvents:
        """Return events for *frame* relative to the preceding frame."""
        current = self._gray_float32(frame)
        shape = (int(current.shape[0]), int(current.shape[1]))
        if self._previous is None or self._previous.shape != current.shape:
            self._previous = current.copy()
            return PseudoEvents.empty(shape, timestamp)

        delta = current - self._previous
        self._previous = current.copy()
        ys, xs = np.nonzero(np.abs(delta) >= self.threshold)
        polarity = np.where(delta[ys, xs] > 0, 1, -1).astype(np.int8)
        times = np.full(xs.shape, float(timestamp), dtype=np.float64)
        return PseudoEvents(
            xs.astype(np.int32), ys.astype(np.int32), polarity, times, shape
        )

    def reset(self) -> None:
        self._previous = None


def split_polarity(events: PseudoEvents) -> Tuple[PseudoEvents, PseudoEvents]:
    """Return positive and negative subsets of a batch."""
    positive = events.polarity > 0
    negative = ~positive

    def select(mask: np.ndarray) -> PseudoEvents:
        return PseudoEvents(
            events.x[mask], events.y[mask], events.polarity[mask],
            events.timestamp[mask], events.shape
        )

    return select(positive), select(negative)
