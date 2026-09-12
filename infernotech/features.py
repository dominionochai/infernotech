"""Parallel spatial and temporal-frequency features for pseudo-events."""

from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pywt

from .events import PseudoEvents

FEATURE_NAMES = [
    "event_count",
    "polarity_balance",
    "event_rate_mean",
    "event_rate_variance",
    "wavelet_d1_energy",
    "wavelet_d2_energy",
    "wavelet_d3_energy",
    "wavelet_zero_crossing_rate",
    "spatial_concentration",
    "short_term_persistence",
]


def zero_crossing_rate(signal: np.ndarray) -> float:
    """Return the fraction of adjacent samples that cross zero, in [0, 1]."""
    values = np.asarray(signal, dtype=np.float32).reshape(-1)
    if values.size < 2:
        return 0.0
    signs = np.sign(values)
    # Carry the last nonzero sign through zero-valued samples.
    for i in range(1, signs.size):
        if signs[i] == 0:
            signs[i] = signs[i - 1]
    if signs[0] == 0:
        nonzero = np.flatnonzero(signs)
        if nonzero.size:
            signs[: nonzero[0]] = signs[nonzero[0]]
    return float(np.mean(signs[1:] != signs[:-1]))


def region_slices(shape: Tuple[int, int], grid: Tuple[int, int] = (2, 2)) -> List[Tuple[slice, slice]]:
    """Return row/column slices for a regular region grid."""
    height, width = map(int, shape)
    rows, cols = map(int, grid)
    if height <= 0 or width <= 0 or rows <= 0 or cols <= 0:
        raise ValueError("shape and grid dimensions must be positive")
    y_edges = np.linspace(0, height, rows + 1, dtype=int)
    x_edges = np.linspace(0, width, cols + 1, dtype=int)
    return [
        (slice(y_edges[r], y_edges[r + 1]), slice(x_edges[c], x_edges[c + 1]))
        for r in range(rows) for c in range(cols)
    ]


def _safe_wavelet_features(
    signal: np.ndarray, wavelet: str = "db2", level: int = 3
) -> Tuple[float, float, float, float]:
    """Compute normalized detail energies and ZCR, padding short signals safely."""
    values = np.asarray(signal, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    try:
        wav = pywt.Wavelet(wavelet)
        target = max(values.size, 2 ** (int(level) + 2), wav.dec_len * 2)
        if values.size < target:
            values = np.pad(values, (0, target - values.size), mode="edge")
        coeffs = pywt.wavedec(
            values, wav, level=int(level), mode="periodization"
        )
        details = {idx: np.asarray(coeffs[-idx], dtype=np.float32) for idx in (1, 2, 3)}
        energies = np.asarray([np.mean(details[idx] ** 2) for idx in (1, 2, 3)], dtype=np.float32)
        total = float(np.sum(energies))
        if total > 0:
            energies /= total
        return float(energies[0]), float(energies[1]), float(energies[2]), zero_crossing_rate(values)
    except (ValueError, IndexError, KeyError):
        return 0.0, 0.0, 0.0, zero_crossing_rate(values)


def _as_history(events_history: Sequence[PseudoEvents]) -> List[PseudoEvents]:
    return list(events_history)


def extract_region_features(
    events_history: Sequence[PseudoEvents],
    frame_shape: Tuple[int, int],
    grid: Tuple[int, int] = (2, 2),
    wavelet: str = "db2",
    level: int = 3,
) -> np.ndarray:
    """Extract a float32 matrix with shape ``(regions, 10)``.

    Each history element is one temporal bin. Empty histories are supported and
    produce an all-zero feature matrix.
    """
    history = _as_history(events_history)
    slices = region_slices(frame_shape, grid)
    result = np.zeros((len(slices), len(FEATURE_NAMES)), dtype=np.float32)
    if not history:
        return result

    height, width = map(int, frame_shape)
    for region_index, (ys, xs) in enumerate(slices):
        y0, y1 = ys.start, ys.stop
        x0, x1 = xs.start, xs.stop
        counts = []
        polarities = []
        positions = []
        for events in history:
            in_region = (
                (events.x >= x0) & (events.x < x1) &
                (events.y >= y0) & (events.y < y1)
            )
            counts.append(int(np.count_nonzero(in_region)))
            polarities.append(events.polarity[in_region])
            positions.append((events.x[in_region], events.y[in_region]))

        count_array = np.asarray(counts, dtype=np.float32)
        all_polarities = np.concatenate(polarities) if any(len(p) for p in polarities) else np.empty(0, dtype=np.int8)
        event_count = float(np.sum(count_array))
        polarity_balance = float(np.mean(all_polarities)) if all_polarities.size else 0.0
        rate_mean = float(np.mean(count_array))
        rate_variance = float(np.var(count_array))
        d1, d2, d3, zcr = _safe_wavelet_features(count_array, wavelet, level)

        if event_count:
            occupancy = np.zeros((max(1, y1 - y0), max(1, x1 - x0)), dtype=np.uint8)
            for x_values, y_values in positions:
                occupancy[y_values - y0, x_values - x0] = 1
            spatial = float(np.count_nonzero(occupancy) / event_count)
        else:
            spatial = 0.0
        active = count_array > 0
        persistence = float(np.mean(active[:-1] & active[1:])) if active.size > 1 else 0.0

        result[region_index] = np.asarray(
            [event_count, polarity_balance, rate_mean, rate_variance,
             d1, d2, d3, zcr, min(1.0, spatial), persistence],
            dtype=np.float32,
        )
    return result
