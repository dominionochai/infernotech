"""Rule and optional two-state HMM classifiers."""

from pathlib import Path
from typing import Any, Iterable, Optional

import joblib
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_NAMES


class RuleClassifier:
    """A transparent flicker rule operating on the ten region features."""

    def __init__(self, flame_threshold: float = 0.55, min_events_for_flicker: int = 10) -> None:
        self.flame_threshold = float(flame_threshold)
        self.min_events_for_flicker = int(min_events_for_flicker)

    def score(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"features must have shape (regions, {len(FEATURE_NAMES)})")
        rate = np.clip(values[:, 2], 0.0, 1.0)
        wavelet = np.clip(values[:, 4] + values[:, 5], 0.0, 1.0)
        zcr = np.clip(values[:, 7], 0.0, 1.0)
        persistence = np.clip(values[:, 9], 0.0, 1.0)
        scores = 0.30 * rate + 0.30 * wavelet + 0.25 * zcr + 0.15 * persistence
        scores = np.where(values[:, 0] < self.min_events_for_flicker, scores * 0.25, scores)
        return np.clip(scores, 0.0, 1.0).astype(np.float32)

    def predict_regions(self, features: np.ndarray) -> dict:
        region_scores = self.score(features)
        score = float(np.max(region_scores)) if region_scores.size else 0.0
        return {
            "score": score,
            "label": "flame" if score >= self.flame_threshold else "background",
            "region_scores": region_scores.tolist(),
            "threshold": self.flame_threshold,
        }


class TwoStateHMM:
    """Optional temporal model with an explicit flame/background state mapping."""

    def __init__(self, random_state: int = 0) -> None:
        self.scaler = StandardScaler()
        self.model = GaussianHMM(
            n_components=2, covariance_type="diag", n_iter=100, random_state=random_state
        )
        self.flame_state: int = 1
        self._fitted = False

    def fit(self, X: np.ndarray, labels: Optional[Iterable[Any]] = None) -> "TwoStateHMM":
        values = np.asarray(X, dtype=np.float32)
        scaled = self.scaler.fit_transform(values)
        self.model.fit(scaled)
        states = self.model.predict(scaled)
        if labels is not None:
            label_values = np.asarray(list(labels))
            flame_mask = np.isin(label_values.astype(str), ["flame", "1", "True", "true"])
            rates = [
                float(np.mean(flame_mask[states == state])) if np.any(states == state) else -1.0
                for state in range(2)
            ]
            self.flame_state = int(np.argmax(rates))
        else:
            self.flame_state = int(np.argmax(self.model.means_[:, 0]))
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TwoStateHMM must be fitted or loaded before prediction")
        states = self.model.predict(self.scaler.transform(np.asarray(X, dtype=np.float32)))
        return np.where(states == self.flame_state, "flame", "background")

    def save(self, path: str) -> None:
        joblib.dump({"scaler": self.scaler, "model": self.model, "flame_state": self.flame_state}, path)

    @classmethod
    def load(cls, path: str) -> "TwoStateHMM":
        payload = joblib.load(Path(path))
        instance = cls()
        instance.scaler = payload["scaler"]
        instance.model = payload["model"]
        instance.flame_state = int(payload["flame_state"])
        instance._fitted = True
        return instance
