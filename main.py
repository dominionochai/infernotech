"""InfernoTech command-line video/camera demo."""

import argparse
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from infernotech.classifier import RuleClassifier
from infernotech.events import PseudoEventGenerator, PseudoEvents, split_polarity
from infernotech.features import extract_region_features


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _events_for_polarity(events: PseudoEvents, positive: bool) -> PseudoEvents:
    mask = events.polarity > 0 if positive else events.polarity < 0
    return PseudoEvents(events.x[mask], events.y[mask], events.polarity[mask], events.timestamp[mask], events.shape)


def run(source: Any, config: dict, no_display: bool = False) -> None:
    video = config.get("video", {})
    features_config = config.get("features", {})
    classifier_config = config.get("classifier", {})
    resize = tuple(int(x) for x in video.get("resize", [320, 240]))
    grid = tuple(int(x) for x in video.get("grid", [2, 2]))
    context_bins = int(video.get("context_bins", 16))
    bin_ms = float(video.get("bin_ms", 33))
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    generator = PseudoEventGenerator(float(video.get("threshold", 0.08)))
    positive_history = deque(maxlen=context_bins)
    negative_history = deque(maxlen=context_bins)
    combined_history = deque(maxlen=context_bins)
    classifier = RuleClassifier(
        flame_threshold=float(classifier_config.get("flame_threshold", 0.55)),
        min_events_for_flicker=int(features_config.get("min_events_for_flicker", 10)),
    )
    frame_number = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame = cv2.resize(frame, resize)
            timestamp = frame_number * bin_ms / 1000.0
            events = generator.generate(frame, timestamp)
            positive, negative = split_polarity(events)
            positive_history.append(positive)
            negative_history.append(negative)
            combined_history.append(events)
            frame_number += 1
            if len(combined_history) < 4:
                key = cv2.waitKey(1) & 0xFF if not no_display else -1
                if key in (ord("q"), 27):
                    break
                continue

            matrix = extract_region_features(
                list(combined_history), resize[::-1], grid=grid,
                wavelet=str(features_config.get("wavelet", "db2")),
                level=int(features_config.get("level", 3)),
            )
            prediction = classifier.predict_regions(matrix)
            print(f"flicker_score={prediction['score']:.2f} decision={prediction['label']}", flush=True)
            if not no_display:
                color = (0, 180, 0) if prediction["label"] == "flame" else (0, 0, 180)
                overlay = frame.copy()
                cv2.putText(overlay, f"{prediction['label']} {prediction['score']:.2f}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.imshow("InfernoTech", overlay)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
    finally:
        capture.release()
        if not no_display:
            cv2.destroyAllWindows()


def parse_source(value: str) -> Any:
    try:
        return int(value)
    except ValueError:
        return value


def main() -> None:
    parser = argparse.ArgumentParser(description="InfernoTech pseudo-event flicker classifier")
    parser.add_argument("video", help="video path or camera index")
    parser.add_argument("--config", default="config.yaml", help="YAML configuration path")
    parser.add_argument("--no-display", action="store_true", help="disable OpenCV display")
    args = parser.parse_args()
    run(parse_source(args.video), load_config(args.config), args.no_display)


if __name__ == "__main__":
    main()
