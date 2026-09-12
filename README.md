# InfernoTech

**fly-inspired event sensing + parallel temporal-frequency analysis + flicker classifier**

InfernoTech is a clean, CPU-only Python starter that converts ordinary video frame differences into pseudo-events, analyzes activity across a spatial grid and short temporal context, and applies a transparent flicker rule.

## Explicit claims

- This project is **inspired by** biological motion and flicker sensing.
- It is **NOT reconstructing a fly brain**.
- It is **NOT literal cSIFE**.
- It is **NOT claiming flies do wavelets**; the wavelet analysis is an engineering choice for a compact temporal-frequency representation.

## Pipeline

1. Resize a video frame and convert it to grayscale `float32`.
2. Difference consecutive frames at threshold `0.08` to make positive/negative pseudo-events.
3. Keep a short deque of bins and split the image into a 2x2 region grid.
4. Compute event counts, polarity balance, rates, persistence, spatial concentration, and db2 level-3 detail energies with a zero-crossing rate.
5. Score each region with the documented rule and report the strongest region as `flame` or `background`.

## Install and run

Python 3.10+ is recommended. Install the CPU-oriented dependencies:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

Run against a video file:

```bash
python main.py path/to/video.mp4 --no-display
```

Run against camera index 0:

```bash
python main.py 0
```

The display is optional; press `q` or `Esc` to stop it. Configuration is in `config.yaml` and can be replaced with `--config path/to/config.yaml`.

Run the sanity tests:

```bash
python -m pytest -q
```

## Biological basis and related work

The engineering design is informed by these references:

- de Haan & Nordström (2013), cSIFE: https://doi.org/10.1523/JNEUROSCI.5713-12.2013
- Spalthoff (2012): https://doi.org/10.3389/fncir.2012.00072
- Tuthill (2013): https://doi.org/10.1016/j.neuron.2013.05.024
- Lichtsteiner (2008), dynamic vision sensor: https://doi.org/10.1109/JSSC.2007.914337
- FlaDE paper: https://doi.org/10.1016/j.eswa.2024.125746
- FlaDE implementation: https://github.com/KugaMaxx/cocoa-flade

These citations motivate the sensing and event-based framing; they do not imply that this repository reproduces the cited biological systems.

## Real event-camera future support

A future adapter can replace `PseudoEventGenerator` with a real event-camera reader using `dv-processing` and AEDAT4 input. The current implementation deliberately stays runnable with ordinary OpenCV video and does not require event-camera hardware.

## Limitations

This is a starter classifier, not a certified fire alarm. Thresholds, camera placement, lighting, frame rate, and scene content require calibration. Pseudo-events from frame differencing are not equivalent to measurements from a real event camera, and the rule is intentionally interpretable rather than production-validated.
