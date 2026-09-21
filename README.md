# okaffpy

A standalone Python package implementing OKAFF (online kernel-based changepoint detector with adaptive forgetting factor) changepoint detection. Includes the streaming statistic, Gaussian random Fourier features, adaptive two-sided thresholds, Gaussian theory helpers, and single- and multiple-alarm detection.

Licensed under the [MIT License](LICENSE).

## Install

Requires Python 3.10 or later. From this directory:

```sh
python -m pip install .
```

For development, use `python -m pip install -e .`. Runtime dependencies are NumPy, SciPy, and pandas. No other project folders or `onlinecp` installation are required.

## Detect changes

```python
import numpy as np
from okaffpy import detect_first_change, detect_multiple_changes

rng = np.random.default_rng(2026)
stream = np.concatenate([rng.normal(size=200), rng.normal(3, 1, size=200)])
first = detect_first_change(stream, burn_in=100, quantile=0.92, n_rff=500, seed=2026)
print(first)
alarms = detect_multiple_changes(stream, burn_in=100, quantile=0.92, n_rff=500, seed=2026)
print(alarms)
```

Input can be a one-dimensional numeric series or an `(n_observations, n_features)` array. All values must be finite. Both functions accept `burn_in` (default 50), an integer of at least 2, and require more than `burn_in` observations. The first `burn_in` observations establish the reference distribution. A reference with zero median pairwise distance cannot define the Gaussian bandwidth and raises `ValueError`.

`detect_first_change` returns a dictionary with the alarm status, statistic, threshold interval, and detection time; when no alarm occurs, `detected_change_point` is `pandas.NA`. `detect_multiple_changes` returns a pandas DataFrame (empty with defined columns if no alarms occur). After each alarm it uses the next `burn_in` observations to establish a new baseline; a remaining tail of `burn_in` or fewer observations cannot be monitored.

Both detection functions also accept these keyword arguments:

| Parameter | Default | Meaning and allowed values |
|---|---|---|
| `thresholding_method` | `"adaptive"` | `"adaptive"` updates bounds during monitoring; `"fixed"` freezes the bounds learned during burn-in |
| `lambda0` | `0.999` | Initial forgetting factor; `0 <= lambda0 < 1` |
| `lambda1` | `0.999` | Forgetting factor assigned after the first observation; `0 <= lambda1 < 1` |
| `eta` | `0.001` | Nonnegative learning rate for adapting the forgetting factor |
| `clip` | `(0.001, 0.999)` | Forgetting-factor bounds; `0 <= lower < upper <= 1` |
| `reference_threshold_rate` | `0.1` | Threshold update rate during burn-in; strictly between 0 and 1 |
| `monitoring_threshold_rate` | `0.01` | Threshold update rate during monitoring; strictly between 0 and 1 |

All numeric settings above must be finite. Forgetting factors are clipped to the selected bounds; theoretical threshold initialization uses the clipped `lambda0`. With `eta=0`, adaptation is disabled and the clipped `lambda1` is retained after the first observation. Multiple-change detection reuses all settings after each alarm. Threshold update rates control adaptation speed, not false-alarm probabilities.

```python
alarms = detect_multiple_changes(
    stream, burn_in=100, quantile=0.95, n_rff=500, seed=2026,
    lambda0=0.99, lambda1=0.99, eta=0.0005, clip=(0.01, 0.999),
    reference_threshold_rate=0.15, monitoring_threshold_rate=0.02,
)
```

**Reported change points are one-based alarm positions**, not retrospective estimates of the actual change onset. The quantile controls the adaptive Gaussian interval; it is not a calibrated stream-wide false-alarm probability.

## Fixed thresholds learned during burn-in

Both modes use two-sided thresholds and the same Gaussian-theory initialization. During burn-in, the threshold mean and second moment are updated using `reference_threshold_rate`. In fixed mode, the lower and upper bounds after the final burn-in observation are frozen. Monitoring raises an alarm when the statistic is strictly below the lower bound or above the upper bound; equality does not trigger an alarm. Monitoring observations never update the frozen bounds or their moments. The detector's forgetting factor can still adapt.

```python
first = detect_first_change(stream, burn_in=100, thresholding_method="fixed")
alarms = detect_multiple_changes(stream, burn_in=100, thresholding_method="fixed")
```

`quantile` controls the learned interval width. `monitoring_threshold_rate` has no effect in fixed mode (it must still be a valid rate). Multiple-change detection learns a fresh interval from the next burn-in period after each alarm. The default remains `thresholding_method="adaptive"`.

## Streaming API

```python
from okaffpy import OKAFF, OKAFFAdaptiveThreshold, estimate_gaussian_gamma, make_rff

reference = rng.normal(size=(50, 1))
feature = make_rff(
    np.random.default_rng(2026),
    gamma=estimate_gaussian_gamma(reference), dimension=1, n_rff=500,
)
detector = OKAFF(feat_func=feature, store_values=False)
threshold = OKAFFAdaptiveThreshold(alpha=0.1, quantile=0.95, t0=51)
for observation in reference:
    detector.update(observation)
    threshold.update(detector.statistic)
threshold.alpha = 0.01
for observation in rng.normal(3, 1, size=(100, 1)):
    detector.update(observation)
    if threshold.update(detector.statistic):
        print('Alarm', detector.statistic, threshold.interval)
        break
```

To freeze the interval in the streaming example, call `threshold.freeze()` immediately after the reference loop. Continue calling `threshold.update(detector.statistic)` during monitoring: it checks both bounds without changing them. `threshold.reset()` clears the state and resumes learning.

`OKAFF` uses identity features unless `feat_func` is provided. Its own `update()` checks a fixed `(lower, upper)` pair, defaulting to `(-np.inf, np.inf)`; learning an interval uses the separate threshold object or the high-level functions. **API change:** a scalar `fixed_threshold` is no longer accepted; explicit bounds require `OKAFF(fixed_threshold=(lower, upper))`. `reset()` clears each object's state. `store_lambdas=True` records the forgetting factors; the legacy `store_values` argument is accepted but has no effect. The Gaussian theory helpers support fixed two-sided rejection bands and theory-based adaptive initialization; the high-level detection functions use that initialization automatically.

## Manually supplied fixed thresholds

Use `OKAFF(fixed_threshold=(lower, upper))` when you already have bounds to apply. This is a manual configuration of the fixed method. The high-level functions and CLI learn their fixed bounds from burn-in; they do not accept a manually supplied pair.

The following runnable example uses illustrative bounds of `(0.2, 0.8)`. Choose bounds appropriate for your data and feature map.

```python
import numpy as np
from okaffpy import OKAFF, estimate_gaussian_gamma, make_rff

rng = np.random.default_rng(2026)
data = np.concatenate([rng.normal(size=200), rng.normal(3, 1, size=200)])
data = np.asarray(data, dtype=float)
if data.ndim == 1:
    data = data[:, None]

burn_in = 50
reference = data[:burn_in]
features = make_rff(
    np.random.default_rng(2026),
    gamma=estimate_gaussian_gamma(reference),
    dimension=data.shape[1],
    n_rff=500,
)
detector = OKAFF(
    feat_func=features,
    lambda0=0.999,
    lambda1=0.999,
    eta=0.001,
    fixed_threshold=(0.2, 0.8),
)

# Initialize detector state; ignore alarms during burn-in.
for observation in reference:
    detector.update(observation)

# Monitor with the supplied bounds; positions are one-based.
for position, observation in enumerate(data[burn_in:], start=burn_in + 1):
    if detector.update(observation):
        print(f"Alarm at observation {position}: {detector.statistic}")
        break
```

An alarm occurs when `statistic < lower` or `statistic > upper`. Equality does not trigger an alarm. Burn-in initializes the detector state and supplies reference data for the feature map; it does not learn or change your supplied bounds. Neither `reference_threshold_rate` nor `monitoring_threshold_rate` is used. The forgetting factor can still adapt, and `detector.reset()` preserves the supplied bounds while clearing detector state.

## CSV command line

`--burn-in` defaults to 50 and applies initially and after each alarm.

CSV input must have a header and only numeric feature columns (exclude timestamps and labels).

```sh
okaff-detect-single data.csv --burn-in 100 --output-csv first.csv
okaff-detect-multiple data.csv --burn-in 100 --quantile 0.92 --n-rff 500 --seed 2026 --output-csv alarms.csv
```

Both commands support `--lambda0`, `--lambda1`, `--eta`, `--clip LOWER UPPER`, `--reference-threshold-rate`, and `--monitoring-threshold-rate`, with the defaults and limits listed above.

Choose `--thresholding-method fixed` to learn and freeze two-sided bounds, or `--thresholding-method adaptive` (default) to keep updating them:

```sh
okaff-detect-single data.csv --burn-in 100 --thresholding-method fixed
okaff-detect-multiple data.csv --burn-in 100 --thresholding-method fixed
```

To customize detector parameters:

```sh
okaff-detect-multiple data.csv --burn-in 100 \
  --lambda0 0.99 --lambda1 0.99 --eta 0.0005 --clip 0.01 0.999 \
  --reference-threshold-rate 0.15 --monitoring-threshold-rate 0.02
```

## Tests and packaging

```sh
python -m unittest discover -s tests -v
python -m pip wheel . --no-deps -w dist
```

The implementation lives in `src/okaffpy`. Unrelated detectors, experiment drivers, and old source files are excluded. During this conversion the original folder was preserved in the sibling `okaffpy-original-20260921` archive; that archive is not needed to install or run this package. Existing research folders elsewhere in the workspace are unchanged.
