import unittest
from unittest.mock import patch
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
from okaffpy import single, multiple
from okaffpy import (
    OKAFF, OKAFFAdaptiveThreshold, detect_first_change,
    detect_multiple_changes, make_rff, estimate_gaussian_gamma,
    okaff_gaussian_theory_moments, gaussian_kernel_theory_terms,
    theoretical_rejection_band,
)


class OKAFFTests(unittest.TestCase):
    def test_frozen_threshold_bounds_and_reset(self):
        threshold = OKAFFAdaptiveThreshold(alpha=.2, quantile=.95, t0=5,
            initialization='theory_A_SV', theory_A=.5, theory_SV=.1)
        for value in (.4, .6, .5, .7):
            self.assertFalse(threshold.update(value))
        threshold.freeze()
        lower, upper = threshold.interval
        state = (threshold.mean, threshold.second_moment, threshold.variance)
        for value, alarm in ((lower, False), (upper, False), (0, True), (upper + 1, True)):
            self.assertEqual(threshold.update(value), alarm)
            self.assertEqual(threshold.interval, (lower, upper))
            self.assertEqual((threshold.mean, threshold.second_moment, threshold.variance), state)
        threshold.reset()
        self.assertFalse(threshold.frozen)
        threshold.update(.5)
        self.assertGreater(threshold.mean, 0)

    def test_direct_fixed_threshold_is_two_sided(self):
        detector = OKAFF(fixed_threshold=(.25, .75))
        with patch.object(detector, 'update_stat', side_effect=[.1, .25, .5, .75, .9]):
            self.assertEqual([detector.update([0]) for _ in range(5)], [True, False, False, False, True])
        for invalid in (.5, (1, 0), (np.nan, 1), None):
            with self.assertRaisesRegex(ValueError, 'fixed_threshold'):
                OKAFF(fixed_threshold=invalid)

    def test_fixed_mode_learns_only_from_burn_in(self):
        stream = np.random.default_rng(9).normal(size=(35, 1))
        burn_in = 20
        _, learned = multiple.initialize_monitor(stream[:burn_in], quantile=.92, n_rff=16,
            seed=2026, thresholding_method='fixed')
        self.assertTrue(learned.frozen)
        lower, upper = learned.interval
        for rate in (.01, .9):
            result = detect_first_change(stream, burn_in=burn_in, n_rff=16,
                thresholding_method='fixed', monitoring_threshold_rate=rate)
            self.assertEqual((result['threshold_lower'], result['threshold_upper']), (lower, upper))
        # Feed the same burn-in statistics, then force either side to alarm.
        for value in (0, 2):
            with patch.object(OKAFF, 'update_stat', side_effect=[.5] * burn_in + [value]):
                result = detect_first_change(stream, burn_in=burn_in, n_rff=16, thresholding_method='fixed')
            self.assertTrue(result['detected'])
            self.assertEqual(result['detected_change_point'], burn_in + 1)
        with patch.object(OKAFF, 'update_stat', return_value=.5):
            result = detect_first_change(stream, burn_in=burn_in, n_rff=16, thresholding_method='fixed')
            self.assertFalse(result['detected'])
            self.assertEqual(result['monitoring_run_length'], 15)

    def test_fixed_mode_relearns_after_each_alarm(self):
        stream = np.random.default_rng(3).normal(size=42)
        stats = [.5] * 20 + [2] + [.2] * 20 + [0]
        with patch.object(OKAFF, 'update_stat', side_effect=stats):
            results = detect_multiple_changes(stream, burn_in=20, n_rff=16, thresholding_method='fixed')
        self.assertEqual(results.detected_change_point.tolist(), [21, 42])
        self.assertEqual(results.alarm_side.tolist(), ['upper', 'lower'])
        self.assertNotEqual(results.threshold_upper.iloc[0], results.threshold_upper.iloc[1])
        for detect in (detect_first_change, detect_multiple_changes):
            with self.assertRaisesRegex(ValueError, 'thresholding_method'):
                detect(stream, burn_in=20, thresholding_method='unknown')

    def test_fixed_mode_cli_matches_python(self):
        stream = np.random.default_rng(4).normal(size=80)
        with TemporaryDirectory() as directory:
            source = Path(directory) / 'input.csv'
            output = Path(directory) / 'output.csv'
            pd.DataFrame({'x': stream}).to_csv(source, index=False)
            for module, detect in ((single, detect_first_change), (multiple, detect_multiple_changes)):
                with patch('sys.argv', ['okaff', str(source), '--thresholding-method', 'fixed', '--n-rff', '16', '--output-csv', str(output)]), redirect_stdout(StringIO()):
                    self.assertEqual(module.main(), 0)
                expected = detect(pd.read_csv(source).to_numpy(), n_rff=16, thresholding_method='fixed')
                if module is single:
                    expected = pd.DataFrame([expected])
                pd.testing.assert_frame_equal(pd.read_csv(output), expected, check_dtype=False)

    def test_detector_options_and_threshold_rates(self):
        stream = np.random.default_rng(19).normal(size=(61, 2))
        options = dict(lambda0=.95, lambda1=.8, eta=0, clip=(.1, .9),
                       reference_threshold_rate=.2, monitoring_threshold_rate=.03)
        detector, threshold = multiple.initialize_monitor(
            stream[:20], quantile=.92, n_rff=16, seed=42, **options)
        self.assertEqual(detector.lambda_t, .8)
        self.assertEqual(detector.clip, (.1, .9))
        self.assertEqual(threshold.alpha, .03)
        theory_a, theory_sv = okaff_gaussian_theory_moments(
            estimate_gaussian_gamma(stream[:20]), np.cov(stream[:20], rowvar=False), .9)
        self.assertAlmostEqual(threshold.theory_A, theory_a)
        self.assertAlmostEqual(threshold.theory_SV, theory_sv)
        # Independent threshold replay verifies both phases use their selected rate.
        replay = OKAFFAdaptiveThreshold(alpha=.2, quantile=.92, t0=21,
            initialization='theory_A_SV', theory_A=theory_a, theory_SV=theory_sv)
        detector.reset()
        for observation in stream[:20]:
            detector.update(observation)
            replay.update(detector.statistic)
        self.assertAlmostEqual(threshold.mean, replay.mean)
        replay.alpha = .03
        for index in range(20, len(stream)):
            detector.update(stream[index])
            if replay.update(detector.statistic):
                break
        result = detect_first_change(stream, burn_in=20, n_rff=16, seed=42, **options)
        self.assertAlmostEqual(result['statistic'], detector.statistic)
        self.assertAlmostEqual(result['threshold_upper'], replay.upper)
        self.assertEqual(result['monitoring_run_length'], index - 19)
        with patch.object(multiple, 'initialize_monitor', wraps=multiple.initialize_monitor) as initialize, patch.object(OKAFFAdaptiveThreshold, 'update', return_value=True):
            alarms = detect_multiple_changes(stream, burn_in=20, n_rff=16, **options)
        self.assertEqual(len(alarms), 2)
        self.assertEqual(initialize.call_count, 2)
        for call in initialize.call_args_list:
            for key, value in options.items():
                self.assertEqual(call.kwargs[key], value)

    def test_detector_option_validation_and_defaults(self):
        stream = np.random.default_rng(4).normal(size=100)
        defaults = dict(lambda0=.999, lambda1=.999, eta=.001, clip=(.001, .999),
                        reference_threshold_rate=.1, monitoring_threshold_rate=.01)
        invalid = dict(lambda0=[-1, 1, np.nan, True], lambda1=[1, np.inf, 'x'],
                       eta=[-1, np.nan, None], clip=[None, (0,), (0, 1, 2), (.9, .1), (-1, .9), (0, 2), (0, np.nan)],
                       reference_threshold_rate=[0, 1, np.inf], monitoring_threshold_rate=[0, 1, np.nan])
        for detect in (detect_first_change, detect_multiple_changes):
            for name, values in invalid.items():
                for value in values:
                    with self.subTest(detect=detect.__name__, name=name, value=value):
                        with self.assertRaisesRegex(ValueError, name):
                            detect(stream, **{name: value})
            before = detect(stream, n_rff=16)
            after = detect(stream, n_rff=16, **defaults)
            if detect is detect_first_change:
                before, after = pd.DataFrame([before]), pd.DataFrame([after])
            pd.testing.assert_frame_equal(before, after)

    def test_cli_detector_options(self):
        flags = ['--lambda0', '.95', '--lambda1', '.8', '--eta', '0', '--clip', '.1', '.9',
                 '--reference-threshold-rate', '.2', '--monitoring-threshold-rate', '.03']
        options = dict(lambda0=.95, lambda1=.8, eta=0, clip=[.1, .9],
                       reference_threshold_rate=.2, monitoring_threshold_rate=.03)
        with TemporaryDirectory() as directory:
            source = Path(directory) / 'input.csv'
            pd.DataFrame({'x': np.random.default_rng(4).normal(size=100)}).to_csv(source, index=False)
            for module, name in ((single, 'detect_first_change'), (multiple, 'detect_multiple_changes')):
                with patch('sys.argv', ['okaff', str(source), '--n-rff', '16'] + flags), redirect_stdout(StringIO()), patch.object(module, name, wraps=getattr(module, name)) as detect:
                    self.assertEqual(module.main(), 0)
                for key, value in options.items():
                    self.assertEqual(detect.call_args.kwargs[key], value)
                for flag, value in (('--eta', '-1'), ('--lambda0', 'nan'), ('--monitoring-threshold-rate', '0')):
                    with patch('sys.argv', ['okaff', str(source), flag, value]), redirect_stderr(StringIO()):
                        with self.assertRaises(SystemExit) as error:
                            module.parse_args()
                        self.assertEqual(error.exception.code, 2)

    def test_custom_burn_in_boundaries_and_resets(self):
        stream = np.random.default_rng(8).normal(size=310)
        for burn_in in (5, 20, 100):
            # Force alarms at the first eligible observation to check exact boundaries.
            with self.subTest(burn_in=burn_in), patch.object(
                OKAFFAdaptiveThreshold, 'update', return_value=True
            ):
                first = detect_first_change(stream, burn_in=burn_in, n_rff=16)
                self.assertEqual(first['detected_change_point'], burn_in + 1)
                self.assertEqual(first['monitoring_run_length'], 1)
                self.assertEqual(first['burn_in'], burn_in)
                alarms = detect_multiple_changes(stream, burn_in=burn_in, n_rff=16)
                expected = np.arange(burn_in + 1, len(stream) + 1, burn_in + 1)
                np.testing.assert_array_equal(alarms.detected_change_point, expected)
                np.testing.assert_array_equal(alarms.burn_in_end, expected - 1)
                np.testing.assert_array_equal(alarms.burn_in_start, expected - burn_in)
                self.assertTrue((alarms.burn_in == burn_in).all())

    def test_custom_burn_in_without_alarms(self):
        stream = np.random.default_rng(8).normal(size=31)
        with patch.object(OKAFFAdaptiveThreshold, 'update', return_value=False):
            first = detect_first_change(stream, burn_in=20, n_rff=16)
            self.assertFalse(first['detected'])
            self.assertEqual(first['monitoring_run_length'], 11)
            self.assertTrue(detect_multiple_changes(stream, burn_in=20, n_rff=16).empty)
        _, threshold = multiple.initialize_monitor(stream[:20, None], quantile=.92, n_rff=16, seed=8)
        self.assertEqual(threshold.t0, 21)
        self.assertEqual(threshold.time, 20)

    def test_burn_in_validation(self):
        stream = np.arange(110)
        for detect in (detect_first_change, detect_multiple_changes):
            for invalid in (True, np.bool_(True), 0, 1, -1, 2.5, 20.0, '20', None, np.nan, np.inf):
                with self.subTest(detect=detect.__name__, invalid=invalid):
                    with self.assertRaisesRegex(ValueError, 'burn_in'):
                        detect(stream, burn_in=invalid)
            with self.assertRaisesRegex(ValueError, 'more than 110'):
                detect(stream, burn_in=110)
            # Explicit default preserves the existing behavior.
            default = detect(stream, n_rff=16)
            explicit = detect(stream, n_rff=16, burn_in=50)
            if detect is detect_first_change:
                default, explicit = pd.DataFrame([default]), pd.DataFrame([explicit])
            pd.testing.assert_frame_equal(default, explicit)

    def test_cli_custom_burn_in(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / 'input.csv'
            pd.DataFrame({'x': np.random.default_rng(8).normal(size=31)}).to_csv(source, index=False)
            for module in (single, multiple):
                output = Path(directory) / 'output.csv'
                argv = ['okaff', str(source), '--burn-in', '20', '--n-rff', '16', '--output-csv', str(output)]
                with patch('sys.argv', argv), redirect_stdout(StringIO()), patch.object(OKAFFAdaptiveThreshold, 'update', return_value=True):
                    self.assertEqual(module.main(), 0)
                result = pd.read_csv(output)
                self.assertEqual(result.burn_in.iloc[0], 20)
                self.assertEqual(result.detected_change_point.iloc[0], 21)
                with patch('sys.argv', ['okaff', str(source), '--burn-in', '1']), redirect_stderr(StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        module.parse_args()
                    self.assertEqual(error.exception.code, 2)

    def test_fixed_forgetting_matches_weighted_mean(self):
        detector = OKAFF(lambda0=0.8, lambda1=0.8, eta=0)
        samples = np.random.default_rng(5).normal(size=(20, 3))
        for i, sample in enumerate(samples):
            detector.update(sample)
            weights = 0.8 ** np.arange(i, -1, -1)
            mean = np.average(samples[:i + 1], axis=0, weights=weights)
            self.assertAlmostEqual(detector.statistic, mean @ mean)

    def test_reset_reproduces_adaptive_sequence(self):
        detector = OKAFF(store_lambdas=True)
        samples = np.random.default_rng(8).normal(size=(30, 2))
        first = [detector.update_stat(x) for x in samples]
        lambdas = detector.lambdas_stored.copy()
        detector.reset()
        np.testing.assert_array_equal(first, [detector.update_stat(x) for x in samples])
        np.testing.assert_array_equal(lambdas, detector.lambdas_stored)

    def test_features_and_bandwidth(self):
        reference = np.random.default_rng(3).normal(size=(50, 2))
        gamma = estimate_gaussian_gamma(reference)
        feature = make_rff(np.random.default_rng(4), gamma=gamma, dimension=2, n_rff=100)
        self.assertAlmostEqual(np.sum(feature(reference[0])**2), 1)
        with self.assertRaises(ValueError):
            estimate_gaussian_gamma(np.ones((50, 2)))

    def test_theory_helpers_agree(self):
        covariance = np.array([[1, .2], [.2, 2]])
        center, scale = okaff_gaussian_theory_moments(.5, covariance, .999)
        theta, eta, zeta = gaussian_kernel_theory_terms(covariance, 1)
        band = theoretical_rejection_band(L=2, lambda_value=.999, theta_K=theta, eta_K=eta, zeta_K=zeta)
        self.assertAlmostEqual(center, band['center'])
        self.assertAlmostEqual(scale, band['SV'])

    def test_threshold_burn_in_and_reset(self):
        threshold = OKAFFAdaptiveThreshold(t0=3)
        self.assertFalse(threshold.update(1))
        self.assertFalse(threshold.update(1))
        threshold.reset()
        self.assertEqual(threshold.time, 0)
        self.assertEqual(threshold.interval, (0, 0))

    def test_batch_results_and_validation(self):
        rng = np.random.default_rng(8)
        stream = np.concatenate([rng.normal(size=150), rng.normal(6, 1, size=150)])
        result = detect_first_change(stream, n_rff=32)
        self.assertTrue(result['detected'])
        self.assertGreater(result['detected_change_point'], 50)
        results = detect_multiple_changes(stream, n_rff=32)
        self.assertGreater(len(results), 0)
        self.assertTrue((np.diff(results.detected_change_point) > 50).all())
        for invalid in [np.ones(50), np.full(60, np.nan), np.empty((60, 0))]:
            with self.assertRaises(ValueError):
                detect_first_change(invalid)
        result = detect_first_change(np.tile(np.arange(50), 2), quantile=.999999, n_rff=32)
        self.assertFalse(result['detected'])


if __name__ == '__main__':
    unittest.main()
