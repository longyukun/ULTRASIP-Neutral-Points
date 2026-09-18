"""No hardware: exercise polarimetry, feedback failure gates and whole scan."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import brewster_feedback_scan as scan


class FeedbackTests(unittest.TestCase):
    def args(self):
        args = scan.parser().parse_args(['--simulate', '--roll-sign', '1'])
        args.pan_offset = 0.
        return args

    def test_physical_roll_uses_double_angle(self):
        q, u = scan.rotate_stokes(np.array([1.]), np.array([0.]), 45)
        np.testing.assert_allclose(q, 0, atol=1e-12)
        np.testing.assert_allclose(u, -1)
        q2, u2 = scan.rotate_stokes(q, u, -45)
        np.testing.assert_allclose(q2, 1)
        np.testing.assert_allclose(u2, 0, atol=1e-12)

    def test_sensor_signs_and_offsets(self):
        a = self.args()
        a.tilt_offset, a.roll_offset, a.roll_sign = .4, -.2, -1
        self.assertEqual(scan.attitude(2, -20, a), (20.4, -2.2))

    def test_zero_localization(self):
        a = self.args()
        d = scan.Simulation(a)
        images = [d.frame(v) for v in scan.ANGLES]
        result, _, _ = scan.analyze(images, 2., a)
        self.assertTrue(result['valid'])
        self.assertAlmostEqual(result['root_deg'], .12, places=3)
        self.assertAlmostEqual(result['q_center'], -.06, places=3)

    def test_flat_and_saturated_data_rejected(self):
        a = self.args()
        for level in (1000, 4095):
            frames = [np.full((128, 512), level) for _ in range(4)]
            result, _, _ = scan.analyze(frames, 0, a)
            self.assertFalse(result['valid'])

    def test_outside_roi_zero_rejected(self):
        a = self.args()
        d = scan.Simulation(a)
        d.pan = 3
        result, _, _ = scan.analyze([d.frame(v) for v in scan.ANGLES], 2, a)
        self.assertFalse(result['valid'])
        self.assertEqual(result['reason'], 'zero outside ROI')

    def test_feedback_corrects_in_both_pan_directions(self):
        for pan in (-.2, .4):
            a = self.args()
            d = scan.Simulation(a)
            d.pan = pan
            with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
                s = scan.Scanner(a, d, Path(directory))
                result = s.track(20)
                self.assertLessEqual(abs(result['root_deg']), a.pan_tolerance)
                self.assertLessEqual(abs(result['zenith'] - 70), a.tilt_tolerance)

    def test_invalid_fit_never_moves_pan(self):
        a = self.args()
        d = scan.Simulation(a)
        d.frame = lambda angle: np.full((128, 512), 1000)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            s = scan.Scanner(a, d, Path(directory))
            with self.assertRaisesRegex(RuntimeError, 'slope too small'):
                s.track(20)
            self.assertEqual(d.pan, 0)
            self.assertTrue((Path(directory) / 'acquisition_0000' / 'raw_135.npy').exists())

    def test_whole_scan_saves_bracket_and_raw_data(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            code = scan.main(['--simulate', '--zenith-step', '2', '--output', directory])
            self.assertEqual(code, 0)
            output = next(Path(directory).iterdir())
            brackets = json.loads((output / 'candidates.json').read_text())
            self.assertTrue(any(b['zenith_b'] < 55 < b['zenith_a'] for b in brackets))
            rows = [json.loads(row) for row in (output / 'aligned.jsonl').read_text().splitlines()]
            self.assertAlmostEqual(rows[0]['zenith'], 70, delta=.15)
            self.assertAlmostEqual(rows[-1]['zenith'], 45, delta=.15)
            self.assertTrue(all(abs(r['root_deg']) <= .08 for r in rows))
            self.assertEqual(json.loads((output / 'status.json').read_text())['status'], 'complete')

    def test_unreachable_start_fails_before_motion(self):
        a = self.args()
        a.start_zenith = 40
        d = scan.Simulation(a)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'Sun must be above'):
                scan.Scanner(a, d, Path(directory)).run()
        self.assertEqual(d.position(), (0, 20))


if __name__ == '__main__':
    unittest.main()
