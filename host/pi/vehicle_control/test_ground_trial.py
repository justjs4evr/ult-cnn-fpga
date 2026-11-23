"""Check ground pulse limits and stop behavior without connected motors."""
import contextlib
import io
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import ground_trial as g


class Link:
    seq = 0
    def __init__(self, fail=False):
        self.armed = False; self.closed = False; self.commands = []; self.fail = fail
    def exchange(self, *_): return ['C', str(self.seq), 'W4']
    def arm(self): self.armed = True
    def drive(self, left, right):
        assert self.armed
        if self.fail: raise TimeoutError('Simulated lost ACK')
        self.commands.append((left, right))
    def close(self): self.closed = True; self.armed = False


class Checks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = json.loads((g.HERE/'vehicle_config.json').read_text(encoding='utf-8'))
        (self.root/'vehicle_config.json').write_text(json.dumps(self.config), encoding='utf-8')
    def run_fake(self, link):
        ticks = itertools.count()
        with patch.object(g, 'HERE', self.root), patch.object(g, 'MotorLink', return_value=link), \
             patch.object(g, 'auto_motor_port', return_value='fake'), \
             patch.object(g.time, 'monotonic', side_effect=lambda: next(ticks)*.025), \
             patch.object(g.time, 'sleep'), contextlib.redirect_stdout(io.StringIO()):
            code = g.run('straight', .5, True, True, output=self.root/'result')
        report = json.loads((self.root/'result/report.json').read_text(encoding='utf-8'))
        return code, report
    def test_limits_before_port_open(self):
        for seconds, floor, power in [(0,True,True),(1.01,True,True),(float('nan'),True,True),(.5,False,True),(.5,True,False)]:
            with self.assertRaises(ValueError): g.validate_trial(self.config, seconds, floor, power)
        with self.assertRaises(ValueError): g.validate_trial(dict(self.config,turn_pwm=121),.5,True,True)
        with self.assertRaises(ValueError): g.validate_trial(dict(self.config,lifted_wheel_mapping_verified=False),.5,True,True)
        with self.assertRaises(ValueError): g.validate_trial(self.config,2,True,True)
        g.validate_trial(dict(self.config,turn_pwm=60),10,True,True,True)
    def test_normal_pulse_stops_without_claiming_calibration(self):
        link = Link(); code, report = self.run_fake(link)
        self.assertEqual(code,0); self.assertTrue(link.closed)
        self.assertTrue(link.commands); self.assertEqual(set(link.commands),{(45,45)})
        self.assertTrue(report['STOP_ACK_verified']); self.assertFalse(report['ground_calibration_completed'])
        self.assertIsNone(report['physical_observation'])
    def test_missing_drive_ACK_still_stops(self):
        link = Link(fail=True); code, report = self.run_fake(link)
        self.assertEqual(code,1); self.assertTrue(link.closed); self.assertTrue(report['STOP_ACK_verified'])
    def test_stop_file_prevents_opening_port(self):
        (self.root/'AUTO_STOP').touch()
        with patch.object(g,'HERE',self.root), patch.object(g,'MotorLink') as constructor:
            with self.assertRaises(RuntimeError):g.run('straight',.5,True,True)
            constructor.assert_not_called()
    def test_stop_during_pulse(self):
        link = Link(); stop = self.root/'AUTO_STOP'
        def arm():link.armed=True; stop.touch()
        link.arm=arm
        with self.assertRaises(RuntimeError):g.pulse(link,dict(left=50,right=50),.5,stop)
        self.assertEqual(link.commands,[])


if __name__ == '__main__': unittest.main()
