"""Offline handoff-specific checks: parity, readiness and explicit port support."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import CAR_START as app
from drive_protocol import wheel_targets


class Checks(unittest.TestCase):
    def setUp(self):
        self.settings,self.profile=app.load_settings()

    def test_last_profile_output_parity_every_class_and_crossing(self):
        original=json.loads((app.VEHICLE/'vehicle_config_mecanum_gentle52_sharp65.json').read_text(encoding='utf-8'))
        for label in ('straight','left','right','sharp_left','sharp_right','stop'):
            for slow in (False,True):
                event=dict(status='ready',inhibit_motion=False,raw_class=label,steering_class=label,slowdown_event=slow)
                self.assertEqual(wheel_targets(event,self.profile),wheel_targets(event,original))

    def test_unready_motion_does_not_open_motor_port(self):
        with patch('vehicle_control.MotorLink',side_effect=AssertionError('must not open')):
            for argv in (['drive'],['fixed'],['drive','--clear-floor'],['drive','--motor-power-on']):
                with self.assertRaises(ValueError):app.action(app.parser().parse_args(argv))

    def test_invalid_settings_rejected_before_motion(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'settings.json'
            for key,value in [('straight_pwm',True),('sharp_outer_pwm',121),('slow_pwm',43),('sharp_inner_pwm',65)]:
                settings=dict(self.settings,**{key:value})
                path.write_text(json.dumps(settings),encoding='utf-8')
                with self.assertRaises(ValueError):app.load_settings(path)

    def test_preview_uses_explicit_port_without_motor_link(self):
        with patch('vehicle_control.preview',return_value=0) as preview, \
             patch('vehicle_control.MotorLink',side_effect=AssertionError('must not open')):
            result=app.action(app.parser().parse_args(['preview','--fpga-port','FPGA_new']))
            self.assertEqual(result,0)
            self.assertEqual(preview.call_args.kwargs['fpga_port'],'FPGA_new')

    def test_drive_passes_config_and_explicit_ports(self):
        with patch('supervised_cnn_ground.run',return_value=0) as run, \
             patch.object(app,'write_profile',return_value=Path('temporary_profile.json')):
            result=app.action(app.parser().parse_args(['drive','--clear-floor','--motor-power-on',
                                                      '--motor-port','UNO_new','--fpga-port','FPGA_new']))
            self.assertEqual(result,0)
            self.assertEqual(run.call_args.kwargs['motor_port'],'UNO_new')
            self.assertEqual(run.call_args.kwargs['fpga_port'],'FPGA_new')


if __name__=='__main__':unittest.main()
