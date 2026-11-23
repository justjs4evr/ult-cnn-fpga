"""Regression for low-output turning during crossing slowdown; no motor access."""
import json
from pathlib import Path
import unittest
from drive_protocol import validate_config, wheel_targets

ROOT=Path(__file__).resolve().parent

class SlowTurnChecks(unittest.TestCase):
    def setUp(self):
        self.config=validate_config(json.loads((ROOT/'vehicle_config_torque120_slowturn.json').read_text(encoding='utf-8')))
    def target(self,label,slow=True,**changes):
        event=dict(status='ready',inhibit_motion=False,steering_class=label,
                   raw_class=label,slowdown_event=slow)
        event.update(changes)
        result=wheel_targets(event,self.config)
        return result['left'],result['right']
    def test_crossing_preserves_direction_and_uses_ground_tested_outputs(self):
        expected={'straight':(40,40),'left':(29,90),'right':(90,29),
                  'sharp_left':(18,90),'sharp_right':(90,18),'stop':(40,40)}
        for label,pair in expected.items():
            with self.subTest(label=label):self.assertEqual(self.target(label),pair)
    def test_normal_outputs_unchanged(self):
        expected={'straight':(45,45),'left':(34,105),'right':(105,34),
                  'sharp_left':(24,120),'sharp_right':(120,24)}
        for label,pair in expected.items():
            with self.subTest(label=label):self.assertEqual(self.target(label,False),pair)
    def test_inhibit_and_stale_override_turns(self):
        self.assertEqual(self.target('sharp_right',inhibit_motion=True),(0,0))
        self.assertEqual(self.target('sharp_right',status='stale'),(0,0))
    def test_slow_turn_must_not_exceed_normal_turn(self):
        for key in ('slow_gentle_turn_pwm','slow_sharp_turn_pwm'):
            for invalid in (0,-1,121,True,90.5):
                with self.subTest(key=key,value=invalid), self.assertRaises(ValueError):
                    validate_config(dict(self.config,**{key:invalid}))
    def test_legacy_config_keeps_legacy_mapping(self):
        legacy={k:v for k,v in self.config.items() if not k.startswith('slow_') or k=='slow_pwm'}
        event=dict(status='ready',inhibit_motion=False,steering_class='sharp_right',slowdown_event=True)
        result=wheel_targets(event,legacy)
        self.assertEqual((result['left'],result['right']),(40,8))

if __name__=='__main__':unittest.main()
