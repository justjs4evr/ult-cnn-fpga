import json
from pathlib import Path
import unittest
from drive_protocol import validate_config,wheel_targets

ROOT=Path(__file__).resolve().parent

class Checks(unittest.TestCase):
    def setUp(self):
        self.config=validate_config(json.loads((ROOT/'vehicle_config_mecanum_forward_floor30.json').read_text()))
    def test_mapping_in_context_and_without_context(self):
        expected={'left':(30,55),'right':(55,30),'sharp_left':(30,60),'sharp_right':(60,30)}
        for slow in (False,True):
            for label in ('straight','left','right','sharp_left','sharp_right','stop'):
                e=dict(status='ready',inhibit_motion=False,steering_class=label,raw_class=label,slowdown_event=slow)
                t=wheel_targets(e,self.config)
                pair=expected.get(label,(40,40) if slow or label=='stop' else (42,42))
                self.assertEqual((t['left'],t['right']),pair)
                for changes in (dict(status='stale'),dict(inhibit_motion=True)):
                    t=wheel_targets(dict(e,**changes),self.config)
                    self.assertEqual((t['left'],t['right']),(0,0))
    def test_invalid_floor_rejected(self):
        for floor in (-1,40,55,120,True,30.5):
            with self.assertRaises(ValueError):validate_config(dict(self.config,inner_pwm_floor=floor))
        with self.assertRaises(ValueError):validate_config(dict(self.config,sharp_pivot_pwm=45,sharp_pivot_max_seconds=.5))

if __name__=='__main__':unittest.main()
