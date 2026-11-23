import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from auto_drive import HERE,Session,Receiver
from drive_protocol import validate_config
from protocol import Frame,encode
from test_auto_drive import Motor

class Checks(unittest.TestCase):
    def setUp(self):
        self.config=validate_config(json.loads((HERE/'vehicle_config_mecanum_turn_decay.json').read_text()))
        manifest=json.loads((HERE.parent/'rpi_receiver/manifest.json').read_text())
        self.receiver=Receiver(manifest);self.motor=Motor();self.events=[]
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.stop=Path(self.temp.name)/'STOP'
        self.session=Session(self.receiver,self.motor,self.config,self.events.append,self.stop)
    def step(self,i,now,raw=4):
        data=b'' if i is None else encode(Frame(i,i*315,31611,self.receiver.policy.model_id,raw,1,0x0201,126797,0))
        with patch('auto_drive.time.monotonic',return_value=now):self.session.step(data,now)
    def arm(self):
        self.step(1,.315);self.step(2,.630);self.step(3,.945)
    def limited(self):
        self.arm();self.step(4,1.26);self.step(None,1.4);self.step(None,1.446)
    def test_timer_forces_reduction_between_periodic_motor_sends(self):
        self.arm();self.assertEqual(self.motor.commands[-1],(60,30))
        self.step(4,1.26);self.step(None,1.4)
        self.assertEqual(self.motor.commands[-1],(60,30))
        self.step(None,1.446)
        self.assertEqual(self.motor.commands[-1],(50,40))
        self.assertTrue(self.events[-1]['turn_limited'])
    def test_opposite_sharp_updates_direction_without_new_boost(self):
        self.limited();self.step(5,1.575,3)
        self.assertEqual(self.motor.commands[-1],(40,50))
    def test_single_nonsharp_frame_does_not_reset_boost(self):
        self.limited();self.step(5,1.575,0);self.step(6,1.89,4)
        self.assertEqual(self.motor.commands[-1],(50,40))
    def test_three_nonsharp_live_frames_reset_boost(self):
        self.limited();self.step(5,1.575,0);self.step(6,1.89,1);self.step(7,2.205,0)
        self.step(8,2.52,3)
        self.assertEqual(self.motor.commands[-1],(30,60))
    def test_stop_class_and_safety_still_override_steering(self):
        self.limited();self.step(5,1.575,5)
        self.assertEqual(self.motor.commands[-1],(40,40))
        self.stop.touch()
        with self.assertRaises(RuntimeError):self.step(None,1.69)
        self.stop.unlink()
        with self.assertRaises(TimeoutError):self.step(None,2.5)
    def test_invalid_config(self):
        for key,value in [('boost_seconds',float('nan')),('boost_seconds',float('inf')),('boost_seconds',True),('reset_frames',1),('reset_frames',True),('reset_seconds',0),('relaxed_inner_pwm',0),('relaxed_outer_pwm',70)]:
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                validate_config(dict(self.config,sharp_turn_decay=dict(self.config['sharp_turn_decay'],**{key:value})))

if __name__=='__main__':unittest.main()
