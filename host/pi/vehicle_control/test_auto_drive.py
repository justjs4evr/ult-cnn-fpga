"""Exercise the automatic-control state machine using actual FPGA packets."""
import json
from pathlib import Path
import tempfile
import itertools
import unittest
from unittest.mock import patch
from auto_drive import HERE,Session,validate_mode,Receiver
from protocol import Frame,encode

class Motor:
    def __init__(self):self.armed=False;self.commands=[];self.closed=False
    def arm(self):self.armed=True
    def drive(self,l,r):
        assert self.armed
        self.commands.append((l,r))
    def close(self):self.armed=False;self.closed=True


class Checks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.stop=Path(self.temp.name)/'STOP'
        self.config=json.loads((HERE/'vehicle_config.json').read_text(encoding='utf-8'))
        self.manifest=json.loads((HERE.parent/'rpi_receiver/manifest.json').read_text(encoding='utf-8'))
        self.motor=Motor();self.receiver=Receiver(self.manifest)
        self.session=Session(self.receiver,self.motor,self.config,lambda _:None,self.stop)
    def packet(self,i,raw=0,model=None):
        return encode(Frame(i,i*315,31611,self.receiver.policy.model_id if model is None else model,
            raw,1,0x0201,126797,0))
    def step(self,data,now):
        with patch('auto_drive.time.monotonic',return_value=now):self.session.step(data,now)
    def stable(self):
        self.step(self.packet(1),.315);self.step(self.packet(2),.630);self.step(self.packet(3),.945)
    def test_no_arm_until_stable_then_live_updates(self):
        self.step(self.packet(1),.315);self.assertFalse(self.motor.armed)
        self.step(self.packet(2),.630);self.assertFalse(self.motor.armed)
        self.step(self.packet(3),.945);self.assertTrue(self.motor.armed)
        self.assertEqual(self.motor.commands[-1],(45,45))
        # Fragmented transport must not turn an old latest-frame into a new sample.
        packet=self.packet(4,2)
        self.step(packet[:1],1.1);self.step(packet[1:],1.26)
        self.assertEqual(self.motor.commands[-1],(75,24))
        self.step(self.packet(5,5),1.575);self.assertEqual(self.motor.commands[-1],(40,40))
    def test_stale_stream_aborts(self):
        self.stable()
        with self.assertRaises(TimeoutError):self.step(b'',1.8)
    def test_bad_CRC_aborts(self):
        self.stable();bad=bytearray(self.packet(4));bad[12]^=1
        with self.assertRaises(RuntimeError):self.step(bytes(bad),1.26)
    def test_wrong_model_aborts_without_arm(self):
        with self.assertRaises(RuntimeError):self.step(self.packet(1,model=0),.315)
        self.assertFalse(self.motor.armed)
    def test_stop_file_aborts(self):
        self.stable();self.stop.write_text('stop')
        with self.assertRaises(RuntimeError):self.step(b'',1.)
    def test_burst_never_arms(self):
        for i in range(1,8):self.step(self.packet(i),i*.01)
        self.assertFalse(self.motor.armed)
    def test_ground_gate_preserved_and_lifted_bounded(self):
        with self.assertRaises(ValueError):validate_mode('ground',5,False,self.config)
        with self.assertRaises(ValueError):validate_mode('lifted',5,False,self.config)
        with self.assertRaises(ValueError):validate_mode('lifted',11,True,self.config)
        validate_mode('lifted',5,True,self.config)
    def test_outer_runtime_stops_on_rejected_frame(self):
        from auto_drive import run
        class Link(Motor):
            seq=0
            def exchange(self,*_):return ['C','0','W4']
        class Port:
            in_waiting=36
            def __enter__(self):return self
            def __exit__(self,*_):return False
            def reset_input_buffer(self):pass
            def read(inner,_):return self.packet(1,model=0)
        link=Link();ticks=itertools.count()
        with patch('auto_drive.MotorLink',return_value=link),patch('auto_drive.choose_port',return_value='FPGA_fake'), \
             patch('auto_drive.auto_motor_port',return_value='motor_fake'),patch('auto_drive.open_readonly',return_value=Port()), \
             patch('auto_drive.time.monotonic',side_effect=lambda:next(ticks)*.1):
            result=run('lifted',5,True,'motor_fake',Path(self.temp.name)/'runtime')
        self.assertEqual(result,1);self.assertTrue(link.closed);self.assertFalse(link.armed)
        self.assertEqual(link.commands,[])


if __name__=='__main__':unittest.main()
