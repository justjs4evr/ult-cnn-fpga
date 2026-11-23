"""Hysteresis for a learned crossing-context score; no image geometry input."""
import math

class ContextPassagePolicy:
    def __init__(self,activate_probability,keep_probability=.90,clear_seconds=1.50):
        if not 0<=keep_probability<activate_probability<=1 or clear_seconds<=0:
            raise ValueError('Invalid context hysteresis thresholds')
        self.activate_probability=activate_probability;self.keep_probability=keep_probability
        self.clear_seconds=clear_seconds;self.reset()
    def reset(self):self.active=False;self.last_keep=-float('inf')
    def choose(self,context_probability,raw_class,now):
        p=float(context_probability)
        if not math.isfinite(p) or not 0<=p<=1:raise ValueError('Invalid context probability')
        if p>=self.activate_probability:self.active=True;self.last_keep=now
        elif self.active and p>=self.keep_probability:self.last_keep=now
        elif self.active and now-self.last_keep>=self.clear_seconds:self.active=False
        return 'straight' if self.active or raw_class=='stop' else raw_class
