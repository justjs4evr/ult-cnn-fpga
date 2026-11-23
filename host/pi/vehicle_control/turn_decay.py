"""Opt-in reduction of sustained sharp steering; never holds an old direction."""

class SharpTurnDecay:
    def __init__(self,config):
        self.options=config.get('sharp_turn_decay')
        self.started=None
        self.clear_started=None
        self.clear_frames=0

    def observe(self,event,now):
        if not self.options:return
        label=event.get('steering_class')
        if event.get('status')!='ready' or event.get('inhibit_motion',True):return
        if label in ('sharp_left','sharp_right'):
            self.clear_started=None;self.clear_frames=0
        elif label in ('straight','left','right','stop'):
            if self.clear_started is None:self.clear_started=now
            self.clear_frames+=1
            if self.clear_frames>=self.options['reset_frames'] and now-self.clear_started>=self.options['reset_seconds']:
                self.started=None

    def apply(self,target,event,now):
        if not self.options:return target
        label=event.get('steering_class')
        if (event.get('status')!='ready' or event.get('inhibit_motion',True) or
            label not in ('sharp_left','sharp_right') or not target['left'] or not target['right']):
            return target
        if self.started is None:self.started=now
        if now-self.started<self.options['boost_seconds']:return target
        outer=min(self.options['relaxed_outer_pwm'],max(target['left'],target['right']))
        inner=self.options['relaxed_inner_pwm']
        left,right=(inner,outer) if label=='sharp_left' else (outer,inner)
        return dict(target,left=left,right=right,turn_limited=True,
                    requested_left_PWM=target['left'],requested_right_PWM=target['right'])
