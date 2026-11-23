#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include "drive_core.h"

// PCA9685 channels copied from the supplied hardware example; wheel direction
// and wiring still require a lifted-wheel check on the actual vehicle.
Adafruit_PWMServoDriver pwm(0x40);
DriveCore core;
int applied[4]={1000,1000,1000,1000};
const uint8_t forwardChannels[4]={11,8,13,14};
const uint8_t backwardChannels[4]={10,9,12,15};
// Roboduino V2 module wiring verified with the connected vehicle:
// TRIG=D11, ECHO=D12. The web page's table and code disagree; the sensor-only
// diagnostic produced echo timing only with this code-defined orientation.
const uint8_t ULTRASONIC_TRIG_PIN=11;
const uint8_t ULTRASONIC_ECHO_PIN=12;
const uint32_t ULTRASONIC_TIMEOUT_US=30000UL;
const float OBSTACLE_STOP_CM=5.0f;
const float OBSTACLE_CLEAR_CM=7.0f;
bool obstacleBlocked=false;
uint32_t lastUltrasonicMs=0;

void motor(uint8_t forward, uint8_t backward, int value) {
  uint16_t magnitude=uint32_t(value<0?-value:value)*4095/255;
  // Clear the opposite channel first when changing direction.
  if(value>=0) { pwm.setPWM(backward,0,0); pwm.setPWM(forward,0,magnitude); }
  else { pwm.setPWM(forward,0,0); pwm.setPWM(backward,0,magnitude); }
}
void applyOutputs() {
  if(!core.ready) return;
  for(uint8_t i=0;i<4;i++) if(core.wheels[i]!=applied[i]) {
    motor(forwardChannels[i],backwardChannels[i],core.wheels[i]);
    applied[i]=core.wheels[i];
  }
}
void updateUltrasonic(uint32_t now) {
  if(uint32_t(now-lastUltrasonicMs)<60) return;
  lastUltrasonicMs=now;
  digitalWrite(ULTRASONIC_TRIG_PIN,LOW);
  delayMicroseconds(2);
  digitalWrite(ULTRASONIC_TRIG_PIN,HIGH);
  delayMicroseconds(10);
  digitalWrite(ULTRASONIC_TRIG_PIN,LOW);
  unsigned long duration=pulseIn(ULTRASONIC_ECHO_PIN,HIGH,ULTRASONIC_TIMEOUT_US);
  if(duration==0) return; // no echo is not treated as an obstacle
  float distanceCm=(duration*0.0343f)/2.0f;
  if(!obstacleBlocked && distanceCm<=OBSTACLE_STOP_CM) obstacleBlocked=true;
  else if(obstacleBlocked && distanceCm>OBSTACLE_CLEAR_CM) obstacleBlocked=false;
  core.setObstacleBlocked(obstacleBlocked);
  if(obstacleBlocked) {
    for(uint8_t i=0;i<4;i++) applied[i]=1000;
    applyOutputs();
  }
}
void setup() {
  Serial.begin(115200);
  pinMode(ULTRASONIC_TRIG_PIN,OUTPUT);
  pinMode(ULTRASONIC_ECHO_PIN,INPUT);
  digitalWrite(ULTRASONIC_TRIG_PIN,LOW);
  Wire.begin();
  Wire.setWireTimeout(10000,true);
  core.ready=pwm.begin();
  if(core.ready) { pwm.setPWMFreq(50); applyOutputs(); }
}
void loop() {
  uint32_t now=millis();
  updateUltrasonic(now);
  core.tick(now); applyOutputs();
  for(uint8_t i=0;i<16 && Serial.available();i++) {
    core.feed(char(Serial.read()),millis());
    applyOutputs();
    if(core.reply[0]) Serial.print(core.reply);
    updateUltrasonic(millis());
  }
  if(Wire.getWireTimeoutFlag()) {
    core.stop(); for(uint8_t i=0;i<4;i++) applied[i]=1000;
    applyOutputs(); core.ready=false;
    Wire.clearWireTimeoutFlag();
  }
}
