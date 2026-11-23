#ifndef ULT_DRIVE_CORE_H
#define ULT_DRIVE_CORE_H
#include <stdint.h>
#include <stdio.h>
#include <string.h>

// Shared by the actual UNO sketch and the native protocol/watchdog tests.
struct DriveCore {
  // Existing 0..75 commands remain unchanged. Values above 75 are reserved
  // for staged supervised ground-turn tests; the active profile still uses 75.
  static const int PWM_LIMIT = 120;
  static const uint32_t WATCHDOG_MS = 400;
  bool ready, armed, receiving, obstacleBlocked;
  int left, right;
  int wheels[4]; // LF, LR, RF, RR
  uint32_t token, lastCommand, lastByte;
  uint16_t sequence;
  uint8_t used;
  char buffer[64], reply[64];

  DriveCore(): ready(false), armed(false), receiving(false), obstacleBlocked(false), left(0), right(0),
    token(0), lastCommand(0), lastByte(0), sequence(0), used(0) { reply[0]=0; for(uint8_t i=0;i<4;i++) wheels[i]=0; }
  static uint16_t crc(const char *s, size_t n) {
    uint16_t c=0xffff;
    for(size_t i=0;i<n;i++) {
      c^=uint16_t(uint8_t(s[i]))<<8;
      for(uint8_t j=0;j<8;j++) c=(c&0x8000)?uint16_t((c<<1)^0x1021):uint16_t(c<<1);
    }
    return c;
  }
  void stop() { left=right=0; armed=false; token=0; for(uint8_t i=0;i<4;i++) wheels[i]=0; }
  void setObstacleBlocked(bool blocked) {
    obstacleBlocked=blocked;
    if(blocked) stop();
  }
  void tick(uint32_t now) {
    if(armed && uint32_t(now-lastCommand)>=WATCHDOG_MS) stop();
    if(receiving && uint32_t(now-lastByte)>=80) { receiving=false; used=0; stop(); }
  }
  static bool number(const char *s, uint32_t max, uint32_t &out) {
    if(!*s) return false;
    uint32_t v=0;
    while(*s) {
      if(*s<'0'||*s>'9') return false;
      uint8_t d=*s++-'0';
      if(v>max/10 || (v==max/10 && d>max%10)) return false;
      v=v*10+d;
    }
    out=v; return true;
  }
  static bool speed(const char *s, int &out) {
    bool negative=*s=='-'; if(negative) ++s;
    uint32_t value;
    if(!number(s,PWM_LIMIT,value)) return false;
    out=negative?-int(value):int(value); return true;
  }
  void emit(const char *body) {
    snprintf(reply,sizeof(reply),"$%s*%04X#",body,unsigned(crc(body,strlen(body))));
  }
  void error(uint16_t seq, uint8_t reason) {
    stop(); char body[32];
    snprintf(body,sizeof(body),"E,%u,%u",unsigned(seq),unsigned(reason)); emit(body);
  }
  void ack(uint16_t seq) {
    char body[40]; snprintf(body,sizeof(body),"K,%u,%u,%d,%d",unsigned(seq),armed?1:0,left,right); emit(body);
  }
  void parse(uint32_t now) {
    char *star=strchr(buffer,'*');
    if(!star || strlen(star+1)!=4) { error(0,1); return; }
    uint16_t expected=0;
    for(uint8_t j=1;j<=4;j++) {
      char ch=star[j]; uint8_t digit;
      if(ch>='0'&&ch<='9') digit=ch-'0';
      else if(ch>='A'&&ch<='F') digit=ch-'A'+10;
      else { error(0,1); return; }
      expected=uint16_t((expected<<4)|digit);
    }
    if(crc(buffer,size_t(star-buffer))!=expected) { error(0,2); return; }
    *star=0;
    char *fields[8]; uint8_t count=1; fields[0]=buffer;
    for(char *p=buffer;*p;p++) if(*p==',') {
      if(count>=8) { error(0,3); return; }
      *p=0; fields[count++]=p+1;
    }
    uint32_t seq;
    if(count<2 || !number(fields[1],65535,seq) || strlen(fields[0])!=1) { error(0,3); return; }
    char cmd=fields[0][0];
    if(cmd=='Q' && count==2) {
      char body[48]; snprintf(body,sizeof(body),"I,%u,ULTDRV1,%u,%u,%u",unsigned(seq),
        unsigned(PWM_LIMIT),unsigned(WATCHDOG_MS),ready?1:0); emit(body); return;
    }
    if(cmd=='S' && count==2) { stop(); ack(uint16_t(seq)); return; }
    if(cmd=='C' && count==2) {
      char body[32]; snprintf(body,sizeof(body),"C,%u,W4",unsigned(seq)); emit(body); return;
    }
    uint32_t incoming;
    if(count<3 || !number(fields[2],0xffffffffUL,incoming) || incoming==0) { error(uint16_t(seq),3); return; }
    if(cmd=='A' && count==3) {
      stop();
      if(!ready || obstacleBlocked) { error(uint16_t(seq),4); return; }
      armed=true; token=incoming; sequence=uint16_t(seq); lastCommand=now; ack(sequence); return;
    }
    if(cmd=='D' && count==5) {
      int l,r;
      if(!speed(fields[3],l) || !speed(fields[4],r)) { error(uint16_t(seq),5); return; }
      uint16_t delta=uint16_t(uint16_t(seq)-sequence);
      if(!ready || obstacleBlocked || !armed || incoming!=token || delta==0 || delta>=0x8000) { error(uint16_t(seq),6); return; }
      sequence=uint16_t(seq); left=l; right=r;
      wheels[0]=wheels[1]=l; wheels[2]=wheels[3]=r;
      lastCommand=now; ack(sequence); return;
    }
    if(cmd=='W' && count==7) {
      int values[4];
      for(uint8_t i=0;i<4;i++) if(!speed(fields[i+3],values[i])) { error(uint16_t(seq),5); return; }
      uint16_t delta=uint16_t(uint16_t(seq)-sequence);
      if(!ready || obstacleBlocked || !armed || incoming!=token || delta==0 || delta>=0x8000) { error(uint16_t(seq),6); return; }
      sequence=uint16_t(seq); lastCommand=now;
      for(uint8_t i=0;i<4;i++) wheels[i]=values[i];
      left=wheels[0]; right=wheels[2];
      char body[48]; snprintf(body,sizeof(body),"V,%u,1,%d,%d,%d,%d",unsigned(seq),wheels[0],wheels[1],wheels[2],wheels[3]);
      emit(body); return;
    }
    error(uint16_t(seq),3);
  }
  void feed(char ch, uint32_t now) {
    reply[0]=0; tick(now);
    if(ch=='$') { receiving=true; used=0; lastByte=now; return; }
    if(!receiving) return;
    lastByte=now;
    if(ch=='#') { receiving=false; buffer[used]=0; used=0; parse(now); return; }
    if(ch<32 || ch>126 || used>=sizeof(buffer)-1) { receiving=false; used=0; error(0,7); return; }
    buffer[used++]=ch;
  }
};
#endif
