#include "firmware/ULT_Vehicle_Control/drive_core.h"
#include <assert.h>
#include <iostream>

static void send(DriveCore &c,const char *body,uint32_t now,bool corrupt=false) {
  char packet[96]; uint16_t check=DriveCore::crc(body,strlen(body));
  if(corrupt) check^=1;
  snprintf(packet,sizeof(packet),"$%s*%04X#",body,unsigned(check));
  for(const char *p=packet;*p;p++) c.feed(*p,now);
}
static void arm(DriveCore &c,uint32_t now) { c.ready=true; send(c,"A,65534,1234",now); assert(c.armed && c.left==0); }
int main(int argc,char **argv) {
  if(argc==2 && strcmp(argv[1],"--exchange")==0) {
    DriveCore simulator; simulator.ready=true;
    unsigned long now; std::string packet;
    while(std::cin>>now>>packet) {
      for(size_t i=0;i<packet.size();i++) simulator.feed(packet[i],uint32_t(now));
      std::cout<<simulator.reply<<std::endl;
    }
    return 0;
  }
  assert(DriveCore::crc("123456789",9)==0x29b1);
  DriveCore c; assert(!c.armed && c.left==0);
  send(c,"Q,1",0); assert(strstr(c.reply,"ULTDRV1,120,400,0"));
  send(c,"A,2,1234",0); assert(!c.armed);
  arm(c,0); send(c,"D,65535,1234,35,35",10); assert(c.left==35 && c.right==35);
  send(c,"D,0,1234,8,35",20); assert(c.left==8 && c.right==35);
  c.tick(419); assert(c.armed); c.tick(420); assert(!c.armed && c.left==0);
  send(c,"D,1,1234,35,35",421); assert(!c.armed);
  arm(c,1000); send(c,"D,65535,1234,35,35",1001,true); assert(!c.armed && c.left==0);
  arm(c,2000); send(c,"D,65535,1234,121,35",2001); assert(!c.armed);
  arm(c,3000); send(c,"D,65535,99,35,35",3001); assert(!c.armed);
  arm(c,4000); send(c,"D,65535,1234,-35,35",4001); assert(c.left==-35);
  send(c,"D,65535,1234,35,35",4002); assert(!c.armed); // duplicate fails closed
  arm(c,5000); send(c,"D,65533,1234,35,35",5001); assert(!c.armed); // older sequence
  arm(c,6000); c.feed('$',6001); c.feed('D',6001); c.tick(6081); assert(!c.armed);
  arm(c,7000); c.feed('$',7001); for(int i=0;i<80;i++) c.feed('9',7001); assert(!c.armed);
  arm(c,8000); send(c,"D,65535,1234,35,35",8001); send(c,"S,3",8002); assert(!c.armed && c.left==0);
  arm(c,0xfffffff0UL); send(c,"D,65535,1234,35,35",0xfffffff1UL); c.tick(400); assert(!c.armed);
  arm(c,9000); send(c,"D,65535,1234,,35",9001); assert(!c.armed);
  arm(c,10000); send(c,"C,1",10001); assert(strstr(c.reply,"W4"));
  send(c,"W,65535,1234,50,0,0,0",10002); assert(c.wheels[0]==50 && c.wheels[1]==0 && c.wheels[2]==0 && c.wheels[3]==0);
  send(c,"W,0,1234,0,0,0,50",10003); assert(c.wheels[0]==0 && c.wheels[3]==50);
  c.tick(10403); for(int i=0;i<4;i++) assert(c.wheels[i]==0);
  arm(c,11000); send(c,"W,65535,1234,0,121,0,0",11001); assert(!c.armed);
  c.setObstacleBlocked(true); assert(!c.armed && c.left==0 && c.wheels[0]==0);
  send(c,"A,2,1234",12000); assert(!c.armed);
  c.setObstacleBlocked(false); arm(c,12001); send(c,"D,65535,1234,35,35",12002); assert(c.armed && c.left==35);
  std::cout<<"PASS: actual firmware core CRC, arming, wheel limits, stale/duplicate rejection, watchdog, time wrap, partial/oversize input, stop\n";
}
