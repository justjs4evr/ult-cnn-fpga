#include <Arduino.h>
#include "esp_camera.h"
#include "ult_preprocess.h"
#include <Preferences.h>

// Camera mapping from the user's existing AP-camera sketch and official Sense
// example. Header UART uses D6/TX GPIO43 and D7/RX GPIO44; no motors are driven.
static constexpr size_t RAW_BYTES=160*120, PIXEL_BYTES=4352;
static uint8_t payload[PIXEL_BYTES];
static uint8_t test_raw[RAW_BYTES];
static uint32_t sequence=0;
static bool camera_ready=false;
static esp_err_t camera_error=ESP_FAIL;
static uint16_t sensor_pid=0;
static Preferences direct_config;
static bool auto_mode=false;
static uint32_t successful_frames=0;

static uint32_t crc_step(uint32_t crc,const uint8_t *data,size_t n) {
    for(size_t i=0;i<n;++i) {
        crc^=data[i];
        for(int b=0;b<8;++b) crc=(crc>>1)^((crc&1)?0xedb88320U:0U);
    }
    return crc;
}
static void put16(uint8_t *p,uint16_t v) {p[0]=v;p[1]=v>>8;}
static void put32(uint8_t *p,uint32_t v) {for(int i=0;i<4;++i)p[i]=v>>(8*i);}
static bool read_exact(uint8_t *p,size_t n,uint32_t timeout_ms=5000) {
    size_t done=0;uint32_t last=millis();
    while(done<n && uint32_t(millis()-last)<timeout_ms) {
        int available=Serial.available();
        if(available>0) {
            size_t count=Serial.readBytes(p+done,min(size_t(available),n-done));
            done+=count;last=millis();
        } else delay(1);
    }
    return done==n;
}
static void write_exact(const uint8_t *p,size_t n) {
    size_t done=0;uint32_t started=millis();
    while(done<n && uint32_t(millis()-started)<5000) {
        size_t count=Serial.write(p+done,min(size_t(512),n-done));
        done+=count;if(!count)delay(1);
    }
}
static void info() {
    Serial.printf("ULTCAM_INFO {\"protocol\":1,\"width\":160,\"height\":120,\"format\":\"GRAY8\",\"camera_ready\":%s,\"camera_error\":%d,\"sensor_pid\":%u,\"psram_bytes\":%u,\"vflip\":1,\"external_fpga_link_enabled\":true}\n",
        camera_ready?"true":"false",int(camera_error),unsigned(sensor_pid),unsigned(ESP.getPsramSize()));
}
static void send_frame(const uint8_t *raw,uint8_t kind,uint32_t captured_ms) {
    uint32_t started=micros();ult_prepare(raw,payload);uint32_t preparation_us=micros()-started;
    uint8_t header[40]={};
    memcpy(header,"ULTCAM01",8);header[8]=1;header[9]=kind;
    put16(header+10,160);put16(header+12,120);put16(header+14,1);
    put32(header+16,++sequence);put32(header+20,captured_ms);put32(header+24,preparation_us);
    put32(header+28,RAW_BYTES);put32(header+32,PIXEL_BYTES);put32(header+36,0);
    uint32_t crc=crc_step(0xffffffffU,header,sizeof(header));
    crc=crc_step(crc,raw,RAW_BYTES);crc=crc_step(crc,payload,PIXEL_BYTES)^0xffffffffU;
    uint8_t tail[4];put32(tail,crc);
    write_exact(header,sizeof(header));write_exact(raw,RAW_BYTES);write_exact(payload,PIXEL_BYTES);write_exact(tail,4);
    Serial.flush();
}
void setup() {
    Serial.begin(115200);Serial.setDebugOutput(false);Serial.setTimeout(1000);delay(1000);
    Serial1.setRxBufferSize(256);
    Serial1.begin(460800,SERIAL_8N1,44,43);
    direct_config.begin("ult_direct",false);
    auto_mode=direct_config.getBool("auto",false);
    camera_config_t c={};
    c.ledc_channel=LEDC_CHANNEL_0;c.ledc_timer=LEDC_TIMER_0;
    c.pin_d0=15;c.pin_d1=17;c.pin_d2=18;c.pin_d3=16;
    c.pin_d4=14;c.pin_d5=12;c.pin_d6=11;c.pin_d7=48;
    c.pin_xclk=10;c.pin_pclk=13;c.pin_vsync=38;c.pin_href=47;
    c.pin_sccb_sda=40;c.pin_sccb_scl=39;c.pin_pwdn=-1;c.pin_reset=-1;
    c.xclk_freq_hz=20000000;c.pixel_format=PIXFORMAT_GRAYSCALE;
    c.frame_size=FRAMESIZE_QQVGA;c.jpeg_quality=12;c.fb_count=1;
    c.fb_location=CAMERA_FB_IN_PSRAM;c.grab_mode=CAMERA_GRAB_WHEN_EMPTY;
    if(psramFound()) {
        camera_error=esp_camera_init(&c);
        if(camera_error==ESP_OK) {
            sensor_t *s=esp_camera_sensor_get();sensor_pid=s->id.PID;
            s->set_vflip(s,1); // Same orientation correction as the existing sketch.
            camera_ready=true;
            for(int i=0;i<12;++i) {camera_fb_t *fb=esp_camera_fb_get();if(fb)esp_camera_fb_return(fb);delay(10);}
        }
    }
    info();
}
static uint16_t crc16_step(uint16_t crc,const uint8_t *data,size_t n) {
    for(size_t i=0;i<n;++i) {
        crc^=uint16_t(data[i])<<8;
        for(int j=0;j<8;++j)crc=(crc&0x8000)?uint16_t((crc<<1)^0x1021):uint16_t(crc<<1);
    }
    return crc;
}
static uint32_t get32(const uint8_t *p) {
    return uint32_t(p[0])|(uint32_t(p[1])<<8)|(uint32_t(p[2])<<16)|(uint32_t(p[3])<<24);
}
static bool read_result(uint8_t *result,uint32_t frame,uint32_t tick) {
    size_t index=0;uint32_t start=millis();
    while(uint32_t(millis()-start)<600) {
        if(!Serial1.available()) {delay(1);continue;}
        uint8_t value=Serial1.read();
        if(index==0) {if(value==0xaa)result[index++]=value;continue;}
        if(index==1 && value!=0x55) {index=(value==0xaa)?1:0;continue;}
        result[index++]=value;
        if(index==36) {
            uint16_t check=uint16_t(result[34])|(uint16_t(result[35])<<8);
            bool ok=result[2]==2 && result[3]==1 && crc16_step(0xffff,result,34)==check
                && get32(result+4)==frame && get32(result+8)==tick
                && get32(result+16)==0x26c8ea39 && result[20]<6
                && (result[21]&0x0f)==1 && result[22]==1 && result[23]==2;
            if(ok)return true;
            index=0;
        }
    }
    return false;
}
static void direct_frame(bool audit) {
    if(!camera_ready) {if(audit)info();else delay(100);return;}
    camera_fb_t *old=esp_camera_fb_get();if(old)esp_camera_fb_return(old);
    camera_fb_t *fb=esp_camera_fb_get();
    if(!fb || fb->width!=160 || fb->height!=120 || fb->format!=PIXFORMAT_GRAYSCALE || fb->len<RAW_BYTES) {
        if(fb)esp_camera_fb_return(fb);
        Serial.println("ULTCAM_ERROR capture_failed");return;
    }
    uint32_t tick=millis(),frame=++sequence;
    uint32_t started=micros();ult_prepare(fb->buf,payload);uint32_t prep=micros()-started;
    uint8_t header[16]={0xa5,0x5a,1,0x10};
    put32(header+4,frame);put32(header+8,tick);put16(header+12,PIXEL_BYTES);put16(header+14,0x0201);
    uint16_t crc=crc16_step(0xffff,header,16);crc=crc16_step(crc,payload,PIXEL_BYTES);
    uint8_t crc_bytes[2];put16(crc_bytes,crc);
    while(Serial1.available())Serial1.read();
    started=micros();
    Serial1.write(header,16);Serial1.write(payload,PIXEL_BYTES);Serial1.write(crc_bytes,2);Serial1.flush();
    uint8_t result[36]={};bool ok=read_result(result,frame,tick);uint32_t rtt=micros()-started;
    if(ok)++successful_frames;
    if(audit) {
        uint8_t response[48]={};memcpy(response,"ULTDIR01",8);response[8]=1;response[9]=3;
        put16(response+10,160);put16(response+12,120);put16(response+14,1);
        put32(response+16,frame);put32(response+20,tick);put32(response+24,prep);
        put32(response+28,RAW_BYTES);put32(response+32,PIXEL_BYTES);put32(response+36,ok?36:0);
        put32(response+40,rtt);put32(response+44,ok?0:1);
        uint32_t crc32=crc_step(0xffffffff,response,48);
        crc32=crc_step(crc32,fb->buf,RAW_BYTES);crc32=crc_step(crc32,payload,PIXEL_BYTES);
        if(ok)crc32=crc_step(crc32,result,36);crc32^=0xffffffff;
        uint8_t tail[4];put32(tail,crc32);
        write_exact(response,48);write_exact(fb->buf,RAW_BYTES);write_exact(payload,PIXEL_BYTES);
        if(ok)write_exact(result,36);write_exact(tail,4);Serial.flush();
    }
    // Autonomous transfer never waits for a PC to consume XIAO USB logs.
    // Results remain observable on the FPGA's independent USB mirror.
    esp_camera_fb_return(fb);
}
void loop() {
    if(!Serial.available()) {if(auto_mode)direct_frame(false);delay(1);return;}
    int command=Serial.read();
    if(command=='I') {info();return;}
    if(command=='G') {auto_mode=true;direct_config.putBool("auto",true);Serial.println("ULT_DIRECT auto=1 saved");return;}
    if(command=='S') {auto_mode=false;direct_config.putBool("auto",false);Serial.println("ULT_DIRECT auto=0 saved");return;}
    if(command=='D') {direct_frame(true);return;}
    if(command=='T') {
        uint8_t tail[4];
        if(!read_exact(test_raw,RAW_BYTES)||!read_exact(tail,4)) {Serial.println("ULTCAM_ERROR input_timeout");return;}
        uint32_t expected=uint32_t(tail[0])|(uint32_t(tail[1])<<8)|(uint32_t(tail[2])<<16)|(uint32_t(tail[3])<<24);
        if((crc_step(0xffffffffU,test_raw,RAW_BYTES)^0xffffffffU)!=expected) {Serial.println("ULTCAM_ERROR input_crc");return;}
        send_frame(test_raw,2,millis());return;
    }
    if(command=='C') {
        if(!camera_ready) {info();return;}
        // Discard the pending frame so the requested capture is fresh.
        camera_fb_t *old=esp_camera_fb_get();if(old)esp_camera_fb_return(old);
        camera_fb_t *fb=esp_camera_fb_get();
        if(!fb) {Serial.println("ULTCAM_ERROR capture_failed");return;}
        if(fb->width!=160 || fb->height!=120 || fb->format!=PIXFORMAT_GRAYSCALE || fb->len<RAW_BYTES) {
            Serial.println("ULTCAM_ERROR unexpected_frame_format");esp_camera_fb_return(fb);return;
        }
        send_frame(fb->buf,1,millis());esp_camera_fb_return(fb);return;
    }
}
