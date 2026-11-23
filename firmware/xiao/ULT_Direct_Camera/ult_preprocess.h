#pragma once
#include <stdint.h>
#include <stddef.h>

// The deployed model expects PIL L/BOX rounding, including a rounded horizontal
// intermediate. An area average over the whole 2-D rectangle is not equivalent.
// This fixed geometry mirrors the BOX pixel-center selection and Q22 weights.
static inline void ult_box_gray(const uint8_t *src, int iw, int ih,
                                uint8_t *dst, int ow, int oh, uint8_t *tmp) {
    const uint32_t one = 1U << 22;
    for (int y=0; y<ih; ++y) {
        for (int x=0; x<ow; ++x) {
            int lo=(2*x*iw+ow)/(2*ow);
            int hi=(2*(x+1)*iw+ow)/(2*ow);
            if (hi>iw) hi=iw;
            int n=hi-lo;
            uint32_t k=(one+uint32_t(n/2))/uint32_t(n);
            uint32_t sum=one/2;
            for (int j=lo; j<hi; ++j) sum+=uint32_t(src[y*iw+j])*k;
            uint32_t value=sum>>22;
            tmp[y*ow+x]=uint8_t(value>255?255:value);
        }
    }
    for (int y=0; y<oh; ++y) {
        int lo=(2*y*ih+oh)/(2*oh);
        int hi=(2*(y+1)*ih+oh)/(2*oh);
        if (hi>ih) hi=ih;
        int n=hi-lo;
        uint32_t k=(one+uint32_t(n/2))/uint32_t(n);
        for (int x=0; x<ow; ++x) {
            uint32_t sum=one/2;
            for (int j=lo; j<hi; ++j) sum+=uint32_t(tmp[j*ow+x])*k;
            uint32_t value=sum>>22;
            dst[y*ow+x]=uint8_t(value>255?255:value);
        }
    }
}

// raw: exactly 160x120 GRAY8, payload: legacy16[256], then full context64[4096].
// No CNN inference is performed by the XIAO.
static inline void ult_prepare(const uint8_t *raw, uint8_t *payload) {
    static uint8_t tmp[64*120];
    static uint8_t medium[64*64];
    ult_box_gray(raw,160,120,payload+256,64,64,tmp);
    ult_box_gray(raw+30*160,160,90,medium,64,64,tmp);
    for (size_t i=0; i<sizeof(medium); ++i) medium[i]=(medium[i]>=127)?255:0;
    ult_box_gray(medium,64,64,payload,16,16,tmp);
}
