#include "cai_model.h"

double cai_pipeline_bubble(uint32_t pp, uint32_t m) {
    if (pp <= 1 || m == 0) return 0.0;
    return (double)(pp - 1) / (double)(m + pp - 1);
}

uint32_t cai_pipeline_schedule_1f1b(uint32_t stage, uint32_t pp, uint32_t m,
                                    cai_tick_t *buf) {
    uint32_t n = 0;
    if (m == 0) return 0;

    /* Stage s starts (pp-1-s) microbatches ahead so the last stage runs 1F1B
     * immediately while earlier stages stay one step "fuller". */
    uint32_t warmup = (pp > 0 && stage < pp) ? (pp - 1 - stage) : 0;
    if (warmup > m) warmup = m;

    uint32_t fwd = 0, bwd = 0;

    for (uint32_t i = 0; i < warmup; i++) {
        buf[n].kind = CAI_TICK_FWD;
        buf[n].microbatch = (uint16_t)fwd++;
        n++;
    }
    /* steady: one forward then one backward, until forwards are exhausted */
    while (fwd < m) {
        buf[n].kind = CAI_TICK_FWD;
        buf[n].microbatch = (uint16_t)fwd++;
        n++;
        buf[n].kind = CAI_TICK_BWD;
        buf[n].microbatch = (uint16_t)bwd++;
        n++;
    }
    /* drain: remaining backwards */
    while (bwd < m) {
        buf[n].kind = CAI_TICK_BWD;
        buf[n].microbatch = (uint16_t)bwd++;
        n++;
    }
    return n; /* == 2*m */
}
