#include "cai_common.h"

void cai_arena_layout(cai_arena_t *a, const uint64_t want[CAI_TCLASS_COUNT]) {
    uint64_t cursor = 0;
    for (int c = 0; c < CAI_TCLASS_COUNT; c++) {
        cursor = cai_align_up(cursor, CAI_ARENA_ALIGN);
        a->seg_off[c] = cursor;
        a->seg_len[c] = want[c];
        cursor += want[c];
    }
    a->total = cai_align_up(cursor, CAI_ARENA_ALIGN);
}

void cai_arena_relocate(const cai_arena_t *a, cai_tensor_desc_t *tensors,
                        uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        uint16_t c = tensors[i].tclass;
        if (c < CAI_TCLASS_COUNT) tensors[i].offset += a->seg_off[c];
    }
}
