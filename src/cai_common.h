/* cai_common.h - internal helpers shared by the library translation units.
 * Not installed as part of the public API. */
#ifndef CAI_COMMON_H
#define CAI_COMMON_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/* ---- logging ---------------------------------------------------------- */

typedef enum { CAI_LOG_ERROR = 0, CAI_LOG_WARN, CAI_LOG_INFO, CAI_LOG_DEBUG } cai_log_level_t;

void cai_log_set_level(cai_log_level_t level);
void cai_logf(cai_log_level_t level, const char *fmt, ...);

#define CAI_ERROR(...) cai_logf(CAI_LOG_ERROR, __VA_ARGS__)
#define CAI_WARN(...) cai_logf(CAI_LOG_WARN, __VA_ARGS__)
#define CAI_INFO(...) cai_logf(CAI_LOG_INFO, __VA_ARGS__)
#define CAI_DEBUG(...) cai_logf(CAI_LOG_DEBUG, __VA_ARGS__)

/* ---- hashing ---------------------------------------------------------- */

/* FNV-1a 64. Used for plan_hash / topology_hash so a rank can refuse a plan
 * that was compiled against a different topology. */
uint64_t cai_fnv1a(const void *data, size_t len, uint64_t seed);
#define CAI_FNV_INIT 1469598103934665603ull

/* ---- raw POD file IO -------------------------------------------------- */

/* Read/write a whole struct or array of structs. The on-disk layout is just the
 * in-memory layout (host endian); all serialized structs are _Static_assert'd to
 * fixed sizes with no padding so this is stable for a homogeneous cluster. */
int cai_write_all(FILE *f, const void *buf, size_t nbytes);
int cai_read_all(FILE *f, void *buf, size_t nbytes);

/* Convenience: allocate-and-read `count` records of `elem` bytes. */
void *cai_read_array(FILE *f, size_t count, size_t elem);

/* ---- human formatting (for CLI reports) ------------------------------ */

const char *cai_fmt_bytes(uint64_t bytes, char *buf, size_t n);
const char *cai_fmt_count(double v, char *buf, size_t n);

/* ---- misc ------------------------------------------------------------- */

static inline uint64_t cai_align_up(uint64_t v, uint64_t a) {
    return (v + a - 1) & ~(a - 1);
}

#define CAI_ARENA_ALIGN 512u /* keep segments aligned for DMA-friendly access */

/* ---- static arena layout --------------------------------------------- */

#include "cai_tensor.h"

/* The runtime reserves one contiguous arena per GPU and never calls malloc on
 * the hot path. Tensors carry class-local offsets; the planner packs the per-class
 * segments back to back and then rebases each tensor into the global arena. */
typedef struct {
    uint64_t seg_off[CAI_TCLASS_COUNT];
    uint64_t seg_len[CAI_TCLASS_COUNT];
    uint64_t total;
} cai_arena_t;

void cai_arena_layout(cai_arena_t *a, const uint64_t want[CAI_TCLASS_COUNT]);
void cai_arena_relocate(const cai_arena_t *a, cai_tensor_desc_t *tensors,
                        uint32_t n);

#endif /* CAI_COMMON_H */
