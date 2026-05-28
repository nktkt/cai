/* cai_tensor.h - tensor descriptors, dtypes, classes.
 *
 * Everything here is plain POD with a fixed binary layout so that the offline
 * plan compiler and the runtime agree byte-for-byte on serialized plans. */
#ifndef CAI_TENSOR_H
#define CAI_TENSOR_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
    CAI_DT_FP32 = 0,
    CAI_DT_BF16 = 1,
    CAI_DT_FP16 = 2,
    CAI_DT_FP8_E4M3 = 3,
    CAI_DT_FP8_E5M2 = 4,
    CAI_DT_INT32 = 5,
    CAI_DT_UINT8 = 6,
    CAI_DT_COUNT
} cai_dtype_t;

/* Size in bytes of one element of the given dtype. */
size_t cai_dtype_size(uint16_t dtype);
const char *cai_dtype_name(uint16_t dtype);

/* Where a tensor lives in the static arena. Lifetime/class drives both the
 * arena segment it is packed into and how the runtime accounts memory. */
typedef enum {
    CAI_TCLASS_PARAM = 0,      /* sharded model weights (resident) */
    CAI_TCLASS_GRAD = 1,       /* sharded gradients (resident) */
    CAI_TCLASS_OPTSTATE = 2,   /* sharded optimizer state (resident) */
    CAI_TCLASS_ACTIVATION = 3, /* transient, lives in the activation ring */
    CAI_TCLASS_COMM = 4,       /* pipeline / collective staging buffers */
    CAI_TCLASS_WORKSPACE = 5,  /* attention / GEMM scratch (reusable) */
    CAI_TCLASS_COUNT
} cai_tensor_class_t;

const char *cai_tclass_name(uint16_t tclass);

/* 24 bytes, 8-byte aligned, no padding. */
typedef struct {
    uint64_t offset; /* byte offset inside the arena segment for its class */
    uint64_t nbytes; /* size after sharding */
    uint32_t shape_id;
    uint16_t dtype;  /* cai_dtype_t */
    uint16_t tclass; /* cai_tensor_class_t */
} cai_tensor_desc_t;

_Static_assert(sizeof(cai_tensor_desc_t) == 24, "cai_tensor_desc_t layout");

#endif /* CAI_TENSOR_H */
