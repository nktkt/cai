#include "cai_common.h"

#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#include "cai.h"

static cai_log_level_t g_level = CAI_LOG_INFO;

void cai_log_set_level(cai_log_level_t level) { g_level = level; }

void cai_logf(cai_log_level_t level, const char *fmt, ...) {
    if (level > g_level) return;
    static const char *tag[] = {"ERROR", "WARN", "INFO", "DEBUG"};
    fprintf(stderr, "[cai %s] ", tag[level]);
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
}

uint64_t cai_fnv1a(const void *data, size_t len, uint64_t seed) {
    const unsigned char *p = (const unsigned char *)data;
    uint64_t h = seed;
    for (size_t i = 0; i < len; i++) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

int cai_write_all(FILE *f, const void *buf, size_t nbytes) {
    if (nbytes == 0) return CAI_OK;
    return fwrite(buf, 1, nbytes, f) == nbytes ? CAI_OK : CAI_ERR_IO;
}

int cai_read_all(FILE *f, void *buf, size_t nbytes) {
    if (nbytes == 0) return CAI_OK;
    return fread(buf, 1, nbytes, f) == nbytes ? CAI_OK : CAI_ERR_IO;
}

void *cai_read_array(FILE *f, size_t count, size_t elem) {
    if (count == 0) return NULL;
    void *buf = malloc(count * elem);
    if (!buf) return NULL;
    if (cai_read_all(f, buf, count * elem) != CAI_OK) {
        free(buf);
        return NULL;
    }
    return buf;
}

const char *cai_status_str(int status) {
    switch (status) {
        case CAI_OK: return "ok";
        case CAI_ERR_IO: return "io error";
        case CAI_ERR_INVALID: return "invalid argument";
        case CAI_ERR_OOM: return "out of memory";
        case CAI_ERR_TOPOLOGY: return "topology error";
        case CAI_ERR_PLAN: return "plan error";
        case CAI_ERR_BACKEND: return "backend error";
        case CAI_ERR_DEPENDENCY: return "dependency violation";
        case CAI_ERR_BUDGET: return "memory budget exceeded";
        default: return "unknown error";
    }
}

/* ---- small enum -> string tables ------------------------------------- */

size_t cai_dtype_size(uint16_t dtype) {
    switch (dtype) {
        case CAI_DT_FP32: return 4;
        case CAI_DT_BF16: return 2;
        case CAI_DT_FP16: return 2;
        case CAI_DT_FP8_E4M3: return 1;
        case CAI_DT_FP8_E5M2: return 1;
        case CAI_DT_INT32: return 4;
        case CAI_DT_UINT8: return 1;
        default: return 0;
    }
}

const char *cai_dtype_name(uint16_t dtype) {
    static const char *n[CAI_DT_COUNT] = {"fp32", "bf16",     "fp16",   "fp8e4m3",
                                          "fp8e5m2", "int32",  "uint8"};
    return dtype < CAI_DT_COUNT ? n[dtype] : "?";
}

const char *cai_tclass_name(uint16_t c) {
    static const char *n[CAI_TCLASS_COUNT] = {"param",      "grad", "optstate",
                                              "activation", "comm", "workspace"};
    return c < CAI_TCLASS_COUNT ? n[c] : "?";
}

const char *cai_stream_name(uint16_t s) {
    static const char *n[CAI_STREAM_COUNT] = {"compute_hi", "compute_lo", "comm_tp",
                                              "comm_pp",    "comm_dp",    "io"};
    return s < CAI_STREAM_COUNT ? n[s] : "?";
}

const char *cai_op_name(uint16_t k) {
    static const char *n[CAI_OP_COUNT] = {
        "gemm",          "attn_fwd",    "attn_bwd",      "rmsnorm",
        "swiglu",        "elementwise", "reduce_scatter", "all_gather",
        "all_reduce",    "all_to_all",  "pipe_send",     "pipe_recv",
        "optimizer",     "event_record", "event_wait"};
    return k < CAI_OP_COUNT ? n[k] : "?";
}

int cai_op_is_comm(uint16_t k) {
    switch (k) {
        case CAI_OP_REDUCE_SCATTER:
        case CAI_OP_ALL_GATHER:
        case CAI_OP_ALL_REDUCE:
        case CAI_OP_ALL_TO_ALL:
        case CAI_OP_PIPE_SEND:
        case CAI_OP_PIPE_RECV: return 1;
        default: return 0;
    }
}

const char *cai_fmt_bytes(uint64_t bytes, char *buf, size_t n) {
    const char *u[] = {"B", "KiB", "MiB", "GiB", "TiB", "PiB"};
    double v = (double)bytes;
    int i = 0;
    while (v >= 1024.0 && i < 5) {
        v /= 1024.0;
        i++;
    }
    snprintf(buf, n, "%.2f %s", v, u[i]);
    return buf;
}

const char *cai_fmt_count(double v, char *buf, size_t n) {
    const char *u[] = {"", "K", "M", "B", "T", "P", "E"};
    int i = 0;
    while (v >= 1000.0 && i < 6) {
        v /= 1000.0;
        i++;
    }
    if (v >= 1000.0)
        snprintf(buf, n, "%.2e", v * 1e18); /* beyond exa: scientific */
    else
        snprintf(buf, n, "%.2f%s", v, u[i]);
    return buf;
}

int cai_op_is_compute(uint16_t k) {
    switch (k) {
        case CAI_OP_GEMM:
        case CAI_OP_ATTENTION_FWD:
        case CAI_OP_ATTENTION_BWD:
        case CAI_OP_RMSNORM:
        case CAI_OP_SWIGLU:
        case CAI_OP_ELEMENTWISE:
        case CAI_OP_OPTIMIZER: return 1;
        default: return 0;
    }
}
