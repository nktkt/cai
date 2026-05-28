#include "cai_topology.h"

#include <stdio.h>
#include <string.h>

#include "cai.h"
#include "cai_common.h"

#define CAI_TOPO_MAGIC 0x504F5443u /* 'CTOP' */

void cai_topology_default(cai_topology_t *t) {
    memset(t, 0, sizeof(*t));
    t->gpus_per_rack = CAI_GPUS_PER_RACK;
    t->trays_per_rack = CAI_TRAYS_PER_RACK;
    t->gpus_per_tray = CAI_GPUS_PER_TRAY;
    t->num_racks = 1;
    t->spare_gpus = 0;
    /* GB300 NVL72 ballpark figures (per GPU unless noted). */
    t->intra_rack_bw = 1800ull * 1000 * 1000 * 1000;  /* ~1.8 TB/s NVLink5 */
    t->inter_rack_bw = 100ull * 1000 * 1000 * 1000;   /* 800 Gb/s ConnectX-8 */
    t->hbm_bw = 8000ull * 1000 * 1000 * 1000;         /* ~8 TB/s HBM3e */
    t->gpu_flops_bf16 = 2.5e15;                        /* sustainable dense BF16 */
    t->gpu_mem_bytes = 288ull * 1000 * 1000 * 1000;   /* ~288 GB */
}

int cai_topology_finalize(cai_topology_t *t, char *err, size_t errlen) {
    if (t->gpus_per_rack == 0 || t->num_racks == 0) {
        snprintf(err, errlen, "gpus_per_rack and num_racks must be > 0");
        return CAI_ERR_TOPOLOGY;
    }
    if (t->gpus_per_tray == 0) t->gpus_per_tray = CAI_GPUS_PER_TRAY;
    if (t->trays_per_rack == 0)
        t->trays_per_rack = t->gpus_per_rack / t->gpus_per_tray;
    if (t->trays_per_rack * t->gpus_per_tray != t->gpus_per_rack) {
        snprintf(err, errlen,
                 "trays_per_rack(%u)*gpus_per_tray(%u) != gpus_per_rack(%u)",
                 t->trays_per_rack, t->gpus_per_tray, t->gpus_per_rack);
        return CAI_ERR_TOPOLOGY;
    }
    uint64_t total = (uint64_t)t->gpus_per_rack * t->num_racks;
    if (t->spare_gpus >= total) {
        snprintf(err, errlen, "spare_gpus(%u) >= total gpus(%llu)", t->spare_gpus,
                 (unsigned long long)total);
        return CAI_ERR_TOPOLOGY;
    }
    uint64_t usable = total - t->spare_gpus;
    if (usable > 0xFFFFFFFFull) {
        snprintf(err, errlen, "world_size %llu exceeds 32-bit rank space",
                 (unsigned long long)usable);
        return CAI_ERR_TOPOLOGY;
    }
    t->world_size = (uint32_t)usable;
    if (t->gpu_flops_bf16 <= 0 || t->hbm_bw == 0 || t->intra_rack_bw == 0 ||
        t->inter_rack_bw == 0) {
        snprintf(err, errlen, "bandwidth/flops fields must be > 0");
        return CAI_ERR_TOPOLOGY;
    }
    return CAI_OK;
}

cai_phys_id_t cai_phys_of_rank(const cai_topology_t *t, uint32_t rank) {
    cai_phys_id_t p;
    memset(&p, 0, sizeof(p));
    p.global_rank = rank;
    uint32_t gpr = t->gpus_per_rack;
    uint32_t gpt = t->gpus_per_tray;
    p.rack_id = (uint16_t)(rank / gpr);
    p.gpu_in_rack = (uint8_t)(rank % gpr);
    p.tray_id = (uint8_t)(p.gpu_in_rack / gpt);
    p.gpu_in_tray = (uint8_t)(p.gpu_in_rack % gpt);
    p.nic_id = p.gpu_in_rack;             /* 1 NIC per GPU */
    p.rail_id = (uint8_t)(p.gpu_in_rack % gpt); /* one rail per tray slot */
    return p;
}

uint32_t cai_rank_of(const cai_topology_t *t, uint16_t rack, uint8_t tray,
                     uint8_t gpu) {
    return (uint32_t)rack * t->gpus_per_rack + (uint32_t)tray * t->gpus_per_tray +
           gpu;
}

int cai_same_rack(const cai_topology_t *t, uint32_t a, uint32_t b) {
    return a / t->gpus_per_rack == b / t->gpus_per_rack;
}

int cai_spare_rank(const cai_topology_t *t, uint32_t failed_rank, uint32_t *out) {
    uint64_t total = (uint64_t)t->gpus_per_rack * t->num_racks;
    if (t->world_size >= total) return CAI_ERR_INVALID; /* no spares */
    if (failed_rank >= t->world_size) return CAI_ERR_INVALID;
    uint32_t frack = failed_rank / t->gpus_per_rack;
    for (uint32_t s = t->world_size; s < total; s++)
        if (s / t->gpus_per_rack != frack) {
            *out = s;
            return CAI_OK;
        }
    *out = t->world_size; /* all spares share the failed rack; take the first */
    return CAI_OK;
}

int cai_topology_write(const char *path, const cai_topology_t *t) {
    FILE *f = fopen(path, "wb");
    if (!f) return CAI_ERR_IO;
    uint32_t magic = CAI_TOPO_MAGIC;
    int rc = cai_write_all(f, &magic, sizeof(magic));
    if (rc == CAI_OK) rc = cai_write_all(f, t, sizeof(*t));
    fclose(f);
    return rc;
}

int cai_topology_read(const char *path, cai_topology_t *t) {
    FILE *f = fopen(path, "rb");
    if (!f) return CAI_ERR_IO;
    uint32_t magic = 0;
    int rc = cai_read_all(f, &magic, sizeof(magic));
    if (rc == CAI_OK && magic != CAI_TOPO_MAGIC) rc = CAI_ERR_TOPOLOGY;
    if (rc == CAI_OK) rc = cai_read_all(f, t, sizeof(*t));
    fclose(f);
    return rc;
}
