/* cai_topology.h - physical cluster model and rank<->location mapping.
 *
 * Target hardware: NVIDIA GB300 NVL72. One rack is a single NVLink/NVSwitch
 * domain of 72 Blackwell-Ultra GPUs across 18 compute trays (4 GPU/tray), with
 * one ConnectX-8 SuperNIC (~800 Gb/s) per GPU on the rack-to-rack fabric.
 *
 * The runtime never *discovers* topology; the offline compiler bakes a resolved
 * topology into topology.bin and every rank loads the same file. */
#ifndef CAI_TOPOLOGY_H
#define CAI_TOPOLOGY_H

#include <stddef.h>
#include <stdint.h>

#define CAI_GPUS_PER_RACK 72u
#define CAI_TRAYS_PER_RACK 18u
#define CAI_GPUS_PER_TRAY 4u

/* Decoded physical location of a global rank. 12 bytes, no padding. */
typedef struct {
    uint32_t global_rank; /* 0 .. world_size-1 */
    uint16_t rack_id;
    uint8_t tray_id;     /* 0 .. trays_per_rack-1 */
    uint8_t gpu_in_tray; /* 0 .. gpus_per_tray-1 */
    uint8_t gpu_in_rack; /* 0 .. gpus_per_rack-1 */
    uint8_t nic_id;      /* 1:1 with GPU in V1 */
    uint8_t rail_id;     /* physical network rail this NIC lands on */
    uint8_t reserved;
} cai_phys_id_t;

_Static_assert(sizeof(cai_phys_id_t) == 12, "cai_phys_id_t layout");

/* Resolved, serialized cluster description. 64 bytes. */
typedef struct {
    uint32_t gpus_per_rack;
    uint32_t trays_per_rack;
    uint32_t gpus_per_tray;
    uint32_t num_racks;
    uint32_t world_size; /* usable ranks = gpus_per_rack*num_racks - spare_gpus */
    uint32_t spare_gpus;
    uint64_t intra_rack_bw;  /* per-GPU NVLink bidir bytes/s inside a rack */
    uint64_t inter_rack_bw;  /* per-GPU NIC bytes/s across racks */
    uint64_t hbm_bw;         /* per-GPU HBM bytes/s */
    double gpu_flops_bf16;   /* per-GPU sustainable dense BF16 FLOP/s */
    uint64_t gpu_mem_bytes;  /* per-GPU HBM capacity */
} cai_topology_t;

_Static_assert(sizeof(cai_topology_t) == 64, "cai_topology_t layout");

/* Sensible GB300 NVL72 defaults; the compiler may override from config. */
void cai_topology_default(cai_topology_t *t);

/* Fill in derived fields (world_size) and validate the rest. Returns CAI_OK or
 * a negative cai_status_t; on error writes a human message into err. */
int cai_topology_finalize(cai_topology_t *t, char *err, size_t errlen);

cai_phys_id_t cai_phys_of_rank(const cai_topology_t *t, uint32_t rank);
uint32_t cai_rank_of(const cai_topology_t *t, uint16_t rack, uint8_t tray,
                     uint8_t gpu);

/* Two ranks share an NVLink domain (intra-rack, fat) vs. cross-rack (thin). */
int cai_same_rack(const cai_topology_t *t, uint32_t a, uint32_t b);

int cai_topology_write(const char *path, const cai_topology_t *t);
int cai_topology_read(const char *path, cai_topology_t *t);

/* Pick a spare physical GPU (in [world_size, gpus_per_rack*num_racks)) to take
 * over a failed rank, preferring a different rack so a rack failure can't claim
 * both. Returns CAI_OK and writes the spare into out, or an error if no spare. */
int cai_spare_rank(const cai_topology_t *t, uint32_t failed_rank, uint32_t *out);

#endif /* CAI_TOPOLOGY_H */
