/* plan_recover - roadmap M5 fault tolerance: hot-swap a failed rank onto a spare.
 *
 *   plan_recover <config> <failed_rank> <outdir>
 *
 * At 220k GPUs, individual failures are routine. A rank's plan depends only on its
 * logical coordinate, so recovery is: pick a spare physical GPU (preferably in a
 * different rack than the failure), regenerate the *identical* plan, and resume
 * that rank there from the latest checkpoint. This tool emits the replacement plan
 * and the placement decision. No GPU needed. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cai.h"
#include "cai_model.h"

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <config> <failed_rank> <outdir>\n", argv[0]);
        return 2;
    }
    uint32_t failed = (uint32_t)strtoul(argv[2], NULL, 10);
    const char *outdir = argv[3];

    cai_config_t cfg;
    char err[160];
    if (cai_config_load(argv[1], &cfg, err, sizeof(err)) != CAI_OK) {
        fprintf(stderr, "config error: %s\n", err);
        return 1;
    }
    cai_topology_t topo = cfg.topo;
    if (cai_topology_finalize(&topo, err, sizeof(err)) != CAI_OK) {
        fprintf(stderr, "topology: %s\n", err);
        return 1;
    }
    cfg.topo = topo;
    cai_decomp_t dec;
    if (cai_decompose(&cfg, &dec, err, sizeof(err)) != CAI_OK) {
        fprintf(stderr, "decompose: %s\n", err);
        return 1;
    }
    if (failed >= topo.world_size) {
        fprintf(stderr, "failed_rank %u >= world_size %u\n", failed, topo.world_size);
        return 1;
    }

    uint32_t spare;
    int rc = cai_spare_rank(&topo, failed, &spare);
    if (rc != CAI_OK) {
        fprintf(stderr,
                "no spare GPUs available (set spare_gpus > 0 in the config so "
                "failures can be absorbed)\n");
        return 1;
    }

    cai_phys_id_t fp = cai_phys_of_rank(&topo, failed);
    cai_phys_id_t sp = cai_phys_of_rank(&topo, spare);
    uint32_t stage = cai_rank_stage(&dec, failed);

    uint64_t ph = cai_plan_hash(&cfg, &dec), th = cai_topology_hash(&topo);
    cai_plan_t plan;
    rc = cai_build_plan(&cfg, &dec, failed, ph, th, &plan, err, sizeof(err));
    if (rc != CAI_OK) {
        fprintf(stderr, "build: %s\n", err);
        return 1;
    }

#ifdef _WIN32
#else
    {
        char cmd[600];
        snprintf(cmd, sizeof(cmd), "mkdir -p '%s'", outdir);
        if (system(cmd) != 0) { /* best effort */ }
    }
#endif
    char path[600];
    snprintf(path, sizeof(path), "%s/plan.recover.rank%06u.bin", outdir, failed);
    rc = cai_plan_write(path, &plan);
    cai_plan_free(&plan);
    if (rc != CAI_OK) {
        fprintf(stderr, "write %s failed\n", path);
        return 1;
    }

    printf("recovery plan for rank %u:\n", failed);
    printf("  failed rank %u  was on rack %u tray %u gpu %u (stage %u)\n", failed,
           fp.rack_id, fp.tray_id, fp.gpu_in_tray, stage);
    printf("  spare phys  %u  -> rack %u tray %u gpu %u  [%s rack]\n", spare,
           sp.rack_id, sp.tray_id, sp.gpu_in_tray,
           sp.rack_id == fp.rack_id ? "SAME" : "different");
    printf("  wrote %s\n", path);
    printf("  resume: load this plan + topology.bin on the spare and "
           "restore the latest checkpoint for rank %u\n", failed);
    return 0;
}
