/* topology_linter - sanity-check a config before compiling 220k plans.
 *
 *   topology_linter <config>
 *
 * Catches the failures that are catastrophic at scale: indivisible world size,
 * tensor-parallel groups that spill across NVLink domains, oversized per-GPU
 * memory, and bad rank<->location math. */
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cai.h"
#include "cai_common.h"
#include "cai_model.h"

static int g_problems;
static void problem(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "  [FAIL] ");
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
    g_problems++;
}
static void warn(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "  [warn] ");
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <config>\n", argv[0]);
        return 2;
    }
    cai_config_t cfg;
    char err[160];
    int rc = cai_config_load(argv[1], &cfg, err, sizeof(err));
    if (rc != CAI_OK) {
        fprintf(stderr, "config error: %s\n", err);
        return 1;
    }
    if (cai_topology_finalize(&cfg.topo, err, sizeof(err)) != CAI_OK) {
        fprintf(stderr, "[FAIL] topology: %s\n", err);
        return 1;
    }

    cai_decomp_t dec;
    rc = cai_decompose(&cfg, &dec, err, sizeof(err));
    if (rc != CAI_OK) {
        fprintf(stderr, "[FAIL] decompose: %s\n", err);
        return 1;
    }

    printf("=== decomposition ===\n");
    int fits = cai_report_decomp(stdout, &cfg, &dec);
    printf("=====================\n");

    cai_topology_t topo = cfg.topo;
    cai_topology_finalize(&topo, err, sizeof(err));

    printf("checks:\n");
    if (!fits) problem("per-GPU memory over budget");

    /* TP must stay inside a single NVLink domain (a rack). */
    if (dec.tp > topo.gpus_per_rack)
        problem("TP=%u exceeds gpus_per_rack=%u (spans NVLink domains)", dec.tp,
                topo.gpus_per_rack);
    else if (topo.gpus_per_rack % dec.tp != 0)
        warn("TP=%u does not divide gpus_per_rack=%u; some TP groups cross racks",
             dec.tp, topo.gpus_per_rack);
    else
        printf("  [ok]   TP=%u fits intra-rack (%u groups/rack)\n", dec.tp,
               topo.gpus_per_rack / dec.tp);

    /* rank<->location round trip on a few representative ranks */
    uint32_t probes[] = {0, topo.gpus_per_rack - 1, topo.gpus_per_rack,
                         topo.world_size - 1};
    for (size_t i = 0; i < sizeof(probes) / sizeof(probes[0]); i++) {
        uint32_t r = probes[i];
        if (r >= topo.world_size) continue;
        cai_phys_id_t p = cai_phys_of_rank(&topo, r);
        uint32_t back = cai_rank_of(&topo, p.rack_id, p.tray_id, p.gpu_in_tray);
        if (back != r)
            problem("rank %u maps to (rack %u,tray %u,gpu %u) but back to %u", r,
                    p.rack_id, p.tray_id, p.gpu_in_tray, back);
        else
            printf("  [ok]   rank %u -> rack %u tray %u gpu %u (nic %u rail %u)\n",
                   r, p.rack_id, p.tray_id, p.gpu_in_tray, p.nic_id, p.rail_id);
    }

    if (dec.bubble_ratio > 0.25)
        warn("pipeline bubble %.1f%% is high; raise grad_accum or lower PP",
             dec.bubble_ratio * 100.0);

    printf(g_problems ? "\nRESULT: %d problem(s)\n" : "\nRESULT: all checks passed\n",
           g_problems);
    return g_problems ? 1 : 0;
}
