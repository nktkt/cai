/* plan_compiler - the offline brain of the system.
 *
 *   plan_compiler <config> <outdir> [--rank R | --ranks N | --all | --sample]
 *
 * Reads a model+parallelism+topology config, computes the cluster decomposition,
 * and emits topology.bin plus one plan.rankNNNNNN.bin per requested rank. The
 * runtime is dumb on purpose; everything that requires global knowledge happens
 * here, once. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "cai.h"
#include "cai_common.h"
#include "cai_model.h"

enum { MODE_SAMPLE, MODE_RANK, MODE_RANGE, MODE_ALL };

/* mkdir -p: create every component of `path`. */
static void mkdir_p(const char *path) {
    char tmp[600];
    snprintf(tmp, sizeof(tmp), "%s", path);
    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = 0;
            mkdir(tmp, 0755);
            *p = '/';
        }
    }
    mkdir(tmp, 0755);
}

static void usage(const char *p) {
    fprintf(stderr,
            "usage: %s <config> <outdir> [options]\n"
            "  --rank R     compile only global rank R\n"
            "  --ranks N    compile ranks [0, N)\n"
            "  --all        compile every rank (needs --force above 4096)\n"
            "  --sample     compile a representative handful (default)\n"
            "  --force      allow large --all\n"
            "  --quiet      suppress the decomposition report\n",
            p);
}

static int write_one(const cai_config_t *cfg, const cai_decomp_t *dec,
                     uint32_t rank, uint64_t ph, uint64_t th, const char *outdir) {
    cai_plan_t plan;
    char err[160], path[600];
    int rc = cai_build_plan(cfg, dec, rank, ph, th, &plan, err, sizeof(err));
    if (rc != CAI_OK) {
        fprintf(stderr, "rank %u: build failed: %s\n", rank, err);
        return rc;
    }
    snprintf(path, sizeof(path), "%s/plan.rank%06u.bin", outdir, rank);
    rc = cai_plan_write(path, &plan);
    if (rc != CAI_OK)
        fprintf(stderr, "rank %u: write failed: %s\n", rank, cai_status_str(rc));
    else
        printf("  wrote %s  (stage %u, ops=%u, arena=%.2f GiB)\n", path,
               plan.hdr.pipeline_stage, plan.hdr.num_ops,
               plan.hdr.arena_bytes / (1024.0 * 1024.0 * 1024.0));
    cai_plan_free(&plan);
    return rc;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        usage(argv[0]);
        return 2;
    }
    const char *cfgpath = argv[1];
    const char *outdir = argv[2];
    int mode = MODE_SAMPLE, force = 0, quiet = 0;
    uint32_t arg_rank = 0, arg_n = 0;

    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--rank") && i + 1 < argc) {
            mode = MODE_RANK;
            arg_rank = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--ranks") && i + 1 < argc) {
            mode = MODE_RANGE;
            arg_n = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--all")) {
            mode = MODE_ALL;
        } else if (!strcmp(argv[i], "--sample")) {
            mode = MODE_SAMPLE;
        } else if (!strcmp(argv[i], "--force")) {
            force = 1;
        } else if (!strcmp(argv[i], "--quiet")) {
            quiet = 1;
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    cai_config_t cfg;
    char err[160];
    int rc = cai_config_load(cfgpath, &cfg, err, sizeof(err));
    if (rc != CAI_OK) {
        fprintf(stderr, "config error: %s\n", err);
        return 1;
    }
    if (cai_topology_finalize(&cfg.topo, err, sizeof(err)) != CAI_OK) {
        fprintf(stderr, "topology error: %s\n", err);
        return 1;
    }

    cai_decomp_t dec;
    rc = cai_decompose(&cfg, &dec, err, sizeof(err));
    if (rc != CAI_OK) {
        fprintf(stderr, "decompose error: %s\n", err);
        return 1;
    }

    if (!quiet) {
        printf("=== decomposition ===\n");
        int fits = cai_report_decomp(stdout, &cfg, &dec);
        printf("=====================\n");
        if (!fits)
            fprintf(stderr,
                    "WARNING: per-GPU memory exceeds budget; adjust tp/pp/dp, "
                    "recompute, or precision.\n");
    }

    cai_topology_t topo = cfg.topo;
    cai_topology_finalize(&topo, err, sizeof(err));
    uint64_t th = cai_topology_hash(&topo);
    uint64_t ph = cai_plan_hash(&cfg, &dec);

    mkdir_p(outdir);
    char tpath[600];
    snprintf(tpath, sizeof(tpath), "%s/topology.bin", outdir);
    rc = cai_topology_write(tpath, &topo);
    if (rc != CAI_OK) {
        fprintf(stderr, "cannot write %s\n", tpath);
        return 1;
    }
    printf("wrote %s (topology_hash=%016llx, plan_hash=%016llx)\n", tpath,
           (unsigned long long)th, (unsigned long long)ph);

    uint32_t ws = topo.world_size;
    uint32_t replica = dec.model_replica_gpus;
    int failed = 0;

    if (mode == MODE_RANK) {
        if (arg_rank >= ws) {
            fprintf(stderr, "rank %u out of range (world=%u)\n", arg_rank, ws);
            return 1;
        }
        failed |= write_one(&cfg, &dec, arg_rank, ph, th, outdir) != CAI_OK;
    } else if (mode == MODE_RANGE) {
        if (arg_n > ws) arg_n = ws;
        for (uint32_t r = 0; r < arg_n; r++)
            failed |= write_one(&cfg, &dec, r, ph, th, outdir) != CAI_OK;
    } else if (mode == MODE_ALL) {
        if (ws > 4096 && !force) {
            fprintf(stderr, "refusing to write %u plans without --force\n", ws);
            return 1;
        }
        for (uint32_t r = 0; r < ws; r++)
            failed |= write_one(&cfg, &dec, r, ph, th, outdir) != CAI_OK;
    } else { /* sample */
        uint32_t s[4];
        uint32_t ns = 0;
        s[ns++] = 0;                                   /* stage 0, dp 0 */
        if (dec.pp > 1) s[ns++] = (dec.pp - 1) * dec.tp * dec.cp; /* last stage */
        if (dec.dp > 1) s[ns++] = replica;             /* dp replica 1 */
        s[ns++] = ws - 1;                              /* last rank */
        printf("compiling %u sample ranks:\n", ns);
        for (uint32_t i = 0; i < ns; i++) {
            int dup = 0;
            for (uint32_t j = 0; j < i; j++)
                if (s[j] == s[i]) dup = 1;
            if (!dup) failed |= write_one(&cfg, &dec, s[i], ph, th, outdir) != CAI_OK;
        }
    }

    return failed ? 1 : 0;
}
