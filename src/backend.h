/* backend.h - execution backend abstraction.
 *
 * The runtime is backend-agnostic: it replays the op stream and asks the backend
 * how long each op takes (and, on real hardware, to actually launch it). V1 ships
 * a CPU analytic backend that *simulates* op cost from the resolved topology, so
 * the whole stack builds and runs end-to-end with no GPU. A CUDA backend that
 * launches precompiled cubins via the driver API is stubbed behind CAI_WITH_CUDA. */
#ifndef CAI_BACKEND_H
#define CAI_BACKEND_H

#include "cai_plan.h"
#include "cai_topology.h"

typedef struct cai_backend cai_backend_t;

struct cai_backend {
    const char *name;
    void *impl;
    /* Seconds this op occupies its stream. For comm ops the cost reflects the
     * collective algorithm and whether the group is intra-rack (NVLink) or
     * inter-rack (NIC). */
    double (*op_seconds)(cai_backend_t *be, const cai_op_t *op,
                         const cai_comm_group_t *group);
    int (*sync)(cai_backend_t *be);
    void (*destroy)(cai_backend_t *be);
};

/* Always available. */
cai_backend_t *cai_backend_cpu_create(const cai_topology_t *topo);

/* Returns NULL unless built with CAI_WITH_CUDA and a device is present. */
cai_backend_t *cai_backend_cuda_create(const cai_topology_t *topo);

#endif /* CAI_BACKEND_H */
