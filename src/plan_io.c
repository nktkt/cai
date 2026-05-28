#include <stdlib.h>

#include "cai.h"
#include "cai_common.h"
#include "cai_plan.h"

int cai_plan_write(const char *path, const cai_plan_t *plan) {
    if (plan->hdr.magic != CAI_PLAN_MAGIC) return CAI_ERR_INVALID;
    FILE *f = fopen(path, "wb");
    if (!f) return CAI_ERR_IO;
    int rc = cai_write_all(f, &plan->hdr, sizeof(plan->hdr));
    if (rc == CAI_OK)
        rc = cai_write_all(f, plan->tensors,
                           (size_t)plan->hdr.num_tensors * sizeof(cai_tensor_desc_t));
    if (rc == CAI_OK)
        rc = cai_write_all(f, plan->groups,
                           (size_t)plan->hdr.num_groups * sizeof(cai_comm_group_t));
    if (rc == CAI_OK)
        rc = cai_write_all(f, plan->ops,
                           (size_t)plan->hdr.num_ops * sizeof(cai_op_t));
    fclose(f);
    return rc;
}

int cai_plan_read(const char *path, cai_plan_t *plan) {
    FILE *f = fopen(path, "rb");
    if (!f) return CAI_ERR_IO;
    plan->tensors = NULL;
    plan->groups = NULL;
    plan->ops = NULL;

    int rc = cai_read_all(f, &plan->hdr, sizeof(plan->hdr));
    if (rc != CAI_OK) goto done;
    if (plan->hdr.magic != CAI_PLAN_MAGIC || plan->hdr.version != CAI_PLAN_VERSION) {
        rc = CAI_ERR_PLAN;
        goto done;
    }

    plan->tensors =
        cai_read_array(f, plan->hdr.num_tensors, sizeof(cai_tensor_desc_t));
    plan->groups =
        cai_read_array(f, plan->hdr.num_groups, sizeof(cai_comm_group_t));
    plan->ops = cai_read_array(f, plan->hdr.num_ops, sizeof(cai_op_t));

    if ((plan->hdr.num_tensors && !plan->tensors) ||
        (plan->hdr.num_groups && !plan->groups) ||
        (plan->hdr.num_ops && !plan->ops)) {
        rc = CAI_ERR_IO;
        cai_plan_free(plan);
    }

done:
    fclose(f);
    return rc;
}

void cai_plan_free(cai_plan_t *plan) {
    free(plan->tensors);
    free(plan->groups);
    free(plan->ops);
    plan->tensors = NULL;
    plan->groups = NULL;
    plan->ops = NULL;
}

void cai_plan_arena_breakdown(const cai_plan_t *plan,
                              uint64_t out_bytes[CAI_TCLASS_COUNT]) {
    for (int c = 0; c < CAI_TCLASS_COUNT; c++) out_bytes[c] = 0;
    for (uint32_t i = 0; i < plan->hdr.num_tensors; i++) {
        uint16_t c = plan->tensors[i].tclass;
        if (c < CAI_TCLASS_COUNT) out_bytes[c] += plan->tensors[i].nbytes;
    }
}
