#include "cai_model.h"

#include <stdlib.h>
#include <string.h>

#include "cai.h"
#include "cai_common.h"

/* ====================================================================== *
 *  config
 * ====================================================================== */

static int parse_dtype(const char *s, uint16_t *out) {
    for (uint16_t d = 0; d < CAI_DT_COUNT; d++)
        if (strcmp(s, cai_dtype_name(d)) == 0) {
            *out = d;
            return 0;
        }
    return -1;
}

void cai_config_defaults(cai_config_t *cfg) {
    memset(cfg, 0, sizeof(*cfg));
    cai_model_spec_t *m = &cfg->model;
    m->hidden_size = 8192;
    m->ffn_hidden = 28672;
    m->num_layers = 80;
    m->num_heads = 64;
    m->num_kv_heads = 8;
    m->head_dim = 128;
    m->vocab_size = 128000;
    m->seq_len = 8192;
    m->is_moe = 0;
    m->num_experts = 0;
    m->moe_top_k = 0;
    m->param_dtype = CAI_DT_BF16;
    m->optim_dtype = CAI_DT_FP32;
    m->optim_states = 2; /* AdamW m, v */
    m->master_weights = 1;
    m->recompute = 1;
    m->tp = 8;
    m->pp = 8;
    m->ep = 1;
    m->cp = 1;
    m->microbatch = 1;
    m->grad_accum = 32;
    m->mem_budget = 0;
    cai_topology_default(&cfg->topo);
}

static char *trim(char *s) {
    while (*s == ' ' || *s == '\t' || *s == '\r' || *s == '\n') s++;
    char *e = s + strlen(s);
    while (e > s && (e[-1] == ' ' || e[-1] == '\t' || e[-1] == '\r' || e[-1] == '\n'))
        *--e = 0;
    return s;
}

/* All numeric values are parsed as double so config may use 288e9 etc. */
static int set_kv(cai_config_t *cfg, const char *k, const char *v) {
    cai_model_spec_t *m = &cfg->model;
    cai_topology_t *t = &cfg->topo;
    double d = strtod(v, NULL);
    uint64_t u = (uint64_t)d;

    if (!strcmp(k, "hidden_size")) m->hidden_size = (uint32_t)u;
    else if (!strcmp(k, "ffn_hidden")) m->ffn_hidden = (uint32_t)u;
    else if (!strcmp(k, "num_layers")) m->num_layers = (uint32_t)u;
    else if (!strcmp(k, "num_heads")) m->num_heads = (uint32_t)u;
    else if (!strcmp(k, "num_kv_heads")) m->num_kv_heads = (uint32_t)u;
    else if (!strcmp(k, "head_dim")) m->head_dim = (uint32_t)u;
    else if (!strcmp(k, "vocab_size")) m->vocab_size = (uint32_t)u;
    else if (!strcmp(k, "seq_len")) m->seq_len = (uint32_t)u;
    else if (!strcmp(k, "is_moe")) m->is_moe = (int)u;
    else if (!strcmp(k, "num_experts")) m->num_experts = (uint32_t)u;
    else if (!strcmp(k, "moe_top_k")) m->moe_top_k = (uint32_t)u;
    else if (!strcmp(k, "optim_states")) m->optim_states = (uint32_t)u;
    else if (!strcmp(k, "master_weights")) m->master_weights = (int)u;
    else if (!strcmp(k, "recompute")) m->recompute = (int)u;
    else if (!strcmp(k, "tp")) m->tp = (uint32_t)u;
    else if (!strcmp(k, "pp")) m->pp = (uint32_t)u;
    else if (!strcmp(k, "ep")) m->ep = (uint32_t)u;
    else if (!strcmp(k, "cp")) m->cp = (uint32_t)u;
    else if (!strcmp(k, "microbatch")) m->microbatch = (uint32_t)u;
    else if (!strcmp(k, "grad_accum")) m->grad_accum = (uint32_t)u;
    else if (!strcmp(k, "mem_budget")) m->mem_budget = u;
    else if (!strcmp(k, "param_dtype")) return parse_dtype(v, &m->param_dtype);
    else if (!strcmp(k, "optim_dtype")) return parse_dtype(v, &m->optim_dtype);
    else if (!strcmp(k, "gpus_per_rack")) t->gpus_per_rack = (uint32_t)u;
    else if (!strcmp(k, "trays_per_rack")) t->trays_per_rack = (uint32_t)u;
    else if (!strcmp(k, "gpus_per_tray")) t->gpus_per_tray = (uint32_t)u;
    else if (!strcmp(k, "num_racks")) t->num_racks = (uint32_t)u;
    else if (!strcmp(k, "spare_gpus")) t->spare_gpus = (uint32_t)u;
    else if (!strcmp(k, "intra_rack_bw")) t->intra_rack_bw = u;
    else if (!strcmp(k, "inter_rack_bw")) t->inter_rack_bw = u;
    else if (!strcmp(k, "hbm_bw")) t->hbm_bw = u;
    else if (!strcmp(k, "gpu_flops_bf16")) t->gpu_flops_bf16 = d;
    else if (!strcmp(k, "gpu_mem_bytes")) t->gpu_mem_bytes = u;
    else return -1;
    return 0;
}

int cai_config_load(const char *path, cai_config_t *cfg, char *err, size_t errlen) {
    cai_config_defaults(cfg);
    FILE *f = fopen(path, "r");
    if (!f) {
        snprintf(err, errlen, "cannot open config '%s'", path);
        return CAI_ERR_IO;
    }
    char line[512];
    int lineno = 0;
    while (fgets(line, sizeof(line), f)) {
        lineno++;
        char *hash = strchr(line, '#');
        if (hash) *hash = 0;
        char *eq = strchr(line, '=');
        if (!eq) {
            char *t = trim(line);
            if (*t) {
                snprintf(err, errlen, "line %d: expected key = value", lineno);
                fclose(f);
                return CAI_ERR_INVALID;
            }
            continue;
        }
        *eq = 0;
        char *k = trim(line);
        char *v = trim(eq + 1);
        if (!*k) continue;
        if (set_kv(cfg, k, v) != 0) {
            snprintf(err, errlen, "line %d: unknown or invalid key '%s'", lineno, k);
            fclose(f);
            return CAI_ERR_INVALID;
        }
    }
    fclose(f);
    return CAI_OK;
}

/* ====================================================================== *
 *  sizing
 * ====================================================================== */

static uint64_t attn_params(const cai_model_spec_t *m) {
    uint64_t qd = (uint64_t)m->num_heads * m->head_dim;
    uint64_t kvd = (uint64_t)m->num_kv_heads * m->head_dim;
    uint64_t q = (uint64_t)m->hidden_size * qd;
    uint64_t kv = 2ull * m->hidden_size * kvd;
    uint64_t o = qd * m->hidden_size;
    return q + kv + o;
}

static uint64_t mlp_one(const cai_model_spec_t *m) {
    return 3ull * m->hidden_size * m->ffn_hidden;
}

/* full parameter count of one transformer block (all experts if MoE). */
static uint64_t layer_params_total(const cai_model_spec_t *m) {
    uint64_t mlp = m->is_moe ? (uint64_t)m->num_experts * mlp_one(m) +
                                   (uint64_t)m->hidden_size * m->num_experts
                             : mlp_one(m);
    return attn_params(m) + mlp + 2ull * m->hidden_size;
}

/* matmul params touched per token (MoE: only top_k experts). */
static uint64_t layer_params_active(const cai_model_spec_t *m) {
    uint64_t mlp = m->is_moe ? (uint64_t)m->moe_top_k * mlp_one(m) : mlp_one(m);
    return attn_params(m) + mlp;
}

int cai_decompose(const cai_config_t *cfg, cai_decomp_t *out, char *err,
                  size_t errlen) {
    const cai_model_spec_t *m = &cfg->model;
    cai_topology_t topo = cfg->topo;
    int rc = cai_topology_finalize(&topo, err, errlen);
    if (rc != CAI_OK) return rc;

    uint32_t ep = m->is_moe ? (m->ep ? m->ep : 1) : 1;
    if (m->tp == 0 || m->pp == 0 || m->cp == 0) {
        snprintf(err, errlen, "tp, pp, cp must be >= 1");
        return CAI_ERR_INVALID;
    }
    uint64_t replica = (uint64_t)m->tp * m->pp * m->cp;
    if (replica > topo.world_size) {
        snprintf(err, errlen,
                 "tp*pp*cp=%llu exceeds world_size=%u",
                 (unsigned long long)replica, topo.world_size);
        return CAI_ERR_INVALID;
    }
    if (topo.world_size % replica != 0) {
        snprintf(err, errlen,
                 "world_size=%u not divisible by tp*pp*cp=%llu (set spare_gpus)",
                 topo.world_size, (unsigned long long)replica);
        return CAI_ERR_INVALID;
    }
    uint32_t dp = (uint32_t)(topo.world_size / replica);
    if (m->is_moe && (dp % ep)) {
        snprintf(err, errlen, "ep=%u must divide dp=%u", ep, dp);
        return CAI_ERR_INVALID;
    }
    if (m->pp > m->num_layers) {
        snprintf(err, errlen, "pp=%u exceeds num_layers=%u", m->pp, m->num_layers);
        return CAI_ERR_INVALID;
    }

    memset(out, 0, sizeof(*out));
    out->tp = m->tp;
    out->pp = m->pp;
    out->cp = m->cp;
    out->ep = ep;
    out->dp = dp;
    out->model_replica_gpus = (uint32_t)replica;
    out->num_microbatches = m->grad_accum;
    out->layers_per_stage = (m->num_layers + m->pp - 1) / m->pp;
    out->global_batch = (uint64_t)m->microbatch * m->grad_accum * dp;
    out->global_tokens = out->global_batch * m->seq_len;
    out->bubble_ratio = cai_pipeline_bubble(m->pp, m->grad_accum);

    /* parameters */
    uint64_t lt = layer_params_total(m);
    uint64_t la = layer_params_active(m);
    uint64_t embed = (uint64_t)m->vocab_size * m->hidden_size; /* tied */
    out->total_params = (uint64_t)m->num_layers * lt + embed + m->hidden_size;
    out->active_params = (uint64_t)m->num_layers * la + embed;

    /* flops per token */
    double hd = (double)m->num_heads * m->head_dim;
    double attn_seq = 2.0 * hd * m->seq_len; /* causal score+context */
    double active_matmul = (double)m->num_layers * la + (double)embed; /* +logits */
    double fwd = 2.0 * active_matmul + (double)m->num_layers * attn_seq;
    double train_mult = m->recompute ? 4.0 : 3.0;
    out->flops_per_token_train = train_mult * fwd;
    double useful = 3.0 * fwd;
    out->step_flops = out->flops_per_token_train * (double)out->global_tokens;

    /* memory per GPU.
     * Non-expert params (attention, embeddings, router, norms) are FSDP-sharded
     * across tp*dp. Expert params are sharded across tp*ep only (and replicated
     * across the remaining dp/ep), so for MoE they occupy far more per GPU than a
     * naive tp*dp split would suggest. Dense reduces to tp*dp everywhere. */
    double pb = (double)cai_dtype_size(m->param_dtype);
    double ob = (double)cai_dtype_size(m->optim_dtype);
    uint64_t expert_per = m->is_moe ? (uint64_t)m->num_experts * mlp_one(m) : mlp_one(m);
    uint64_t nonexpert_per = attn_params(m) + 2ull * m->hidden_size +
                             (m->is_moe ? (uint64_t)m->hidden_size * m->num_experts : 0);
    double total_expert = (double)m->num_layers * expert_per;
    double total_nonexpert =
        (double)m->num_layers * nonexpert_per + (double)embed + m->hidden_size;
    double nonexpert_shard = (double)m->tp * dp;
    double expert_shard = m->is_moe ? (double)m->tp * ep : (double)m->tp * dp;
    double pcount = (total_nonexpert / m->pp) / nonexpert_shard +
                    (total_expert / m->pp) / expert_shard;
    out->mem_param = (uint64_t)(pcount * pb);
    out->mem_grad = (uint64_t)(pcount * pb);
    out->mem_opt =
        (uint64_t)(pcount * ((double)m->optim_states * ob + (m->master_weights ? 4.0 : 0.0)));

    double tokens_eff = (double)m->microbatch * m->seq_len / ((double)m->tp * m->cp);
    double a_per = (m->recompute ? 2.0 : 16.0) * m->hidden_size * pb;
    double in_flight = (m->pp < m->grad_accum) ? m->pp : m->grad_accum;
    double lstage = (double)m->num_layers / m->pp;
    out->mem_act = (uint64_t)(a_per * tokens_eff * lstage * in_flight);

    double pipe_buf = 2.0 * m->hidden_size * tokens_eff * pb;
    double tp_scratch = (double)m->hidden_size * tokens_eff * pb;
    double dp_w = 2.0 * ((double)lt / m->tp / (m->is_moe ? ep : 1)) * pb;
    out->mem_comm = (uint64_t)(pipe_buf + tp_scratch + dp_w);
    out->mem_ws = (uint64_t)(4.0 * m->hidden_size * tokens_eff * pb);

    out->mem_total = out->mem_param + out->mem_grad + out->mem_opt + out->mem_act +
                     out->mem_comm + out->mem_ws;

    /* stash useful flops/token into a temp via out (caller copies to header) */
    (void)useful;
    return CAI_OK;
}

uint64_t cai_mem_budget(const cai_config_t *cfg) {
    return cfg->model.mem_budget ? cfg->model.mem_budget : cfg->topo.gpu_mem_bytes;
}

int cai_report_decomp(FILE *f, const cai_config_t *cfg, const cai_decomp_t *d) {
    const cai_model_spec_t *m = &cfg->model;
    char b0[32], b1[32], b2[32], b3[32];
    uint64_t budget = cai_mem_budget(cfg);
    int fits = d->mem_total <= budget;

    fprintf(f, "model      : hidden=%u ffn=%u layers=%u heads=%u(kv=%u) head_dim=%u\n",
            m->hidden_size, m->ffn_hidden, m->num_layers, m->num_heads,
            m->num_kv_heads, m->head_dim);
    fprintf(f, "             vocab=%u seq=%u %s\n", m->vocab_size, m->seq_len,
            m->is_moe ? "MoE" : "dense");
    if (m->is_moe)
        fprintf(f, "             experts=%u top_k=%u\n", m->num_experts, m->moe_top_k);
    fprintf(f, "precision  : param=%s optim=%s states=%u master=%d recompute=%d\n",
            cai_dtype_name(m->param_dtype), cai_dtype_name(m->optim_dtype),
            m->optim_states, m->master_weights, m->recompute);
    fprintf(f, "cluster    : racks=%u gpus/rack=%u world=%u spare=%u\n",
            cfg->topo.num_racks, cfg->topo.gpus_per_rack, cfg->topo.world_size,
            cfg->topo.spare_gpus);
    fprintf(f, "per-gpu hw : %s FLOP/s bf16, mem %s\n",
            cai_fmt_count(cfg->topo.gpu_flops_bf16, b0, sizeof(b0)),
            cai_fmt_bytes(cfg->topo.gpu_mem_bytes, b1, sizeof(b1)));
    fprintf(f, "parallelism: TP=%u PP=%u DP=%u EP=%u CP=%u (replica=%u gpus)\n",
            d->tp, d->pp, d->dp, d->ep, d->cp, d->model_replica_gpus);
    fprintf(f, "batch      : microbatch=%u grad_accum=%u -> global_batch=%llu\n",
            m->microbatch, m->grad_accum, (unsigned long long)d->global_batch);
    fprintf(f, "tokens/step: %s  (%llu)\n",
            cai_fmt_count((double)d->global_tokens, b0, sizeof(b0)),
            (unsigned long long)d->global_tokens);
    fprintf(f, "params     : total=%s  active/token=%s\n",
            cai_fmt_count((double)d->total_params, b0, sizeof(b0)),
            cai_fmt_count((double)d->active_params, b1, sizeof(b1)));
    fprintf(f, "pipeline   : stages=%u microbatches=%u bubble=%.1f%% layers/stage~%u\n",
            d->pp, d->num_microbatches, d->bubble_ratio * 100.0, d->layers_per_stage);
    fprintf(f, "compute    : useful=%s FLOP/token  step=%s FLOP (cluster)\n",
            cai_fmt_count(d->flops_per_token_train / (cfg->model.recompute ? 4.0 : 3.0) * 3.0,
                          b0, sizeof(b0)),
            cai_fmt_count(d->step_flops, b1, sizeof(b1)));
    fprintf(f, "mem/gpu    : param=%s grad=%s opt=%s\n",
            cai_fmt_bytes(d->mem_param, b0, sizeof(b0)),
            cai_fmt_bytes(d->mem_grad, b1, sizeof(b1)),
            cai_fmt_bytes(d->mem_opt, b2, sizeof(b2)));
    fprintf(f, "             act=%s comm=%s ws=%s\n",
            cai_fmt_bytes(d->mem_act, b0, sizeof(b0)),
            cai_fmt_bytes(d->mem_comm, b1, sizeof(b1)),
            cai_fmt_bytes(d->mem_ws, b2, sizeof(b2)));
    fprintf(f, "             total=%s / budget=%s  [%s]\n",
            cai_fmt_bytes(d->mem_total, b0, sizeof(b0)),
            cai_fmt_bytes(budget, b1, sizeof(b1)),
            fits ? "OK" : "OVER BUDGET");
    if (!fits)
        fprintf(f, "             ^ over by %s\n",
                cai_fmt_bytes(d->mem_total - budget, b3, sizeof(b3)));
    return fits;
}

uint64_t cai_topology_hash(const cai_topology_t *topo) {
    return cai_fnv1a(topo, sizeof(*topo), CAI_FNV_INIT);
}

uint64_t cai_plan_hash(const cai_config_t *cfg, const cai_decomp_t *dec) {
    uint64_t h = cai_fnv1a(&cfg->model, sizeof(cfg->model), CAI_FNV_INIT);
    h = cai_fnv1a(&dec->dp, sizeof(dec->dp), h);
    h = cai_fnv1a(&dec->global_batch, sizeof(dec->global_batch), h);
    return h;
}

/* ====================================================================== *
 *  op-table builder
 * ====================================================================== */

typedef struct {
    cai_op_t *v;
    uint32_t n, cap;
    uint32_t next_event;
} opbuf_t;

static int ob_init(opbuf_t *ob) {
    ob->cap = 1024;
    ob->n = 0;
    ob->next_event = 0;
    ob->v = malloc(ob->cap * sizeof(cai_op_t));
    return ob->v ? 0 : -1;
}

static int ob_push(opbuf_t *ob, cai_op_t op) {
    if (ob->n == ob->cap) {
        uint32_t nc = ob->cap * 2;
        cai_op_t *nv = realloc(ob->v, (size_t)nc * sizeof(cai_op_t));
        if (!nv) return -1;
        ob->v = nv;
        ob->cap = nc;
    }
    ob->v[ob->n++] = op;
    return 0;
}

static void ob_compute(opbuf_t *ob, uint16_t kind, uint16_t stream, double flops,
                       uint16_t mb) {
    cai_op_t o = {0};
    o.kind = kind;
    o.stream_id = stream;
    o.flops = (uint64_t)(flops < 0 ? 0 : flops);
    o.aux1 = mb;
    ob_push(ob, o);
}

static void ob_comm(opbuf_t *ob, uint16_t kind, uint16_t stream, double bytes,
                    uint32_t group, uint16_t mb) {
    cai_op_t o = {0};
    o.kind = kind;
    o.stream_id = stream;
    o.bytes = (uint64_t)(bytes < 0 ? 0 : bytes);
    o.aux0 = group;
    o.aux1 = mb;
    ob_push(ob, o);
}

static uint32_t ob_record(opbuf_t *ob, uint16_t stream, uint16_t mb) {
    uint32_t e = ob->next_event++;
    cai_op_t o = {0};
    o.kind = CAI_OP_EVENT_RECORD;
    o.stream_id = stream;
    o.aux0 = e;
    o.aux1 = mb;
    ob_push(ob, o);
    return e;
}

static void ob_wait(opbuf_t *ob, uint16_t stream, uint32_t e, uint16_t mb) {
    cai_op_t o = {0};
    o.kind = CAI_OP_EVENT_WAIT;
    o.stream_id = stream;
    o.aux0 = e;
    o.aux1 = mb;
    ob_push(ob, o);
}

/* shapes & costs for one microbatch, precomputed once per stage */
typedef struct {
    double f_norm, f_qkv, f_attn, f_o, f_mlp_up, f_mlp_down, f_swiglu;
    double f_logits, f_loss, f_recompute, opt_flops;
    double tp_msg, pipe_msg, layer_w_bytes, a2a_bytes;
    uint32_t tp, dp, ep, pp, stage, layers, last_stage;
    int is_moe;
} stagecost_t;

static void emit_layer_fwd(opbuf_t *ob, const stagecost_t *c, uint16_t mb) {
    if (c->dp > 1) {
        ob_comm(ob, CAI_OP_ALL_GATHER, CAI_STREAM_COMM_DP, c->layer_w_bytes,
                CAI_GROUP_DP, mb);
        uint32_t e = ob_record(ob, CAI_STREAM_COMM_DP, mb);
        ob_compute(ob, CAI_OP_RMSNORM, CAI_STREAM_COMPUTE_LO, c->f_norm, mb);
        ob_wait(ob, CAI_STREAM_COMPUTE_HI, e, mb);
    } else {
        ob_compute(ob, CAI_OP_RMSNORM, CAI_STREAM_COMPUTE_LO, c->f_norm, mb);
    }
    ob_compute(ob, CAI_OP_GEMM, CAI_STREAM_COMPUTE_HI, c->f_qkv, mb);
    ob_compute(ob, CAI_OP_ATTENTION_FWD, CAI_STREAM_COMPUTE_HI, c->f_attn, mb);
    ob_compute(ob, CAI_OP_GEMM, CAI_STREAM_COMPUTE_HI, c->f_o, mb);
    if (c->tp > 1) {
        uint32_t e = ob_record(ob, CAI_STREAM_COMPUTE_HI, mb);
        ob_wait(ob, CAI_STREAM_COMM_TP, e, mb);
        ob_comm(ob, CAI_OP_ALL_REDUCE, CAI_STREAM_COMM_TP, c->tp_msg, CAI_GROUP_TP, mb);
        uint32_t e2 = ob_record(ob, CAI_STREAM_COMM_TP, mb);
        ob_wait(ob, CAI_STREAM_COMPUTE_HI, e2, mb);
        ob_wait(ob, CAI_STREAM_COMPUTE_LO, e2, mb);
    }
    ob_compute(ob, CAI_OP_RMSNORM, CAI_STREAM_COMPUTE_LO, c->f_norm, mb);
    if (c->is_moe && c->ep > 1) {
        ob_comm(ob, CAI_OP_ALL_TO_ALL, CAI_STREAM_COMM_DP, c->a2a_bytes, CAI_GROUP_EP, mb);
        uint32_t e = ob_record(ob, CAI_STREAM_COMM_DP, mb);
        ob_wait(ob, CAI_STREAM_COMPUTE_HI, e, mb);
    }
    ob_compute(ob, CAI_OP_GEMM, CAI_STREAM_COMPUTE_HI, c->f_mlp_up, mb);
    ob_compute(ob, CAI_OP_SWIGLU, CAI_STREAM_COMPUTE_LO, c->f_swiglu, mb);
    ob_compute(ob, CAI_OP_GEMM, CAI_STREAM_COMPUTE_HI, c->f_mlp_down, mb);
    if (c->is_moe && c->ep > 1) {
        ob_comm(ob, CAI_OP_ALL_TO_ALL, CAI_STREAM_COMM_DP, c->a2a_bytes, CAI_GROUP_EP, mb);
        uint32_t e = ob_record(ob, CAI_STREAM_COMM_DP, mb);
        ob_wait(ob, CAI_STREAM_COMPUTE_HI, e, mb);
    }
    if (c->tp > 1) {
        uint32_t e = ob_record(ob, CAI_STREAM_COMPUTE_HI, mb);
        ob_wait(ob, CAI_STREAM_COMM_TP, e, mb);
        ob_comm(ob, CAI_OP_ALL_REDUCE, CAI_STREAM_COMM_TP, c->tp_msg, CAI_GROUP_TP, mb);
        uint32_t e2 = ob_record(ob, CAI_STREAM_COMM_TP, mb);
        ob_wait(ob, CAI_STREAM_COMPUTE_HI, e2, mb);
    }
}

static void emit_layer_bwd(opbuf_t *ob, const stagecost_t *c, uint16_t mb) {
    if (c->f_recompute > 0)
        ob_compute(ob, CAI_OP_GEMM, CAI_STREAM_COMPUTE_HI, c->f_recompute, mb);
    /* weight + input grads ~ 2x forward matmul flops */
    ob_compute(ob, CAI_OP_GEMM, CAI_STREAM_COMPUTE_HI, 2.0 * c->f_mlp_down, mb);
    ob_compute(ob, CAI_OP_GEMM, CAI_STREAM_COMPUTE_HI, 2.0 * c->f_mlp_up, mb);
    ob_compute(ob, CAI_OP_ATTENTION_BWD, CAI_STREAM_COMPUTE_HI, 2.0 * c->f_attn, mb);
    ob_compute(ob, CAI_OP_GEMM, CAI_STREAM_COMPUTE_HI, 2.0 * (c->f_qkv + c->f_o), mb);
    if (c->tp > 1) {
        uint32_t e = ob_record(ob, CAI_STREAM_COMPUTE_HI, mb);
        ob_wait(ob, CAI_STREAM_COMM_TP, e, mb);
        ob_comm(ob, CAI_OP_ALL_REDUCE, CAI_STREAM_COMM_TP, c->tp_msg, CAI_GROUP_TP, mb);
        uint32_t e2 = ob_record(ob, CAI_STREAM_COMM_TP, mb);
        ob_wait(ob, CAI_STREAM_COMPUTE_HI, e2, mb);
    }
    /* reduce-scatter weight grads, overlapped with following backward compute */
    if (c->dp > 1)
        ob_comm(ob, CAI_OP_REDUCE_SCATTER, CAI_STREAM_COMM_DP, c->layer_w_bytes,
                CAI_GROUP_DP, mb);
}

static void emit_microbatch_fwd(opbuf_t *ob, const stagecost_t *c, uint16_t mb) {
    if (c->stage > 0) {
        ob_comm(ob, CAI_OP_PIPE_RECV, CAI_STREAM_COMM_PP, c->pipe_msg, CAI_GROUP_PP, mb);
        uint32_t e = ob_record(ob, CAI_STREAM_COMM_PP, mb);
        ob_wait(ob, CAI_STREAM_COMPUTE_HI, e, mb);
        ob_wait(ob, CAI_STREAM_COMPUTE_LO, e, mb);
    }
    for (uint32_t l = 0; l < c->layers; l++) emit_layer_fwd(ob, c, mb);
    if (c->stage < c->last_stage) {
        uint32_t e = ob_record(ob, CAI_STREAM_COMPUTE_HI, mb);
        ob_wait(ob, CAI_STREAM_COMM_PP, e, mb);
        ob_comm(ob, CAI_OP_PIPE_SEND, CAI_STREAM_COMM_PP, c->pipe_msg, CAI_GROUP_PP, mb);
    } else {
        ob_compute(ob, CAI_OP_GEMM, CAI_STREAM_COMPUTE_HI, c->f_logits, mb);
        ob_compute(ob, CAI_OP_ELEMENTWISE, CAI_STREAM_COMPUTE_LO, c->f_loss, mb);
    }
}

static void emit_microbatch_bwd(opbuf_t *ob, const stagecost_t *c, uint16_t mb) {
    if (c->stage == c->last_stage) {
        ob_compute(ob, CAI_OP_ELEMENTWISE, CAI_STREAM_COMPUTE_LO, c->f_loss, mb);
        ob_compute(ob, CAI_OP_GEMM, CAI_STREAM_COMPUTE_HI, 2.0 * c->f_logits, mb);
    } else {
        ob_comm(ob, CAI_OP_PIPE_RECV, CAI_STREAM_COMM_PP, c->pipe_msg, CAI_GROUP_PP, mb);
        uint32_t e = ob_record(ob, CAI_STREAM_COMM_PP, mb);
        ob_wait(ob, CAI_STREAM_COMPUTE_HI, e, mb);
    }
    for (uint32_t l = 0; l < c->layers; l++) emit_layer_bwd(ob, c, mb);
    if (c->stage > 0) {
        uint32_t e = ob_record(ob, CAI_STREAM_COMPUTE_HI, mb);
        ob_wait(ob, CAI_STREAM_COMM_PP, e, mb);
        ob_comm(ob, CAI_OP_PIPE_SEND, CAI_STREAM_COMM_PP, c->pipe_msg, CAI_GROUP_PP, mb);
    }
}

/* ====================================================================== *
 *  per-rank plan
 * ====================================================================== */

int cai_build_plan(const cai_config_t *cfg, const cai_decomp_t *dec, uint32_t rank,
                   uint64_t plan_hash, uint64_t topo_hash, cai_plan_t *out,
                   char *err, size_t errlen) {
    const cai_model_spec_t *m = &cfg->model;
    cai_topology_t topo = cfg->topo;
    int rc = cai_topology_finalize(&topo, err, errlen);
    if (rc != CAI_OK) return rc;
    if (rank >= topo.world_size) {
        snprintf(err, errlen, "rank %u >= world_size %u", rank, topo.world_size);
        return CAI_ERR_INVALID;
    }

    uint32_t tp = dec->tp, pp = dec->pp, cp = dec->cp, dp = dec->dp, ep = dec->ep;
    uint64_t replica = (uint64_t)tp * pp * cp;

    /* rank coordinate: TP innermost (kept intra-rack), then CP, PP, DP */
    uint32_t tp_id = rank % tp;
    uint32_t cp_id = (rank / tp) % cp;
    uint32_t pp_id = (uint32_t)((rank / ((uint64_t)tp * cp)) % pp);
    uint32_t dp_id = (uint32_t)(rank / replica);
    uint32_t stage = pp_id;

    uint32_t base = m->num_layers / pp, rem = m->num_layers % pp;
    uint32_t layers = base + (stage < rem ? 1 : 0);

    memset(out, 0, sizeof(*out));
    cai_plan_header_t *h = &out->hdr;
    h->magic = CAI_PLAN_MAGIC;
    h->version = CAI_PLAN_VERSION;
    h->plan_hash = plan_hash;
    h->topology_hash = topo_hash;
    h->global_rank = rank;
    h->world_size = topo.world_size;
    h->pipeline_stage = stage;
    h->num_stages = pp;
    h->tp = tp;
    h->dp = dp;
    h->ep = ep;
    h->cp = cp;
    h->microbatch = m->microbatch;
    h->grad_accum = m->grad_accum;
    h->num_microbatches = dec->num_microbatches;
    h->seq_len = m->seq_len;
    h->hidden_size = m->hidden_size;
    h->layers_in_stage = layers;
    h->global_tokens_per_step = dec->global_tokens;
    {
        double hd = (double)m->num_heads * m->head_dim;
        double attn_seq = 2.0 * hd * m->seq_len;
        double active_matmul =
            (double)m->num_layers * layer_params_active(m) +
            (double)m->vocab_size * m->hidden_size;
        double fwd = 2.0 * active_matmul + (double)m->num_layers * attn_seq;
        h->flops_per_token_useful = (uint64_t)(3.0 * fwd);
    }

    /* comm groups */
    cai_comm_group_t groups[CAI_GROUP_COUNT];
    uint32_t ng = 0;
    {
        uint32_t tp_first = rank - tp_id;
        groups[ng].group_id = ng;
        groups[ng].kind = CAI_GROUP_TP;
        groups[ng].size = tp;
        groups[ng].color = tp_first;
        groups[ng].intra_rack =
            cai_same_rack(&topo, tp_first, tp_first + tp - 1) ? 1 : 0;
        ng++;

        uint32_t within = (uint32_t)(rank % replica);
        groups[ng].group_id = ng;
        groups[ng].kind = CAI_GROUP_DP;
        groups[ng].size = dp;
        groups[ng].color = within;
        groups[ng].intra_rack =
            (dp > 1 && cai_same_rack(&topo, within, within + (dp - 1) * replica)) ? 1 : 0;
        ng++;

        uint32_t pp_color = dp_id * (tp * cp) + cp_id * tp + tp_id;
        groups[ng].group_id = ng;
        groups[ng].kind = CAI_GROUP_PP;
        groups[ng].size = pp;
        groups[ng].color = pp_color;
        groups[ng].intra_rack = 0;
        ng++;

        if (m->is_moe && ep > 1) {
            groups[ng].group_id = ng;
            groups[ng].kind = CAI_GROUP_EP;
            groups[ng].size = ep;
            groups[ng].color = within + (dp_id / ep) * ep * (uint32_t)replica;
            groups[ng].intra_rack = 0;
            ng++;
        }
    }

    /* per-microbatch costs */
    stagecost_t c;
    memset(&c, 0, sizeof(c));
    c.tp = tp; c.dp = dp; c.ep = ep; c.pp = pp;
    c.stage = stage; c.layers = layers; c.last_stage = pp - 1;
    c.is_moe = m->is_moe;
    double pb = (double)cai_dtype_size(m->param_dtype);
    double T = (double)m->microbatch * m->seq_len;
    double hd = (double)m->num_heads * m->head_dim;
    double kvd = (double)m->num_kv_heads * m->head_dim;
    double tpf = (double)tp;
    c.f_qkv = 2.0 * T * m->hidden_size * (hd + 2.0 * kvd) / tpf;
    c.f_o = 2.0 * T * hd * m->hidden_size / tpf;
    c.f_attn = 2.0 * T * hd * m->seq_len / tpf;
    double mlp_full = 2.0 * 3.0 * m->hidden_size * m->ffn_hidden * T *
                      (m->is_moe ? m->moe_top_k : 1);
    double f_mlp = m->is_moe ? mlp_full / (tpf * ep) : mlp_full / tpf;
    c.f_mlp_up = f_mlp * 2.0 / 3.0;
    c.f_mlp_down = f_mlp / 3.0;
    c.f_norm = 5.0 * T * m->hidden_size / tpf;
    c.f_swiglu = 4.0 * T * m->ffn_hidden * (m->is_moe ? m->moe_top_k : 1) / tpf;
    c.f_logits = 2.0 * T * m->hidden_size * m->vocab_size / tpf;
    c.f_loss = 5.0 * T * m->vocab_size / tpf;
    c.f_recompute = m->recompute ? (c.f_qkv + c.f_o + c.f_mlp_up + c.f_mlp_down) : 0.0;
    double pcount = (double)dec->mem_param / (pb > 0 ? pb : 1);
    c.opt_flops = pcount * 18.0; /* AdamW-ish per param */

    double act_bytes = T * m->hidden_size * pb;
    c.tp_msg = act_bytes / cp;
    c.pipe_msg = act_bytes / (tpf * cp);
    c.layer_w_bytes =
        (double)layer_params_total(m) / tpf / (m->is_moe ? ep : 1) * pb;
    c.a2a_bytes = m->is_moe ? T * m->moe_top_k * m->hidden_size * pb / tpf : 0.0;

    /* op table follows the 1F1B schedule for this stage */
    uint32_t mtot = dec->num_microbatches;
    cai_tick_t *ticks = malloc((size_t)2 * mtot * sizeof(cai_tick_t));
    if (!ticks) {
        snprintf(err, errlen, "oom ticks");
        return CAI_ERR_OOM;
    }
    uint32_t nt = cai_pipeline_schedule_1f1b(stage, pp, mtot, ticks);

    opbuf_t ob;
    if (ob_init(&ob) != 0) {
        free(ticks);
        snprintf(err, errlen, "oom opbuf");
        return CAI_ERR_OOM;
    }
    for (uint32_t i = 0; i < nt; i++) {
        if (ticks[i].kind == CAI_TICK_FWD)
            emit_microbatch_fwd(&ob, &c, ticks[i].microbatch);
        else
            emit_microbatch_bwd(&ob, &c, ticks[i].microbatch);
    }
    free(ticks);

    /* optimizer: barrier on grad reduce-scatter, then update */
    if (dp > 1) {
        uint32_t e = ob_record(&ob, CAI_STREAM_COMM_DP, 0);
        ob_wait(&ob, CAI_STREAM_COMPUTE_LO, e, 0);
    }
    ob_compute(&ob, CAI_OP_OPTIMIZER, CAI_STREAM_COMPUTE_LO, c.opt_flops, 0);

    /* tensor table: one descriptor per class for memory accounting */
    uint64_t want[CAI_TCLASS_COUNT] = {dec->mem_param, dec->mem_grad, dec->mem_opt,
                                       dec->mem_act,   dec->mem_comm, dec->mem_ws};
    cai_arena_t arena;
    cai_arena_layout(&arena, want);
    cai_tensor_desc_t *tensors = calloc(CAI_TCLASS_COUNT, sizeof(cai_tensor_desc_t));
    if (!tensors) {
        free(ob.v);
        snprintf(err, errlen, "oom tensors");
        return CAI_ERR_OOM;
    }
    for (int cc = 0; cc < CAI_TCLASS_COUNT; cc++) {
        tensors[cc].offset = 0;
        tensors[cc].nbytes = want[cc];
        tensors[cc].dtype = (cc == CAI_TCLASS_OPTSTATE) ? m->optim_dtype : m->param_dtype;
        tensors[cc].tclass = (uint16_t)cc;
        tensors[cc].shape_id = (uint32_t)cc;
    }
    cai_arena_relocate(&arena, tensors, CAI_TCLASS_COUNT);

    cai_comm_group_t *gcopy = malloc(ng * sizeof(cai_comm_group_t));
    if (!gcopy) {
        free(ob.v);
        free(tensors);
        snprintf(err, errlen, "oom groups");
        return CAI_ERR_OOM;
    }
    memcpy(gcopy, groups, ng * sizeof(cai_comm_group_t));

    h->num_tensors = CAI_TCLASS_COUNT;
    h->num_groups = ng;
    h->num_ops = ob.n;
    h->num_events = ob.next_event;
    h->arena_bytes = arena.total;

    out->tensors = tensors;
    out->groups = gcopy;
    out->ops = ob.v;
    return CAI_OK;
}
