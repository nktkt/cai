# cai — 220k GB300 専用 C トレーニングスタック (設計 / V1.0)

## 0. このリポジトリの位置づけ

「Cで書いた、220,000 GPU 専用の AI トレーニングスタック」という構想を、**実際にビルド・実行・検証できる形**で起こしたもの。

ただし本物の学習(GEMM/collective/kernel を GPU で回す)は、ここには 22 万 GPU が無いので **実行できない**。
そこでこの V1.0 では、設計が「最重要」と位置づける部分 ——

- offline plan compiler（並列化分解・メモリ配置・通信グループ・パイプラインスケジュールの事前生成）
- topology mapping（rank ↔ rack/tray/gpu）
- static schedule（1F1B の発行順）
- static memory arena（実行時 malloc ゼロ前提のレイアウト）
- op-table runtime（イベント駆動でスケジュールを再生し、step time / bubble / MFU / tokens/s を算出）

—— を **GPU 非依存の本物の C コード**として完成させている。GPU が要る所(GEMM/cuDNN/NCCL/NVSHMEM/CUDA Graphs/cubin)は `#ifdef CAI_WITH_CUDA` の背後に隔離し、CPU 解析バックエンドで全体が動く。

つまり: **「学習は回せないが、220,032 rank の plan を生成・検証し、runtime でスケジュールを流して性能特性を見積もる」**ところまでは実物として動く。

## 1. 目標 / 非目標

作るもの (V1.0):
- 固定モデル・固定 shape・固定 precision・固定 topology・固定並列構成 専用の **静的分散トレーニング実行系**
- runtime / scheduler / memory planner / comm coordinator / checkpoint / telemetry

作らないもの (V1.0):
- 自動微分・動的 shape・Python runtime・汎用 operator dispatcher
- 実行時の malloc/free・実行時グラフコンパイル
- 「C 版 JAX」になること（汎用性は捨てる、が勝ち筋）

## 2. アーキテクチャ

判断は全部オフライン、runtime はそれを replay するだけ。

```
                 model + parallelism + topology (config)
                                 |
                    +------------v------------+
                    |   offline plan compiler  |   tools/plan_compiler
                    |  decompose -> validate   |
                    |  -> emit per-rank plans  |
                    +------------+------------+
                                 |
              topology.bin + plan.rankNNNNNN.bin  (binary)
                                 |
                    +------------v------------+
                    |       C runtime          |   src/runtime.c
                    |  load + replay op-table  |
                    |  event-driven streams    |
                    +------------+------------+
                                 |
                    +------------v------------+
                    |        backend           |   src/backend_*.c
                    |  CPU analytic (here)     |
                    |  CUDA launch (#ifdef)    |
                    +-------------------------+
```

runtime は「考えない」: op を stream 毎に発行順で進め、record/wait イベントで stream 間を同期するだけ。

## 3. バイナリ形式

すべて固定レイアウトの POD で、`_Static_assert` でサイズを固定（同一クラスタ=同一エンディアン前提）。

- `topology.bin` : magic `CTOP` + `cai_topology_t`(64B)
- `plan.rankNNNNNN.bin` : `cai_plan_header_t`(128B) + tensor table + comm groups + op table
  - header に `plan_hash` / `topology_hash` を埋め込み、runtime は**異なる topology 用に compile された plan を拒否**する（22 万 GPU での事故防止）。

主要構造体:
| 構造体 | サイズ | 役割 |
|---|---|---|
| `cai_phys_id_t` | 12B | rank の物理位置 (rack/tray/gpu/nic/rail) |
| `cai_topology_t` | 64B | クラスタ記述 (帯域/FLOPS/メモリ含む) |
| `cai_op_t` | 32B | 1 命令 (flops or bytes, stream, event/group) |
| `cai_comm_group_t` | 16B | communicator (kind/color/size、ranks は導出) |
| `cai_tensor_desc_t` | 24B | arena 内テンソル (class/offset/bytes) |
| `cai_plan_header_t` | 128B | plan メタ + hash + 派生量 |

## 4. rank マッピングと並列化

GB300 NVL72 = 1 rack 72 GPU = 1 NVLink ドメイン。rank 座標は **TP を最内**に置く:

```
tp_id = rank % tp
cp_id = (rank / tp) % cp
pp_id = (rank / (tp*cp)) % pp      <- pipeline stage
dp_id =  rank / (tp*cp*pp)
```

これで TP グループが連続 rank=同一 rack に収まり、NVLink 内で閉じる（linter が `TP <= gpus_per_rack` と割り切れを検査）。
- ラック内: Tensor / Sequence / 小規模 Expert parallel
- ラック間: Pipeline activation / DP reduce-scatter / MoE all-to-all

`tp*pp*cp` が world を割り切らない場合は compiler がエラーにし、`spare_gpus` での調整を促す。

## 5. op-table とシミュレーションモデル

compiler は各 stage の 1 step 分（全 microbatch の fwd+bwd + optimizer）を、1F1B 発行順で op に展開する（`tools` ではなく `src/model.c`）。

stream は 6 本: `compute_hi / compute_lo / comm_tp / comm_pp / comm_dp / io`。
依存は **EVENT_RECORD / EVENT_WAIT** で表現:
- TP all-reduce は compute と直列（隠せない通信、現実的）
- DP weight all-gather (FSDP) は次レイヤ計算と**オーバーラップ**するよう発行
- DP grad reduce-scatter は backward 計算と**オーバーラップ**、optimizer 直前にバリア

runtime のシミュレーション（`cai_train_step`）:
1. op を発行順に 1 パス走査。stream 毎に時刻を進め、RECORD は event 時刻を記録、WAIT は `max()` で stream を待たせる（compiler が DAG を保証するので 1 パスで正しい）。
2. `busy = max(stream 時刻)` … TP/DP 通信が計算に隠れたかが自然に出る。
3. パイプライン fill/drain は単一 rank では見えないので解析式で補正:
   `step_time = busy / (1 - bubble)`, `bubble = (pp-1)/(m+pp-1)`。
4. `MFU = (useful 6N flops/GPU) / (peak FLOPS * step_time)`、tokens/s なども算出。

op の所要時間は backend が返す（CPU: `flops/peak` と `bytes/帯域`、collective は ring 係数 + intra/inter-rack 帯域）。CUDA backend は同じ関数を「実 launch + 実測」に差し替える設計。

## 6. メモリ arena

実行時 malloc 禁止。クラス別セグメント（param/grad/optstate/activation/comm/workspace）を連続配置し、テンソルは class 内オフセット→arena 全体オフセットへ rebase（`src/arena.c`）。
per-GPU メモリ見積りは FSDP/ZeRO-3 シャーディング前提で算出し、`budget`(=HBM 容量 or 上書き値) と突き合わせて OK/OVER を報告。

## 7. ファイル対応表

| 設計概念 | 実体 |
|---|---|
| offline plan compiler | `tools/plan_compiler.c` + `src/model.c` |
| topology linter | `tools/topology_linter.c` |
| runtime driver (1 proc = 1 GPU) | `tools/trainer.c` + `src/runtime.c` |
| rank/topology mapping | `src/topology.c` |
| static memory arena | `src/arena.c` |
| 1F1B schedule + bubble | `src/pipeline.c` |
| op-table / sizing / 並列分解 | `src/model.c` |
| plan (de)serialize | `src/plan_io.c` |
| backend 抽象 | `src/backend.h` |
| CPU 解析 backend | `src/backend_cpu.c` |
| CUDA backend (stub) | `src/backend_cuda.c` |
| io / hash / log / fmt | `src/common.c` |

## 8. V1.0 の近似と TODO

意図的な単純化（数値は妥当な桁になるが厳密ではない）:
- collective は ring 係数 + 単一リンク帯域の概算（rail/輻輳は未モデル）
- MoE の expert ストレージ分散は EP を厳密に分けず dp シャードに丸め（all-to-all と active param は反映）
- activation/通信バッファ係数は代表値（recompute 有無で切替）

CUDA 実装で埋める順（`backend_cuda.c`）:
1. Driver API で precompiled cubin を `cuLaunchKernel`、stream/event を実体化
2. comm groups から NCCL communicator を構築、collective を実 launch
3. NVSHMEM で pipeline / MoE の細粒度通信、CUDA Graphs で steady step を焼く
4. `op_seconds` を実測 (CUDA events) に差し替え → 同じ runtime がそのまま本番に

速度の現実的な狙い（「JAX 比 10x」の内訳）:
- V1.0: static memory + CUDA Graph + topology-aware mapping + comm/compute overlap で **goodput 1.3–2.5x**
- V1.5: custom kernel + pipeline 最適化で **3–5x**
- V2 : megakernel + GPU-initiated scheduler + model/topology co-design で条件次第 **5x 超**
- 10x は「C だから」ではなく、**汎用性を捨てて 220k GB300 の物理構成へ完全特化**したときの複合効果として狙う。

## 9. 最大の失敗ポイント（自戒）

1. 汎用フレームワーク化（→ JAX/PyTorch の劣化コピー）
2. GEMM/NCCL を最初から自作（→ V1 が破綻）。まず NVIDIA stack を叩き、真の bottleneck だけ置換
3. pipeline stage imbalance の軽視（最遅 stage が全体を支配）
4. checkpoint の後回し（22 万 GPU で無故障前提は成立しない → V1 機能）
5. 雑な JAX 比較（同一 model/tokens/precision/収束/故障条件で測る）
