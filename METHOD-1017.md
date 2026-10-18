# METHOD — DeepSeek-V4.1-Flash-1017 (EXL3 2.9bpw) on 2× DGX Spark, at Sudosu's tuned ceiling

**Date:** 2026-10-18 · **Engine:** `dsv41-serve:v51` (vLLM 0.31.1-dev `6e517b15` + 22 patches + `vllm_exl3_moe` plugin, exllamav3 1.6.0, TP2 over RoCE) · **Weights:** LibertAIDAI 1017 2.9bpw (210.8 GB, 43 shards) + fresh 1017 Engram shards 47/48 (~203 GB)

This document is the exact procedure that took the DeepSeek-V4.1-**1017** community EXL3 pack from
"cannot load" to serving at Sudosu's measured ceiling on two DGX Spark GB10 nodes — including every
blocker, its root cause, and the fix. Anyone with the links below should be able to reproduce all of
it in an afternoon instead of an 18-hour grind.

## Result summary (Sudosu's own benchmark client, `kit_bench.py`, greedy, streaming)

Like-for-like by prompt, measured on the running lane:

| prompt | Sudosu v1.1 (own 2.70bpw pack) | this build (2.9bpw pack) |
|---|---|---|
| set b code | 102.7 tok/s | 93.8 tok/s |
| set b prose | 57.9 tok/s | 52.4 tok/s |
| set b structured | 141.0 tok/s | 126.8 tok/s |

The −9-10 % delta is **the bitrate, not the tuning**: Sudosu's pack is 2.70 bpw, this pack is 2.9 bpw
(+7.4 % weight bytes per token). Per byte moved we are at or above parity. On his set-a prompts the
two lanes measure within 1 % of each other (code 86.0/86.6, prose 51.4/54.2, structured 102.7/103.7).
Against the Mia-AI-Lab stock kit measured same-box by the same client (41.6 / 25.1 / 40.4) this lane
is 2–3× faster on every set.

Depth policy (k=6, corrected cost table, per-round adaptive):

| raw depth | code | prose | structured |
|---|---|---|---|
| 0 | 31.3 | 31.5 | 31.5 |
| 1 | 54.7 | 45.8 | 55.1 |
| 2 | 69.3 | 50.3 | 77.7 |
| 3 | 78.9 | 49.3 | 96.5 |
| 4 | 82.8 | 47.8 | 114.5 |
| 5 | 85.6 | 45.0 | 131.6 |
| 6 | 87.3 | 42.7 | 127.8 |

Prose peaks shallow (~50, acceptance collapses after ~2 tokens on flowing text); structured peaks
deep (131.6). The confidence policy with the **correct cost table** beats every fixed depth
(93.8 / 52.4 / 126.8) and beats the stale table it was shipped with — the table below is why.

Vision (the V4.1 vision tower) is served and verified (`TEXT_ONLY=0`).

## Step 1 — weights

```bash
# the pack (210.8 GB, 43 shards)
hf download LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw --local-dir /home/jaita/models/dsv41-1017-exl3-2.9

# FRESH Engram: shards 47+48 + the two support files from the 1017 base. The pre-1017
# tables are POISON for this pack (all 48 shards changed upstream)
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 \
   --include "model-00047-of-00048.safetensors" "model-00048-of-00048.safetensors" \
   "model.safetensors.index.json" "config.json" --local-dir /home/jaita/models/dsv41-1017-engram-src
```

Verify shard sizes against the HF API (byte-exact) and sha256 the Engram shards against the card
(~200 GB, ~30 min on NVMe).

**Pack layout facts** (probe the safetensors headers, no torch needed — the header is a JSON block
after an 8-byte little-endian length):

- routed experts `ffn.experts.{N}.w{1,2,3}.trellis`: `[in/16, out/16, 16K]` int16 — K = last dim / 16
- target routed experts 384 across 40 layers, 2.9bpw; head (indexer) 6 bpw
- compressor tensors are **stacked per-piece pairs**: `compressor.wkv.*` / `compressor.wgate.*`,
  K=5, layers {2, 8, 14, 20}; layer 20 is wkv-only
- **MTP/drafter (mtp.0/1/2, 3 stages, 128 experts each):** attn dense K=6
  (`wq_a` [320,80,96], `wo_b` [512,320,96], `wkv` [320,32,96], `wo_a` **8 head-group slices**
  each [256,64,96]), `main_proj` [960,320,96] K=6, shared experts w1/w2/w3 K=6, routed experts K=4

## Step 2 — repack into the recipe's serving format

The recipe consumes `quant_method: "deepseek_v41_exl3"` + a per-layer `exl3_moe_plan.json` and routes
every dense module by module-name at load. The pack advertises `quant_method: "exl3"` (the Mia
k_map schema), so the conversion is mechanical — `repack/repack_ours.py` does all of it:

1. **Hardlink tree** `dsv41-1017-ours/` from the pack (zero-copy; du shows 2× but nlink=3), plus
   the two Engram shards.
2. **Union index**: the pack's 43-shard `model.safetensors.index.json` + 4 engram entries appended
   (`layers.1/14.engram.embed.{weight,scale}` → shards 47/48). Without this the loader raises
   `KeyError: 'layers.1.engram.embed.weight'` — the engine reads the Engram tables THROUGH the
   checkpoint's own index.
3. **config.json**: `quantization_config.quant_method` → `deepseek_v41_exl3`, keep the pack's
   model-side fields; ensure `expert_dtype: fp4`.
4. **`quantization_config.exl3_moe.dense.drafter`** (K per drafter module):
   `{"attn.wq_a": 6, "attn.wq_b": 6, "attn.wo_b": 6, "attn.wo_a": 6, "main_proj": 6,
     "ffn.shared_experts.w1": 6, "ffn.shared_experts.w2": 6}` — note `attn.wkv` and
   `shared_experts.w3` are **not** keys: the loader maps wkv as shard 1 of the fused
   `fused_wqa_wkv` module and w3 as shard 1 of `gate_up_proj`.
5. **`exl3_moe_plan.json`**: the resident 40-layer plan (the pack's expert layout matches it) plus
   a `drafter` section: `{"num_experts": 128, "layers": {"40": {"k": "4"×128}, "41": …, "42": …}}`
   (layers are **absolute** indices ≥ num_layers; drafter routed experts are K4).

## Step 3 — the engine image

Two layers:

- **The release recipe image** (`build_serve_image.sh`): vLLM `6e517b15` + patches 0001-0021 +
  the `vllm_exl3_moe` plugin. This is Sudosu's repo, unchanged upstream.
- **The 1017 overlay** (`overlay/`, six files, ~40 KB of diffs on top of the release image):

| file | what changed and why |
|---|---|
| `vllm_exl3_moe/config.py` | (1) `_dense_ckpt_k`: per-module K read from the safetensors *header* at method-resolution time (packs with per-layer K can't use the flat config map; dtype is `"I16"`, not `"INT16"`). (2) `mtp=` parameter so drafter modules probe `mtp.{stage}` tensors instead of colliding with the target's `layers.0-2` (same path space!). (3) compressor + indexer branches; the compressor gets a dequant method (below). (4) drafter allowlist extended to `attn.wo_a` + `main_proj`. (5) draft/backbone discriminator via the `_draft_config` tag. |
| `vllm_exl3_moe/dense.py` | `Exl3DequantDenseLinearMethod`: dequantizes EXL3 trellis to a plain bf16 `layer.weight` (`weight = (suh · H·W·H · svh)ᵀ`. **All math on CPU** — the CUDA decode path SIGSEGVs here; pieces are ~20 MB so it's free. Handles the **stacked 4-D layout** `[n, in16, out16, 16K]`, not the flat concat). Registered only for `compressor.fused_wkv_wgate`. `apply` = `F.linear` → the weight must be `[out, in]`. |
| `deepseek_v41/compressor.py` | passes the real `quant_config` into the compressor's `MergedColumnParallelLinear` (v4/v4e hard-coded `quant_config=None` = the FP8-convention pack). |
| `deepseek_v41/attention.py` | 1017 layout touch-ups for the compressor wiring. |
| `deepseek_v41/dspark.py` | the drafter's `_remap_dspark_name` maps `mtp.{i}.X` → `model.layers.{i}.X` (verified empirically — the modules *load* under the stage index even though they're constructed at `layers.{40+i}`; **do not** "fix" this offset, it is correct as-is). |
| `spec_decode/dspark/utils.py` | tags the draft's quant-config instance `_draft_config=True` — the draft config parses the same JSON as the target, nothing in the file distinguishes them; without the tag the drafter's layers collide with the target's `layers.0-2` in `get_quant_method`. |

## Step 4 — launch and calibrate

```bash
SPEC='{"method":"dspark","num_speculative_tokens":6,"draft_parallel_config":{"tensor_parallel_size":2}}' \
UTIL=0.85 TEXT_ONLY=0 \
IMAGE=dsv41-serve:v51 HEAD=10.200.0.1 WORKER=10.200.0.2 \
MODEL_DIR=/home/jaita/models/dsv41-1017-ours \
bash scripts/launch/dsv41_tp2.sh up
```

- **TP2 drafter** (`draft_parallel_config.tensor_parallel_size: 2`): the drafter's 3 MoE stages
  were serializing on one rank; TP2 is worth +20-30 tok/s at fixed depth.
- **Depth calibration** (`VLLM_DSPARK_ROUND_MS`): the launcher ships Sudosu's TP1-measured table
  `28.90,32.85,36.57,40.13,42.62,45.69`. Re-measure on your boot with `scripts/bench/depth_sweep.sh`
  (boot once with
  `DOCKER_ENV="VLLM_DSPARK_DEPTH_POLICY=fixed:5 VLLM_DSPARK_DEPTH_FILE=/root/.cache/dspark_depth VLLM_DSPARK_DEPTH_LOG=200"`),
  then set your measured table. Ours: `31.9,35.2,38.6,42.6,45.3,48.1,51.0` (depths 0-6).
  **A stale table under-speculates** — with the fresh table the adaptive policy beat every fixed
  depth on every prompt. The depth file (`~/.cache/dsv41-serve/dspark_depth`) is read every 64
  verify steps; write `conf` there (or restart with the policy env) after sweeps.
- **Benchmarks**: `python3 kit_bench.py --base http://127.0.0.1:8000 --model dsv41 decode`,
  `depth`, `concurrent`, `prefill` — greedy, streaming; this is the yardstick the community numbers
  above were measured with.
- Memory: at UTIL 0.85 the KV pool is ~711K tokens (2.7× concurrency at 262K). A clean reboot gives
  ~117 GB free → UTIL 0.88-0.885 for the 275K×4-lane goal. Boot cycles drain free memory; a
  full-node reboot + `journalctl --vacuum-size=50M` + drop-caches restores it.

## The grind (blockers in boot order — each cost a boot cycle)

1. `KeyError: 'layers.1.engram.embed.weight'` — the engine wants the Engram through the checkpoint
   index → union index (step 2.2).
2. `_dense_ckpt_k` rejected every shard — safetensors dtype is `"I16"`, the check wanted `"INT16"`.
3. `KeyError: 'compressor…'` — the pack carries compressors as EXL3 pieces, the wiring expected
   FP8 `.weight` (the `quant_config=None` convention) → pass the quant config through, add the
   dequant method.
4. `KeyError: 'indexer.wq_b'` / `.wk` — the 1017 packs quantize the indexer too → method branches.
5. `KeyError: 'model.layers.0.attn.wo_a.slice.0.mul1'` — the drafter's `wo_a` is EXL3 head-group
   slices in 1017 (was FP8 in the v2.x line) → allowlist + the map + the remap investigation.
6. `KeyError: 'model.layers.0.ffn.experts.routed_experts.w13_mul1'` — the drafter's routed experts
   need the plan's `drafter` section (128 experts, K4, absolute layer indices 40-42).
7. `'MergedColumnParallelLinear' object has no attribute 'weight'` — the first dequant attempt never
   ran: the fix had been made on spark1's copy while the overlay was built from a stale local copy.
   **Always build the overlay from one source of truth.**
8. `KeyError: 'MergedColumnParallelLinear…exl3_flat'` — the dequant must derive shapes from the
   registered params (the load truth), not from the method's bookkeeping.
9. `RuntimeError: shape '[320, 32, 80]' is invalid for input of size 1638400` — trellis arrives
   **4-D stacked** `[n, in16, out16, 16K]`, not flat-concatenated.
10. `mat1 and mat2 shapes cannot be multiplied (2048x5120 and 1024x5120)` — `F.linear` wants
    `[out, in]`; the dequant weight must be transposed at registration.
11. Hard GPU abort inside the CUDA decode in the dequant — moved the whole dequant to CPU.
12. The depth-policy table being static (his TP1 numbers) → the measured TP2 table (step 4).

## What is still open

- `data/dspark/markov_top_tokens.json` is counted on Sudosu's old-base calibration corpus; a 1017
  regeneration (needs a calibration pass over the new base) might buy the last 2-5 % of draft
  acceptance, which would show up most on prose.
- Prose ceiling ~52 tok/s is a drafter-quality property: the 1017 MTP accepts ~2 tokens per round
  on flowing text (Sudosu's old drafter: ~2.2). A deeper fix is a drafter re-fine-tune on the
  1017 base with his `p4_drafter_experts` tooling.
- A 2.7 bpw repack of this same lane would presumably beat his published numbers outright, at the
  cost of re-quantization time; the 2.9 pack trades ~9 % of decode bandwidth for ~7 % better
  perplexity on one file set.

## Credits

Built on **Sudosu's** `dgx-spark-deepseek-v41` (Apache-2.0) — engine, plugin, kernels, benchmark
client, and the tuning method are his; this document and the 1017 patches layer his recipe onto
the community 1017 pack. DeepSeek-V4.1-Flash (MIT, DeepSeek). vLLM (Apache-2.0). exllamav3
(drorlab, Apache-2.0). TensorFold's RDMA ring (Apache-2.0). The LibertAIDAI 1017 2.9bpw pack
(MIT, DeepSeek derivative).