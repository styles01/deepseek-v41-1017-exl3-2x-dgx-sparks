# DeepSeek-V4.1-Flash-**1017** EXL3 2.9bpw on two DGX Sparks — one button, Sudosu's tuned engine

Two NVIDIA DGX Spark GB10 nodes serving the **October 17 1017 refresh** of DeepSeek-V4.1-Flash
(community EXL3 2.9bpw pack + the fresh 1017 Engram tables) through the tuned engine built for
this recipe: vLLM + the EXL3 MoE / RDMA-ring / L2-prefetch plugin, DSpark speculative decode
with a **TP2 drafter** and a **measured per-round depth policy**, 262K context, vision on.

Measured with the included `kit_bench.py` (greedy, streaming, single stream):

| prompt (set b) | this lane | Sudosu's 2.70bpw lane | Mia-AI-Lab stock kit |
|---|---|---|---|
| code | 93.8 | 102.7 | 41.6 |
| prose | 52.4 | 57.9 | 25.1 |
| structured | 126.8 | 141.0 | 40.4 |

(The −9-10 % vs the 2.70bpw lane = weight bytes per token: the 2.9bpw pack reads 7.4 % more weight
per token. Per byte moved, parity. The 100 tok/s line = the **structured** set, as for every
published lane; flowing prose is capped by drafter acceptance, a model property.)

## One button

```bash
cd dgx-spark-deepseek-v41
MODEL_ROOT=/home/jaita bash one-button.sh
```

It pulls the prebuilt engine image from GHCR, downloads the pack (210.8 GB, MIT,
[LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw))
and the **fresh 1017 Engram shards 47+48** (~203 GB, from `deepseek-ai/DeepSeek-V4.1-Flash-1017` —
the older V4.1 tables are poison for this pack), repacks (hardlinks, union index, config/plan
rewrites) into `dsv41-1017-ours/`, and boots both ranks on :8000. ~430 GB NVMe and `hf` needed.
Needs the two-node wiring the recipe assumes (RoCE fabric, see `RECIPE.md`).

Then verify the way the numbers above were measured:

```bash
python3 kit_bench.py --base http://127.0.0.1:8000 --model dsv41 decode
bash scripts/bench/depth_sweep.sh ...   # re-measure the round-cost table on your boot (see METHOD-1017.md)
```

**How it works / what had to change:** [`METHOD-1017.md`](METHOD-1017.md) — the full method, every
load-path blocker and its fix, the depth-sweep tables, and what is still open. Sudosu's original
README is kept as [`README-sudoingx-v1.1.md`](README-sudoingx-v1.1.md); his repo + engine + benchmark
client are the base of all of this (Apache-2.0), with our diffs layered in
`patches/deepseek_v41/` and `patches/exl3_moe/{config,dense}.py`.

## Status / notes

- Vision: `TEXT_ONLY=0` serves the V4.1 vision tower (verified on a color test image).
- Draft depth: the launcher defaults to the measured table; re-sweep it on your hardware.
- `data/dspark/markov_top_tokens.json` still carries the old base's calibration counts
  (harmless, possibly worth regenerating).
- Not affiliated with DeepSeek / NVIDIA; weights keep their upstream licenses
  (DeepSeek MIT; see `README-sudoingx-v1.1.md` credits for the rest).

## License

Apache-2.0 (this repo's code, like the base recipe). The published engine image = the same terms;
model weights stay under their upstream licenses (DeepSeek MIT for the 1017 base and the pack
derivative) and are downloaded from their own repos — this repo ships code and configs only.