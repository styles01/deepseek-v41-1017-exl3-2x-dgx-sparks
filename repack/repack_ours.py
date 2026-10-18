#!/usr/bin/env python3
"""Repack the LibertAIDAI DeepSeek-V4.1-Flash-1017 EXL3 2.9bpw pack into this recipe's serving format.

Builds a hardlink tree (zero-copy), appends the Engram shards to a union weight index,
rewrites config.json / exl3_moe_plan.json for the deepseek_v41_exl3 quant method. stdlib only.

  python3 repack_ours.py --pack <pack-dir> --engram <engram-dir> --out <serving-dir>

  pack   = the 43-shard LibertAIDAI download (config.json + shards + tokenizer + index)
  engram = the fresh 1017 download of shards 47+48 + its index + config (old tables poison!)
  out    = the directory the launcher's MODEL_DIR points at (all three on the same filesystem)
"""
import argparse, json, os, re, sys

DRAFTER_DENSE_K = {
    "attn.wq_a": 6, "attn.wq_b": 6, "attn.wo_b": 6, "attn.wo_a": 6, "main_proj": 6,
    "ffn.shared_experts.w1": 6, "ffn.shared_experts.w2": 6,
}
DRAFTER_STAGES = 3
DRAFTER_EXPERTS = 128
DRAFTER_DRAFT_K = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--engram", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pack, engram, out = a.pack, a.engram, a.out

    idx = json.load(open(f"{pack}/model.safetensors.index.json"))
    wm = idx["weight_map"]
    print(f"pack: {len(wm)} tensors across {len(set(wm.values()))} shards")

    os.makedirs(out, exist_ok=True)

    # 1. hardlink the pack files (zero-copy; du may report the clone but nlink shares bytes)
    for name in set(wm.values()) | {f for f in os.listdir(pack) if not f.endswith(".safetensors")}:
        src = os.path.join(pack, name)
        if not os.path.exists(src):
            print(f"WARN missing {name}", file=sys.stderr)
            continue
        dst = os.path.join(out, name)
        if os.path.exists(dst):
            os.unlink(dst)
        os.link(src, dst)

    # 2. hardlink the engram files
    eidx = json.load(open(f"{engram}/model.safetensors.index.json"))
    ewm = eidx["weight_map"]
    for name in set(ewm.values()) | {f for f in os.listdir(engram) if f.endswith((".json",))}:
        src = os.path.join(engram, name)
        if not os.path.exists(src):
            print(f"WARN engram missing {name}", file=sys.stderr)
            continue
        dst = os.path.join(out, name)
        if os.path.exists(dst):
            os.unlink(dst)
        os.link(src, dst)

    nhl = int((json.load(open(f"{out}/config.json")).get("text_config", {})
               or json.load(open(f"{out}/config.json"))).get("num_hidden_layers") or 40)
    print(f"num_hidden_layers = {nhl}")

    # 3. union weight index: the pack's rows + the engram entries the loader reads the
    #    tables through. Engram names in the base index are model.-prefixed; the serving
    #    convention is bare layers.N.* entries.
    eng = {n: ewm[n] for n in ewm if re.search(r"engram\.embed\.(weight|scale)$", n)}
    for entry, shard in eng.items():
        key = re.sub(r"^model\.", "", entry)
        wm[key] = shard
        print(f"index: {key} -> {shard}")
    if not eng:
        sys.exit("FATAL: no engram.embed tensors in the engram index — is this the 1017 base dir?")
    json.dump({"metadata": idx.get("metadata", {}) | {"repacked": "1017->serving"},
               "weight_map": wm}, open(f"{out}/model.safetensors.index.json", "w"))

    # 4. config.json: quant method + the drafter dense-K map.
    #    NOTE: 'attn.wkv' and 'ffn.shared_experts.w3' are NOT keys here — the loader routes
    #    wkv as shard 1 of the fused wq_a module and w3 as shard 1 of gate_up_proj.
    cfg = json.load(open(f"{out}/config.json"))
    q = cfg["quantization_config"]
    q["quant_method"] = "deepseek_v41_exl3"
    q.setdefault("expert_dtype", "fp4")
    ex = q.setdefault("exl3_moe", {})
    ex["plan_file"] = ex.get("plan_file", "exl3_moe_plan.json")
    ex["decode_kernel"] = ex.get("decode_kernel", "coop")
    ex["fused_rows"] = ex.get("fused_rows", 256)
    ex["tp_layout"] = ex.get("tp_layout", "channels")
    ex["head"] = ex.get("head", "exl3")
    ex["head_k"] = 6
    ex["exllamav3_version"] = ex.get("exllamav3_version", "1.6.0")
    d = ex.setdefault("dense", {})
    d.setdefault("layers", [0, nhl - 1])
    d.setdefault("k", {"attn.wo_a": 5, "attn.wo_b": 5, "attn.wq_a": 5, "attn.wq_b": 5,
                       "ffn.shared_experts.w1": 5, "ffn.shared_experts.w2": 5})
    d["drafter"] = dict(DRAFTER_DENSE_K)
    json.dump(cfg, open(f"{out}/config.json", "w"), indent=2)
    print("config.json: quant_method=deepseek_v41_exl3 + drafter dense map written")

    # 5. plan: the pack's own plan (or a uniform fallback) + the drafter section. Drafter
    #    layers are ABSOLUTE indices >= num_layers (= num_hidden_layers + stage).
    if os.path.exists(f"{out}/exl3_moe_plan.json"):
        plan = json.load(open(f"{out}/exl3_moe_plan.json"))
    elif os.path.exists(f"{pack}/exl3_moe_plan.json"):
        plan = json.load(open(f"{pack}/exl3_moe_plan.json"))
    else:
        plan = {"format": "exl3_moe_plan_v1", "num_layers": nhl, "num_experts": 384,
                "hidden_size": 5120, "moe_intermediate_size": 2304,
                "layers": {str(L): {"k": "4" * 384} for L in range(nhl)}}
        print("WARN no plan in the pack; wrote a uniform fallback (prefer a calibrated plan)",
              file=sys.stderr)
    plan.setdefault("hidden_size", 5120)
    plan.setdefault("num_layers", nhl)
    plan["drafter"] = {"num_experts": DRAFTER_EXPERTS,
                       "layers": {str(nhl + s): {"k": str(DRAFTER_DRAFT_K) * DRAFTER_EXPERTS}
                                  for s in range(DRAFTER_STAGES)}}
    json.dump(plan, open(f"{out}/exl3_moe_plan.json", "w"), indent=1)
    print(f"plan: {len(plan['layers'])} backbone layers + drafter "
          f"{list(plan['drafter']['layers'])} ({DRAFTER_EXPERTS} experts @ K{DRAFTER_DRAFT_K})")

    # 6. verify: every index target exists on disk, nonzero
    missing = [n for n in wm
               if not (os.path.exists(os.path.join(out, wm[n])) and os.path.getsize(os.path.join(out, wm[n])) > 0)]
    print("VERIFY:", "OK — all index targets present" if not missing else f"MISSING {missing[:5]}",
          file=sys.stderr)
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()