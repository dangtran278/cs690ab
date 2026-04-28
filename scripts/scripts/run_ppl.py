import argparse
import sys
from pathlib import Path

import torch
from datasets import load_dataset

# Prefer this repo's `src/kvbench` when running this script directly.
_repo_src = Path(__file__).resolve().parents[1] / "src"
if _repo_src.is_dir():
    _src = str(_repo_src)
    if _src not in sys.path:
        sys.path.insert(0, _src)

from kvbench.hf_utils import load_model_and_tokenizer, perplexity_on_tokens
from kvbench.kivi_cache import validate_kivi_bits
from kvbench.modeling_patch import patch_hf_model_kv_cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--method", type=str, default="fp16", choices=["fp16", "kivi2", "kivi4", "kvquant_nuq3_1p", "kvquant_nuq4_1p"])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--ppl_prefill_tokens", type=int, default=None, help="Prefill tokens before decode streaming (default: max_tokens-1)")
    ap.add_argument("--cache_dir", type=str, default=None)

    # KIVI-style params
    ap.add_argument("--k_bits", type=int, default=2)
    ap.add_argument("--v_bits", type=int, default=2)
    ap.add_argument("--group_size", type=int, default=32)
    ap.add_argument("--residual_length", type=int, default=128)
    ap.add_argument("--k_residual_length", type=int, default=None)
    ap.add_argument("--v_residual_length", type=int, default=None)
    ap.add_argument("--kivi_mode", type=str, default="legacy", choices=["legacy", "official_like"])
    ap.add_argument("--kivi_diagnostics", action="store_true")
    ap.add_argument("--kivi_parity_checks", action="store_true")
    ap.add_argument("--kivi_drift_probe_interval", type=int, default=0)

    # KVQuant-style params
    ap.add_argument("--nuq_bits", type=int, default=4)
    ap.add_argument("--outlier_percent", type=float, default=0.01)
    ap.add_argument("--first_few_fp16", type=int, default=0)
    ap.add_argument("--use_nf", action="store_true")

    args = ap.parse_args()

    # Match harness behavior: kivi2/kivi4 imply symmetric K/V bit-widths unless user overrides.
    user_set_k_bits = "--k_bits" in sys.argv
    user_set_v_bits = "--v_bits" in sys.argv
    if args.method == "kivi4" and not user_set_k_bits and not user_set_v_bits:
        args.k_bits = 4
        args.v_bits = 4
    if args.method == "kivi2" and not user_set_k_bits and not user_set_v_bits:
        args.k_bits = 2
        args.v_bits = 2
    if args.method.startswith("kivi"):
        validate_kivi_bits(args.k_bits, args.v_bits)
    if args.method != "fp16":
        print("info: running cache-aware streaming perplexity for quantized KV cache evaluation.")

    model, tok = load_model_and_tokenizer(args.model, device=args.device, cache_dir=args.cache_dir, use_flash_attn_2=False)
    model, _ = patch_hf_model_kv_cache(
        model,
        method=args.method,
        k_bits=args.k_bits,
        v_bits=args.v_bits,
        group_size=args.group_size,
        residual_length=args.residual_length,
        k_residual_length=args.k_residual_length,
        v_residual_length=args.v_residual_length,
        nuq_bits=args.nuq_bits,
        outlier_percent=args.outlier_percent,
        first_few_fp16=args.first_few_fp16,
        use_nf=args.use_nf,
        kivi_mode=args.kivi_mode,
        kivi_diagnostics=args.kivi_diagnostics,
        kivi_parity_checks=args.kivi_parity_checks,
        kivi_drift_probe_interval=args.kivi_drift_probe_interval,
    )

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    input_ids = tok(text, return_tensors="pt", truncation=True, max_length=args.max_tokens).input_ids.to(args.device)

    ppl = perplexity_on_tokens(model, input_ids, prefill_tokens=args.ppl_prefill_tokens)
    print(f"ppl={ppl:.4f}")


if __name__ == "__main__":
    main()

