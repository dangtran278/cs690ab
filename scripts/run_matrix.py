import argparse
import csv
import os
import random
import traceback
import uuid
from datetime import datetime, timezone

import torch
from datasets import load_dataset

from kvbench.config import KvQuantConfig
from kvbench.hf_utils import load_model_and_tokenizer, perplexity_on_tokens
from kvbench.modeling_patch import patch_hf_model_kv_cache, reset_kvbench_state
from run_passkey import build_passkey_prompt, extract_int, greedy_decode_next_tokens


RUNS_COLUMNS = [
    "run_id",
    "timestamp",
    "model",
    "method",
    "bits",
    "outlier_percent",
    "group_size",
    "residual_length",
    "first_few_fp16",
    "task",
    "context_len",
    "batch_size",
    "prefill_tokens",
    "decode_tokens",
    "seed",
    "success",
    "oom",
    "error_msg",
]

METRICS_COLUMNS = [
    "run_id",
    "metric_name",
    "metric_value",
    "unit",
    "split",
    "notes",
]


def ensure_csv(path: str, columns: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()


def append_csv(path: str, columns: list[str], row: dict) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writerow({k: row.get(k, "") for k in columns})


def method_params(method: str) -> dict:
    if method == "kivi2":
        return dict(k_bits=2, v_bits=2, nuq_bits=4, outlier_percent=0.01)
    if method == "kivi4":
        return dict(k_bits=4, v_bits=4, nuq_bits=4, outlier_percent=0.01)
    if method == "kvquant_nuq3_1p":
        return dict(k_bits=2, v_bits=2, nuq_bits=3, outlier_percent=0.01)
    if method == "kvquant_nuq4_1p":
        return dict(k_bits=2, v_bits=2, nuq_bits=4, outlier_percent=0.01)
    return dict(k_bits=2, v_bits=2, nuq_bits=4, outlier_percent=0.0)


@torch.no_grad()
def run_ppl_task(model, tok, max_tokens: int, device: str) -> float:
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    input_ids = tok(text, return_tensors="pt", truncation=True, max_length=max_tokens).input_ids.to(device)
    return float(perplexity_on_tokens(model, input_ids))


@torch.no_grad()
def run_passkey_task(model, tok, context_tokens: int, decode_tokens: int, seed: int, device: str) -> tuple[float, int, int]:
    random.seed(seed)
    torch.manual_seed(seed)
    prompt, key = build_passkey_prompt(context_tokens)
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=context_tokens).to(device)
    reset_kvbench_state(model)
    gen_ids = greedy_decode_next_tokens(model, enc["input_ids"], max_new_tokens=decode_tokens)
    gen = tok.decode(gen_ids[0].tolist(), skip_special_tokens=True)
    pred = extract_int(gen)
    return (1.0 if pred == key else 0.0), int(key), int(pred) if pred is not None else -1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--cache_dir", type=str, default=None)
    ap.add_argument("--output_dir", type=str, default="logs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--decode_tokens", type=int, default=10)
    ap.add_argument("--ppl_tokens", type=int, default=4096)
    ap.add_argument("--passkey_contexts", type=int, nargs="+", default=[2048, 4096, 8192, 16384, 32768])
    ap.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["fp16", "kivi4", "kivi2", "kvquant_nuq4_1p", "kvquant_nuq3_1p"],
    )
    ap.add_argument("--run_ppl", action="store_true", help="Run Wikitext-2 perplexity")
    ap.add_argument("--run_passkey", action="store_true", help="Run passkey retrieval")
    args = ap.parse_args()

    if not args.run_ppl and not args.run_passkey:
        args.run_ppl = True
        args.run_passkey = True

    cfg = KvQuantConfig(method="fp16", device=args.device, cache_dir=args.cache_dir)
    runs_path = os.path.join(args.output_dir, "runs.csv")
    metrics_path = os.path.join(args.output_dir, "metrics.csv")
    ensure_csv(runs_path, RUNS_COLUMNS)
    ensure_csv(metrics_path, METRICS_COLUMNS)

    for method in args.methods:
        params = method_params(method)

        # Accuracy model according to proposal.
        model_path = cfg.model_accuracy
        model, tok = load_model_and_tokenizer(
            model_path,
            device=args.device,
            cache_dir=args.cache_dir,
            use_flash_attn_2=False,
        )
        model, _ = patch_hf_model_kv_cache(
            model,
            method=method,
            k_bits=params["k_bits"],
            v_bits=params["v_bits"],
            group_size=cfg.group_size,
            residual_length=cfg.residual_length,
            nuq_bits=params["nuq_bits"],
            outlier_percent=params["outlier_percent"],
            first_few_fp16=cfg.first_few_fp16,
            use_nf=cfg.use_nf,
        )

        if args.run_ppl:
            run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
            err = ""
            success = 0
            oom = 0
            ppl = None
            try:
                ppl = run_ppl_task(model, tok, max_tokens=args.ppl_tokens, device=args.device)
                success = 1
            except RuntimeError as e:
                err = str(e)
                if "out of memory" in err.lower():
                    oom = 1
            except Exception:
                err = traceback.format_exc(limit=1).strip().replace("\n", " ")

            append_csv(
                runs_path,
                RUNS_COLUMNS,
                {
                    "run_id": run_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "model": "llama3_8b",
                    "method": method,
                    "bits": params["nuq_bits"] if "kvquant" in method else params["k_bits"],
                    "outlier_percent": params["outlier_percent"],
                    "group_size": cfg.group_size,
                    "residual_length": cfg.residual_length,
                    "first_few_fp16": cfg.first_few_fp16,
                    "task": "ppl_wikitext2",
                    "context_len": args.ppl_tokens,
                    "batch_size": 1,
                    "prefill_tokens": args.ppl_tokens,
                    "decode_tokens": 0,
                    "seed": args.seed,
                    "success": success,
                    "oom": oom,
                    "error_msg": err,
                },
            )
            if ppl is not None:
                append_csv(
                    metrics_path,
                    METRICS_COLUMNS,
                    {
                        "run_id": run_id,
                        "metric_name": "wikitext2_ppl_4k",
                        "metric_value": ppl,
                        "unit": "ppl",
                        "split": "test",
                        "notes": f"max_tokens={args.ppl_tokens}",
                    },
                )

        if args.run_passkey:
            for context_len in args.passkey_contexts:
                run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
                err = ""
                success = 0
                oom = 0
                acc = None
                key = -1
                pred = -1
                try:
                    acc, key, pred = run_passkey_task(
                        model,
                        tok,
                        context_tokens=context_len,
                        decode_tokens=args.decode_tokens,
                        seed=args.seed,
                        device=args.device,
                    )
                    success = 1
                except RuntimeError as e:
                    err = str(e)
                    if "out of memory" in err.lower():
                        oom = 1
                except Exception:
                    err = traceback.format_exc(limit=1).strip().replace("\n", " ")

                append_csv(
                    runs_path,
                    RUNS_COLUMNS,
                    {
                        "run_id": run_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "model": "llama3_8b",
                        "method": method,
                        "bits": params["nuq_bits"] if "kvquant" in method else params["k_bits"],
                        "outlier_percent": params["outlier_percent"],
                        "group_size": cfg.group_size,
                        "residual_length": cfg.residual_length,
                        "first_few_fp16": cfg.first_few_fp16,
                        "task": "passkey",
                        "context_len": context_len,
                        "batch_size": 1,
                        "prefill_tokens": context_len,
                        "decode_tokens": args.decode_tokens,
                        "seed": args.seed,
                        "success": success,
                        "oom": oom,
                        "error_msg": err,
                    },
                )
                if acc is not None:
                    append_csv(
                        metrics_path,
                        METRICS_COLUMNS,
                        {
                            "run_id": run_id,
                            "metric_name": "passkey_accuracy",
                            "metric_value": acc,
                            "unit": "ratio",
                            "split": "eval",
                            "notes": f"context_len={context_len},target={key},pred={pred}",
                        },
                    )

        # Free model between methods.
        del model
        torch.cuda.empty_cache()

    print(f"Wrote {runs_path}")
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()

