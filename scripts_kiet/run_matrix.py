import argparse
import csv
import os
import random
import time
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
    # KiVi presets: kivi2/kivi4 use symmetric K/V quantization bit-widths.
    if method == "kivi2":
        return dict(k_bits=2, v_bits=2, nuq_bits=4, outlier_percent=0.01)
    if method == "kivi4":
        return dict(k_bits=4, v_bits=4, nuq_bits=4, outlier_percent=0.01)
    if method == "kvquant_nuq3_1p":
        return dict(k_bits=2, v_bits=2, nuq_bits=3, outlier_percent=0.01)
    if method == "kvquant_nuq4_1p":
        return dict(k_bits=2, v_bits=2, nuq_bits=4, outlier_percent=0.01)
    return dict(k_bits=2, v_bits=2, nuq_bits=4, outlier_percent=0.0)


def sync_device(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def clear_cuda_cache(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def synthetic_input_ids(tok, context_tokens: int, batch_size: int, device: str) -> torch.Tensor:
    prompt = "KV cache quantization benchmark. " * max(1, context_tokens // 8)
    prompts = [prompt] * batch_size
    enc = tok(prompts, return_tensors="pt", truncation=True, max_length=context_tokens, padding=True)
    return enc["input_ids"].to(device)


@torch.no_grad()
def run_system_profile_task(
    model,
    tok,
    context_tokens: int,
    decode_tokens: int,
    batch_size: int,
    device: str,
) -> dict[str, float]:
    input_ids = synthetic_input_ids(tok, context_tokens=context_tokens, batch_size=batch_size, device=device)
    reset_kvbench_state(model)

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    bsz, prefill_len = input_ids.shape

    # Quantized backends use AttentionCacheAdapter with internal state; HF past_key_values is None.
    # Match run_passkey greedy_decode: use_cache=False + absolute position_ids on decode steps.
    sync_device(device)
    t0 = time.perf_counter()
    out = model(input_ids, use_cache=False)
    sync_device(device)
    prefill_s = time.perf_counter() - t0

    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    sync_device(device)
    t1 = time.perf_counter()
    for i in range(decode_tokens):
        token_pos = prefill_len + i
        pos_ids = torch.full((bsz, 1), token_pos, dtype=torch.long, device=device)
        out = model(next_token, position_ids=pos_ids, use_cache=False)
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    sync_device(device)
    decode_s = time.perf_counter() - t1

    peak_vram_mib = 0.0
    if device.startswith("cuda") and torch.cuda.is_available():
        peak_vram_mib = torch.cuda.max_memory_allocated() / (1024 * 1024)

    total_new_tokens = batch_size * decode_tokens
    decode_ms_per_token = 0.0
    decode_tokens_per_s = 0.0
    if decode_s > 0 and total_new_tokens > 0:
        decode_ms_per_token = (decode_s * 1000.0) / total_new_tokens
        decode_tokens_per_s = total_new_tokens / decode_s

    return {
        "peak_vram_mib": float(peak_vram_mib),
        "prefill_latency_s": float(prefill_s),
        "decode_latency_s": float(decode_s),
        "decode_ms_per_token": float(decode_ms_per_token),
        "decode_tokens_per_s": float(decode_tokens_per_s),
    }


@torch.no_grad()
def probe_max_batch_size(
    model,
    tok,
    context_tokens: int,
    decode_tokens: int,
    max_batch_cap: int,
    device: str,
) -> int:
    def fits(batch_size: int) -> bool:
        try:
            run_system_profile_task(
                model,
                tok,
                context_tokens=context_tokens,
                decode_tokens=decode_tokens,
                batch_size=batch_size,
                device=device,
            )
            return True
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                clear_cuda_cache(device)
                return False
            raise

    lo, hi = 0, 1
    while hi <= max_batch_cap and fits(hi):
        lo = hi
        hi *= 2
    hi = min(hi, max_batch_cap)

    while lo < hi:
        mid = (lo + hi + 1) // 2
        if fits(mid):
            lo = mid
        else:
            hi = mid - 1

    return lo


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
        # KiVi variants are included here, then selected in the loop below.
        default=["fp16", "kivi4", "kivi2", "kvquant_nuq4_1p", "kvquant_nuq3_1p"],
    )
    ap.add_argument("--run_ppl", action="store_true", help="Run Wikitext-2 perplexity")
    ap.add_argument("--run_passkey", action="store_true", help="Run passkey retrieval")
    system_group = ap.add_mutually_exclusive_group()
    system_group.add_argument(
        "--run_system",
        action="store_true",
        dest="run_system",
        help="Run system-side profiling (peak VRAM, decode latency, throughput; default: on)",
    )
    system_group.add_argument(
        "--no_run_system",
        action="store_false",
        dest="run_system",
        help="Skip system-side profiling",
    )
    ap.set_defaults(run_system=True)
    ap.add_argument("--system_contexts", type=int, nargs="+", default=[2048], help="Contexts for system profiling")
    ap.add_argument("--system_batch_sizes", type=int, nargs="+", default=[1], help="Batch sizes for system profiling")
    ap.add_argument("--system_decode_tokens", type=int, default=32, help="Decode tokens for system profiling")
    ap.add_argument("--probe_max_batch", action="store_true", help="Probe OOM-bounded max batch size")
    ap.add_argument(
        "--profile_max_batch",
        action="store_true",
        help="When probing max batch, also log peak VRAM/throughput at that batch",
    )
    ap.add_argument("--max_batch_cap", type=int, default=32, help="Upper cap for max-batch probing")
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
        # Main KiVi/KVQuant implementation point:
        # this patches the HF model's KV cache with the selected quantization method.
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

        if args.run_system:
            for context_len in args.system_contexts:
                for batch_size in args.system_batch_sizes:
                    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
                    err = ""
                    success = 0
                    oom = 0
                    profile = None
                    try:
                        profile = run_system_profile_task(
                            model,
                            tok,
                            context_tokens=context_len,
                            decode_tokens=args.system_decode_tokens,
                            batch_size=batch_size,
                            device=args.device,
                        )
                        success = 1
                    except RuntimeError as e:
                        err = str(e)
                        if "out of memory" in err.lower():
                            oom = 1
                            clear_cuda_cache(args.device)
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
                            "task": "system_profile",
                            "context_len": context_len,
                            "batch_size": batch_size,
                            "prefill_tokens": context_len,
                            "decode_tokens": args.system_decode_tokens,
                            "seed": args.seed,
                            "success": success,
                            "oom": oom,
                            "error_msg": err,
                        },
                    )

                    if profile is not None:
                        for metric_name, metric_value, unit in [
                            ("peak_vram", profile["peak_vram_mib"], "MiB"),
                            ("prefill_latency", profile["prefill_latency_s"], "s"),
                            ("decode_latency", profile["decode_latency_s"], "s"),
                            ("decode_ms_per_token", profile["decode_ms_per_token"], "ms/token"),
                            ("decode_tokens_per_s", profile["decode_tokens_per_s"], "token/s"),
                        ]:
                            append_csv(
                                metrics_path,
                                METRICS_COLUMNS,
                                {
                                    "run_id": run_id,
                                    "metric_name": metric_name,
                                    "metric_value": metric_value,
                                    "unit": unit,
                                    "split": "eval",
                                    "notes": f"context_len={context_len},batch_size={batch_size}",
                                },
                            )

                if args.probe_max_batch:
                    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
                    err = ""
                    success = 0
                    oom = 0
                    max_batch = None
                    try:
                        max_batch = probe_max_batch_size(
                            model,
                            tok,
                            context_tokens=context_len,
                            decode_tokens=max(1, min(args.system_decode_tokens, 8)),
                            max_batch_cap=args.max_batch_cap,
                            device=args.device,
                        )
                        success = 1
                    except RuntimeError as e:
                        err = str(e)
                        if "out of memory" in err.lower():
                            oom = 1
                            clear_cuda_cache(args.device)
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
                            "task": "max_batch_probe",
                            "context_len": context_len,
                            "batch_size": 0,
                            "prefill_tokens": context_len,
                            "decode_tokens": max(1, min(args.system_decode_tokens, 8)),
                            "seed": args.seed,
                            "success": success,
                            "oom": oom,
                            "error_msg": err,
                        },
                    )
                    if max_batch is not None:
                        append_csv(
                            metrics_path,
                            METRICS_COLUMNS,
                            {
                                "run_id": run_id,
                                "metric_name": "max_batch_size",
                                "metric_value": max_batch,
                                "unit": "batch",
                                "split": "eval",
                                "notes": f"context_len={context_len},cap={args.max_batch_cap}",
                            },
                        )
                        if args.profile_max_batch and max_batch > 0:
                            prof_run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
                            prof = run_system_profile_task(
                                model,
                                tok,
                                context_tokens=context_len,
                                decode_tokens=max(1, min(args.system_decode_tokens, 8)),
                                batch_size=max_batch,
                                device=args.device,
                            )
                            append_csv(
                                runs_path,
                                RUNS_COLUMNS,
                                {
                                    "run_id": prof_run_id,
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "model": "llama3_8b",
                                    "method": method,
                                    "bits": params["nuq_bits"] if "kvquant" in method else params["k_bits"],
                                    "outlier_percent": params["outlier_percent"],
                                    "group_size": cfg.group_size,
                                    "residual_length": cfg.residual_length,
                                    "first_few_fp16": cfg.first_few_fp16,
                                    "task": "max_batch_profile",
                                    "context_len": context_len,
                                    "batch_size": max_batch,
                                    "prefill_tokens": context_len,
                                    "decode_tokens": max(1, min(args.system_decode_tokens, 8)),
                                    "seed": args.seed,
                                    "success": 1,
                                    "oom": 0,
                                    "error_msg": "",
                                },
                            )
                            for metric_name, metric_value, unit in [
                                ("max_batch_peak_vram", prof["peak_vram_mib"], "MiB"),
                                ("max_batch_decode_tokens_per_s", prof["decode_tokens_per_s"], "token/s"),
                                ("max_batch_decode_ms_per_token", prof["decode_ms_per_token"], "ms/token"),
                                ("max_batch_prefill_latency", prof["prefill_latency_s"], "s"),
                                ("max_batch_decode_latency", prof["decode_latency_s"], "s"),
                            ]:
                                append_csv(
                                    metrics_path,
                                    METRICS_COLUMNS,
                                    {
                                        "run_id": prof_run_id,
                                        "metric_name": metric_name,
                                        "metric_value": metric_value,
                                        "unit": unit,
                                        "split": "eval",
                                        "notes": f"context_len={context_len},batch_size={max_batch}",
                                    },
                                )

        # Free model between methods.
        del model
        clear_cuda_cache(args.device)

    print(f"Wrote {runs_path}")
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()

