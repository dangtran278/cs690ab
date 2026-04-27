import argparse
import csv
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Prefer this repo's `src/kvbench` when running this script directly.
_repo_src = Path(__file__).resolve().parents[1] / "src"
if _repo_src.is_dir():
    _src = str(_repo_src)
    if _src not in sys.path:
        sys.path.insert(0, _src)

from kvbench.config import KvQuantConfig


@dataclass
class AblationCase:
    name: str
    method: str
    k_bits: int
    v_bits: int
    residual_length: int
    k_residual_length: Optional[int] = None
    v_residual_length: Optional[int] = None
    prefill_tokens: Optional[int] = None
    kivi_mode: str = "legacy"
    group_size: int = 32


def parse_ppl(stdout: str) -> Optional[float]:
    m = re.search(r"ppl=([0-9]+(?:\.[0-9]+)?)", stdout)
    if not m:
        return None
    return float(m.group(1))


def run_case(
    repo_root: Path,
    model: str,
    device: str,
    max_tokens: int,
    cache_dir: Optional[str],
    case: AblationCase,
) -> dict:
    cmd = [
        sys.executable,
        "scripts/run_ppl.py",
        "--model",
        model,
        "--method",
        case.method,
        "--device",
        device,
        "--max_tokens",
        str(max_tokens),
        "--k_bits",
        str(case.k_bits),
        "--v_bits",
        str(case.v_bits),
        "--group_size",
        str(case.group_size),
        "--residual_length",
        str(case.residual_length),
        "--k_residual_length",
        str(case.k_residual_length if case.k_residual_length is not None else case.residual_length),
        "--v_residual_length",
        str(case.v_residual_length if case.v_residual_length is not None else case.residual_length),
        "--kivi_mode",
        str(case.kivi_mode),
    ]
    if case.prefill_tokens is not None:
        cmd.extend(["--ppl_prefill_tokens", str(case.prefill_tokens)])
    if cache_dir:
        cmd.extend(["--cache_dir", cache_dir])

    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    ppl = parse_ppl(out)
    return {
        "name": case.name,
        "method": case.method,
        "k_bits": case.k_bits,
        "v_bits": case.v_bits,
        "residual_length": case.residual_length,
        "k_residual_length": case.k_residual_length if case.k_residual_length is not None else case.residual_length,
        "v_residual_length": case.v_residual_length if case.v_residual_length is not None else case.residual_length,
        "prefill_tokens": case.prefill_tokens if case.prefill_tokens is not None else max_tokens - 1,
        "kivi_mode": case.kivi_mode,
        "group_size": case.group_size,
        "success": int(proc.returncode == 0 and ppl is not None),
        "return_code": proc.returncode,
        "ppl": ppl if ppl is not None else "",
        "raw_output": out.strip().replace("\n", " | "),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "name",
        "method",
        "k_bits",
        "v_bits",
        "residual_length",
        "k_residual_length",
        "v_residual_length",
        "prefill_tokens",
        "kivi_mode",
        "group_size",
        "success",
        "return_code",
        "ppl",
        "raw_output",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_md(path: Path, rows: list[dict], *, max_tokens: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    baseline = next((r for r in rows if r["name"] == "fp16_baseline"), None)
    baseline_ppl = float(baseline["ppl"]) if baseline and baseline["ppl"] != "" else None

    lines = [
        "# KIVI Ablation Debug Report",
        "",
        "This report isolates whether PPL degradation is primarily from K quantization, V quantization, or residual flush policy.",
        "",
        "## Results",
        "",
        "| Case | Method | k_bits | v_bits | residual_length | k_residual_length | v_residual_length | prefill_tokens | decode_tokens | kivi_mode | success | PPL |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for r in rows:
        ppl_str = f'{float(r["ppl"]):.4f}' if r["ppl"] != "" else "NA"
        prefill = int(r["prefill_tokens"])
        decode = max_tokens - prefill
        lines.append(
            f'| {r["name"]} | {r["method"]} | {r["k_bits"]} | {r["v_bits"]} | {r["residual_length"]} | {r["k_residual_length"]} | {r["v_residual_length"]} | {prefill} | {decode} | {r["kivi_mode"]} | {r["success"]} | {ppl_str} |'
        )

    lines.extend(["", "## Interpretation", ""])
    if baseline_ppl is None:
        lines.append("- Baseline fp16 failed, so interpretation is inconclusive.")
    else:
        lines.append(f"- fp16 baseline PPL: `{baseline_ppl:.4f}`")
        lines.append("- Compare each ablation against fp16 and `kivi4_default` to identify dominant error source.")
        lines.append("- If `k8_v8_largeR` is close to fp16, adapter path is likely healthy and quantization config/math is the main issue.")
        lines.append("- If `k8_v4_defaultR` is bad but `k4_v8_defaultR` is good, V quantization dominates error; inverse means K dominates.")
        lines.append("- If larger `residual_length` improves sharply, flush policy aggressiveness is likely the main factor.")
        lines.append("- If `kivi4` and `kivi2` both diverge as decode length grows, error likely accumulates in repeated cache materialize/quantize cycles.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--output_dir", type=str, default="logs_kivi_debug")
    ap.add_argument("--cache_dir", type=str, default=None)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--ppl_prefill_tokens", type=int, default=None, help="Prefill tokens for all cases (default: max_tokens-1)")
    ap.add_argument("--residual_values", type=int, nargs="+", default=[128, 256, 512, 1024])
    ap.add_argument("--kivi_mode", type=str, default="legacy", choices=["legacy", "official_like"])
    ap.add_argument("--k_residual_length", type=int, default=None)
    ap.add_argument("--v_residual_length", type=int, default=None)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cfg = KvQuantConfig(method="fp16", device=args.device, cache_dir=args.cache_dir)
    model = args.model or cfg.model_accuracy

    prefill_tokens = args.ppl_prefill_tokens
    default_r = args.residual_values[0]
    cases = [
        AblationCase("fp16_baseline", "fp16", 2, 2, default_r, k_residual_length=args.k_residual_length, v_residual_length=args.v_residual_length, prefill_tokens=prefill_tokens, kivi_mode=args.kivi_mode),
        AblationCase("kivi4_default", "kivi4", 4, 4, default_r, k_residual_length=args.k_residual_length, v_residual_length=args.v_residual_length, prefill_tokens=prefill_tokens, kivi_mode=args.kivi_mode),
        AblationCase("kivi2_default", "kivi2", 2, 2, default_r, k_residual_length=args.k_residual_length, v_residual_length=args.v_residual_length, prefill_tokens=prefill_tokens, kivi_mode=args.kivi_mode),
        AblationCase("k8_v8_largeR", "kivi4", 8, 8, max(4096, default_r), k_residual_length=args.k_residual_length, v_residual_length=args.v_residual_length, prefill_tokens=prefill_tokens, kivi_mode=args.kivi_mode),
        AblationCase("k8_v4_defaultR", "kivi4", 8, 4, default_r, k_residual_length=args.k_residual_length, v_residual_length=args.v_residual_length, prefill_tokens=prefill_tokens, kivi_mode=args.kivi_mode),
        AblationCase("k4_v8_defaultR", "kivi4", 4, 8, default_r, k_residual_length=args.k_residual_length, v_residual_length=args.v_residual_length, prefill_tokens=prefill_tokens, kivi_mode=args.kivi_mode),
    ]
    for r in args.residual_values:
        cases.append(AblationCase(f"kivi4_R{r}", "kivi4", 4, 4, r, k_residual_length=args.k_residual_length, v_residual_length=args.v_residual_length, prefill_tokens=prefill_tokens, kivi_mode=args.kivi_mode))
        cases.append(AblationCase(f"kivi2_R{r}", "kivi2", 2, 2, r, k_residual_length=args.k_residual_length, v_residual_length=args.v_residual_length, prefill_tokens=prefill_tokens, kivi_mode=args.kivi_mode))

    rows = []
    for case in cases:
        print(f"running: {case.name}")
        rows.append(
            run_case(
                repo_root=repo_root,
                model=model,
                device=args.device,
                max_tokens=args.max_tokens,
                cache_dir=args.cache_dir,
                case=case,
            )
        )

    out_dir = repo_root / args.output_dir
    csv_path = out_dir / "kivi_ablation_results.csv"
    md_path = out_dir / "kivi_ablation_report.md"
    write_csv(csv_path, rows)
    write_md(md_path, rows, max_tokens=args.max_tokens)
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
