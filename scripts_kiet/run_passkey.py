import argparse
import random
import re

import torch

from kvbench.hf_utils import load_model_and_tokenizer
from kvbench.kivi_cache import validate_kivi_bits
from kvbench.modeling_patch import patch_hf_model_kv_cache, reset_kvbench_state


def _is_kvbench_patched(model) -> bool:
    try:
        layers = getattr(getattr(model, "model", None), "layers", None)
        if not layers:
            return False
        return hasattr(layers[0].self_attn, "cache_impl")
    except Exception:
        return False


def build_passkey_prompt(max_tokens: int) -> tuple[str, int]:
    # Rough token budgeting by repetition; tokenizer-specific, but adequate for benchmarking.
    garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.\n"
    pass_key = random.randint(1, 50000)
    info = f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key.\n"
    question = "What is the pass key? The pass key is "

    # Construct prompt with info in the middle
    n_reps = max(1, max_tokens // 50)
    prefix = garbage * (n_reps // 2)
    suffix = garbage * (n_reps - n_reps // 2)
    prompt = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you.\n"
    prompt += prefix + info + suffix + question
    return prompt, pass_key


def extract_int(s: str):
    m = re.search(r"\d+", s)
    return int(m.group()) if m else None


def extract_passkey_prediction(text: str):
    """Robustly parse passkey answer from generated text."""
    # Prefer explicit "pass key is <digits>" anchors if present.
    anchored = re.findall(r"pass\s*key\s*is\s*(\d+)", text, flags=re.IGNORECASE)
    if anchored:
        return int(anchored[-1])

    # Fallback: choose the last integer token in the output.
    ints = re.findall(r"\d+", text)
    if ints:
        return int(ints[-1])
    return None


def parser_self_check() -> None:
    cases = [
        ("The pass key is 25248", 25248),
        ("... pass key is 252 and then 48", 252),
        ("I think 9999", 9999),
        ("no digits here", None),
    ]
    for text, expected in cases:
        got = extract_passkey_prediction(text)
        if got != expected:
            raise AssertionError(f"parser_self_check failed: text={text!r} expected={expected!r} got={got!r}")


@torch.no_grad()
def greedy_decode_next_tokens(
    model,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
):
    """Greedy decode using our internal KV-cache adapter (no HF past_key_values)."""
    device = input_ids.device
    bsz, prefill_len = input_ids.shape
    assert bsz == 1, "This benchmark script currently assumes batch size 1."

    use_internal_cache = _is_kvbench_patched(model)
    past_key_values = None

    # Prefill: compute logits for the next token.
    if use_internal_cache:
        out = model(input_ids, use_cache=False)
    else:
        out = model(input_ids, use_cache=True)
        past_key_values = out.past_key_values
    next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)  # (1, 1)

    generated = []
    cur_len = prefill_len
    for i in range(max_new_tokens):
        generated.append(next_token)

        # Decode one token at a time.
        if use_internal_cache:
            token_pos = cur_len + i
            pos_ids = torch.tensor([[token_pos]], dtype=torch.long, device=device)
            out = model(next_token, position_ids=pos_ids, use_cache=False)
        else:
            out = model(next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = out.past_key_values
        next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

    return torch.cat(generated, dim=1)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self_check", action="store_true", help="Run parser self-check and exit")
    ap.add_argument("--model", type=str, default=None, help="HF model name or path (not needed for --self_check)")
    ap.add_argument("--method", type=str, default="fp16", choices=["fp16", "kivi2", "kivi4", "kvquant_nuq3_1p", "kvquant_nuq4_1p"])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--context_tokens", type=int, default=8192)
    ap.add_argument("--max_new_tokens", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache_dir", type=str, default=None)

    ap.add_argument("--k_bits", type=int, default=2)
    ap.add_argument("--v_bits", type=int, default=2)
    ap.add_argument("--group_size", type=int, default=32)
    ap.add_argument("--residual_length", type=int, default=128)

    ap.add_argument("--nuq_bits", type=int, default=4)
    ap.add_argument("--outlier_percent", type=float, default=0.01)
    ap.add_argument("--first_few_fp16", type=int, default=0)
    ap.add_argument("--use_nf", action="store_true")

    args = ap.parse_args()
    if args.self_check:
        parser_self_check()
        print("parser_self_check=ok")
        return
    if not args.model:
        ap.error("the following arguments are required: --model (unless using --self_check)")
    if args.method.startswith("kivi"):
        validate_kivi_bits(args.k_bits, args.v_bits)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, tok = load_model_and_tokenizer(args.model, device=args.device, cache_dir=args.cache_dir, use_flash_attn_2=False)
    model, _ = patch_hf_model_kv_cache(
        model,
        method=args.method,
        k_bits=args.k_bits,
        v_bits=args.v_bits,
        group_size=args.group_size,
        residual_length=args.residual_length,
        nuq_bits=args.nuq_bits,
        outlier_percent=args.outlier_percent,
        first_few_fp16=args.first_few_fp16,
        use_nf=args.use_nf,
    )

    prompt, key = build_passkey_prompt(args.context_tokens)
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=args.context_tokens).to(args.device)

    # Reset KV states for the new prompt.
    reset_kvbench_state(model)

    gen_ids = greedy_decode_next_tokens(model, enc["input_ids"], max_new_tokens=args.max_new_tokens)
    gen = tok.decode(gen_ids[0].tolist(), skip_special_tokens=True)
    pred = extract_passkey_prediction(gen)
    ok = pred == key
    print(f"target={key} pred={pred} ok={ok} gen={gen!r}")


if __name__ == "__main__":
    main()

