from typing import Optional, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .modeling_patch import reset_kvbench_state


def load_model_and_tokenizer(
    model_name_or_path: str,
    *,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.float16,
    use_flash_attn_2: bool = False,
    cache_dir: Optional[str] = None,
) -> Tuple[torch.nn.Module, any]:
    config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    config.use_cache = True

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        config=config,
        torch_dtype=torch_dtype,
        device_map=None,
        cache_dir=cache_dir,
        trust_remote_code=False,
        attn_implementation=("flash_attention_2" if use_flash_attn_2 else "eager"),
    )
    tok = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, cache_dir=cache_dir)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model.to(device)
    model.eval()
    return model, tok


@torch.no_grad()
def perplexity_on_tokens(model, input_ids: torch.Tensor, *, prefill_tokens: Optional[int] = None) -> float:
    """Cache-aware streaming perplexity.

    This evaluates next-token NLL under autoregressive decoding semantics:
      - fp16/HF path uses past_key_values (use_cache=True)
      - patched kvbench path uses internal cache state with explicit position_ids
    """
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"expected input_ids shape (1, T), got {tuple(input_ids.shape)}")
    if input_ids.shape[1] < 2:
        raise ValueError("need at least 2 tokens to compute perplexity")

    def _is_kvbench_patched(m) -> bool:
        try:
            layers = getattr(getattr(m, "model", None), "layers", None)
            if not layers:
                return False
            return hasattr(layers[0].self_attn, "cache_impl")
        except Exception:
            return False

    # Ensure internal quantized cache state doesn't leak across evaluations.
    reset_kvbench_state(model)
    use_internal_cache = _is_kvbench_patched(model)
    device = input_ids.device
    total_nll = torch.zeros((), dtype=torch.float32, device=device)
    count = 0

    total_tokens = int(input_ids.shape[1])
    if prefill_tokens is None:
        prefill_len = total_tokens - 1
    else:
        prefill_len = max(2, min(int(prefill_tokens), total_tokens - 1))

    prefill_ids = input_ids[:, :prefill_len]
    if use_internal_cache:
        prefill_pos_ids = torch.arange(0, prefill_len, dtype=torch.long, device=device).unsqueeze(0)
        out = model(prefill_ids, position_ids=prefill_pos_ids, use_cache=False)
        past_key_values = None
    else:
        out = model(prefill_ids, use_cache=True)
        past_key_values = out.past_key_values

    # Prefill logits at position i predict token i+1 (including boundary token prefill_len).
    prefill_logits = out.logits.contiguous().view(-1, out.logits.shape[-1])
    prefill_targets = input_ids[:, 1 : prefill_len + 1].contiguous().view(-1)
    if prefill_targets.numel() > 0:
        nll = torch.nn.functional.cross_entropy(prefill_logits, prefill_targets, reduction="sum")
        total_nll = total_nll + nll.to(torch.float32)
        count += int(prefill_targets.numel())

    decode_pos_ids = torch.zeros((1, 1), dtype=torch.long, device=device) if use_internal_cache else None
    # Decode continuation: step t consumes token[t] and predicts token[t+1].
    for t in range(prefill_len, total_tokens - 1):
        token_t = input_ids[:, t : t + 1]
        target = input_ids[:, t + 1]
        if use_internal_cache:
            assert decode_pos_ids is not None
            decode_pos_ids[0, 0] = t
            out = model(token_t, position_ids=decode_pos_ids, use_cache=False)
        else:
            out = model(token_t, past_key_values=past_key_values, use_cache=True)
            past_key_values = out.past_key_values

        logits = out.logits[:, -1, :]
        nll = torch.nn.functional.cross_entropy(logits, target, reduction="sum")
        total_nll = total_nll + nll.to(torch.float32)
        count += 1

    mean_nll = total_nll / max(count, 1)
    ppl_t = torch.exp(mean_nll)
    ppl = float(ppl_t.item())
    if not torch.isfinite(ppl_t):
        raise RuntimeError(f"non-finite perplexity computed: {ppl}")
    return ppl

