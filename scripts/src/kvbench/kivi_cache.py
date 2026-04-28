from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple

import torch

from .quant_utils import (
    AffineQuantParams,
    affine_dequantize_per_group_last_dim,
    affine_dequantize_per_group_token_dim,
    affine_quantize_per_group_last_dim,
    affine_quantize_per_group_token_dim,
)

ALLOWED_KIVI_BITS = (2, 4, 8)


def validate_kivi_bits(k_bits: int, v_bits: int) -> None:
    k_bits = int(k_bits)
    v_bits = int(v_bits)
    if k_bits not in ALLOWED_KIVI_BITS or v_bits not in ALLOWED_KIVI_BITS:
        raise ValueError(
            "KIVI only supports k_bits/v_bits in {2, 4, 8}; "
            f"got k_bits={k_bits}, v_bits={v_bits}"
        )


@dataclass
class KiviCacheState:
    # Quantized long-term storage
    k_q: Optional[torch.Tensor] = None  # (b, kvh, t, d) uint8/int
    k_q_len: int = 0
    k_params: Optional[AffineQuantParams] = None
    k_param_len: int = 0
    v_q: Optional[torch.Tensor] = None  # (b, kvh, t, d)
    v_q_len: int = 0
    v_params: Optional[AffineQuantParams] = None
    v_param_len: int = 0

    # Full-precision residual window
    k_fp: Optional[torch.Tensor] = None  # (b, kvh, cap, d)
    v_fp: Optional[torch.Tensor] = None  # (b, kvh, cap, d)
    fp_start: int = 0
    fp_len: int = 0

    # Total tokens seen (for bookkeeping)
    total_len: int = 0
    # Cached dequantized quantized-prefix for official_like mode.
    cache_k_deq_prefix: Optional[torch.Tensor] = None
    cache_v_deq_prefix: Optional[torch.Tensor] = None
    cache_q_len: int = 0
    cache_dtype: Optional[torch.dtype] = None
    # Counters and diagnostics metadata.
    quant_append_count: int = 0
    total_flushed_tokens: int = 0
    last_flush_len: int = 0
    # Lightweight telemetry buffer (optional consumers can read/reset externally).
    telemetry: list[dict[str, Any]] = field(default_factory=list)


class KiviCache:
    """KIVI-style KV-cache quantization.

    - Store most tokens quantized with per-group affine params
    - Keep a fp16 residual window of the most recent `residual_length` tokens

    Quantization axes (paper): K groups along *token* dim (stats across tokens per
    channel); V groups along *head_dim* (stats across channels per token). See
    _flush_if_full and materialize — K must use affine_*_token_dim, V uses
    affine_*_per_group_last_dim on (..., t, d).
    """

    def __init__(
        self,
        *,
        k_bits: int,
        v_bits: int,
        group_size: int,
        residual_length: int,
        k_residual_length: Optional[int] = None,
        v_residual_length: Optional[int] = None,
        kivi_mode: str = "legacy",
        diagnostics: bool = False,
        telemetry_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    ):
        self.k_bits = int(k_bits)
        self.v_bits = int(v_bits)
        validate_kivi_bits(self.k_bits, self.v_bits)
        self.group_size = int(group_size)
        self.residual_length = int(residual_length)
        self.k_residual_length = int(k_residual_length) if k_residual_length is not None else self.residual_length
        self.v_residual_length = int(v_residual_length) if v_residual_length is not None else self.residual_length
        self.kivi_mode = str(kivi_mode)
        self.diagnostics = bool(diagnostics)
        self.telemetry_callback = telemetry_callback
        if self.kivi_mode not in ("legacy", "official_like"):
            raise ValueError(f"unknown kivi_mode={self.kivi_mode}")
        if self.residual_length % self.group_size != 0:
            raise ValueError(
                f"residual_length {self.residual_length} must be divisible by group_size {self.group_size} "
                "for strict queue flushing on token axis"
            )
        if self.k_residual_length % self.group_size != 0:
            raise ValueError(
                f"k_residual_length {self.k_residual_length} must be divisible by group_size {self.group_size}"
            )
        if self.v_residual_length <= 0:
            raise ValueError("v_residual_length must be positive")

    def init_state(self) -> KiviCacheState:
        return KiviCacheState()

    def _invalidate_prefix_cache(self, state: KiviCacheState) -> None:
        state.cache_k_deq_prefix = None
        state.cache_v_deq_prefix = None
        state.cache_q_len = 0
        state.cache_dtype = None

    def _emit_telemetry(self, state: KiviCacheState, payload: dict[str, Any]) -> None:
        enriched = {
            "total_len": int(state.total_len),
            "k_q_len": int(state.k_q_len),
            "v_q_len": int(state.v_q_len),
            "k_fp_len": int(state.fp_len),
            "v_fp_len": int(state.fp_len),
            **payload,
        }
        state.telemetry.append(enriched)
        if self.telemetry_callback is not None:
            self.telemetry_callback(enriched)

    def _record_parity_warning(self, state: KiviCacheState, kind: str, details: dict[str, Any]) -> None:
        self._emit_telemetry(state, {"event": "parity_warning", "kind": kind, **details})

    def _quant_error_stats(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        params: AffineQuantParams,
        *,
        token_axis: bool,
    ) -> dict[str, float]:
        if token_axis:
            deq = affine_dequantize_per_group_token_dim(
                q, params, self.group_size, out_dtype=x.dtype, diagnostics=self.diagnostics
            )
        else:
            deq = affine_dequantize_per_group_last_dim(
                q, params, self.group_size, out_dtype=x.dtype, diagnostics=self.diagnostics
            )
        err = (deq.to(torch.float32) - x.to(torch.float32)).abs()
        return {"mean_abs_err": float(err.mean().item()), "max_abs_err": float(err.max().item())}

    def _fp_compact_if_needed(self, state: KiviCacheState) -> None:
        if state.k_fp is None or state.v_fp is None or state.fp_start == 0:
            return
        end = state.fp_start + state.fp_len
        state.k_fp[..., : state.fp_len, :].copy_(state.k_fp[..., state.fp_start:end, :])
        state.v_fp[..., : state.fp_len, :].copy_(state.v_fp[..., state.fp_start:end, :])
        state.fp_start = 0

    def _ensure_fp_capacity(self, state: KiviCacheState, k: torch.Tensor, v: torch.Tensor, needed_len: int) -> None:
        if state.k_fp is None or state.v_fp is None:
            cap = max(needed_len, self.residual_length + self.group_size)
            state.k_fp = torch.empty(*k.shape[:2], cap, k.shape[-1], dtype=k.dtype, device=k.device)
            state.v_fp = torch.empty(*v.shape[:2], cap, v.shape[-1], dtype=v.dtype, device=v.device)
            state.fp_start = 0
            state.fp_len = 0
            return
        self._fp_compact_if_needed(state)
        cur_cap = int(state.k_fp.shape[-2])
        if needed_len <= cur_cap:
            return
        new_cap = cur_cap
        while new_cap < needed_len:
            new_cap *= 2
        k_new = torch.empty(*state.k_fp.shape[:2], new_cap, state.k_fp.shape[-1], dtype=state.k_fp.dtype, device=state.k_fp.device)
        v_new = torch.empty(*state.v_fp.shape[:2], new_cap, state.v_fp.shape[-1], dtype=state.v_fp.dtype, device=state.v_fp.device)
        if state.fp_len > 0:
            k_new[..., : state.fp_len, :].copy_(state.k_fp[..., : state.fp_len, :])
            v_new[..., : state.fp_len, :].copy_(state.v_fp[..., : state.fp_len, :])
        state.k_fp = k_new
        state.v_fp = v_new
        state.fp_start = 0

    def _fp_view(self, state: KiviCacheState) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if state.k_fp is None or state.v_fp is None or state.fp_len == 0:
            return None, None
        s = state.fp_start
        e = s + state.fp_len
        return state.k_fp[..., s:e, :], state.v_fp[..., s:e, :]

    def _reserve_quant_storage(self, state: KiviCacheState, k_q_new: torch.Tensor, v_q_new: torch.Tensor, k_p_new: AffineQuantParams, v_p_new: AffineQuantParams) -> None:
        needed_q = state.k_q_len + int(k_q_new.shape[-2])
        needed_kp = state.k_param_len + int(k_p_new.scale.shape[-2])
        needed_vp = state.v_param_len + int(v_p_new.scale.shape[-3])
        if state.k_q is None:
            cap_q = max(needed_q, self.residual_length + self.group_size)
            cap_kp = max(needed_kp, max(1, cap_q // self.group_size))
            cap_vp = max(needed_vp, cap_q)
            state.k_q = torch.empty(*k_q_new.shape[:2], cap_q, k_q_new.shape[-1], dtype=k_q_new.dtype, device=k_q_new.device)
            state.v_q = torch.empty(*v_q_new.shape[:2], cap_q, v_q_new.shape[-1], dtype=v_q_new.dtype, device=v_q_new.device)
            state.k_params = AffineQuantParams(
                scale=torch.empty(*k_p_new.scale.shape[:-2], cap_kp, 1, dtype=k_p_new.scale.dtype, device=k_p_new.scale.device),
                zero_point=torch.empty(*k_p_new.zero_point.shape[:-2], cap_kp, 1, dtype=k_p_new.zero_point.dtype, device=k_p_new.zero_point.device),
                qmin=k_p_new.qmin,
                qmax=k_p_new.qmax,
            )
            state.v_params = AffineQuantParams(
                scale=torch.empty(*v_p_new.scale.shape[:-3], cap_vp, *v_p_new.scale.shape[-2:], dtype=v_p_new.scale.dtype, device=v_p_new.scale.device),
                zero_point=torch.empty(*v_p_new.zero_point.shape[:-3], cap_vp, *v_p_new.zero_point.shape[-2:], dtype=v_p_new.zero_point.dtype, device=v_p_new.zero_point.device),
                qmin=v_p_new.qmin,
                qmax=v_p_new.qmax,
            )
            return
        assert state.k_q is not None and state.v_q is not None and state.k_params is not None and state.v_params is not None
        q_cap = int(state.k_q.shape[-2])
        if needed_q > q_cap:
            new_q_cap = q_cap
            while new_q_cap < needed_q:
                new_q_cap *= 2
            k_q_buf = torch.empty(*state.k_q.shape[:2], new_q_cap, state.k_q.shape[-1], dtype=state.k_q.dtype, device=state.k_q.device)
            v_q_buf = torch.empty(*state.v_q.shape[:2], new_q_cap, state.v_q.shape[-1], dtype=state.v_q.dtype, device=state.v_q.device)
            if state.k_q_len > 0:
                k_q_buf[..., : state.k_q_len, :].copy_(state.k_q[..., : state.k_q_len, :])
                v_q_buf[..., : state.v_q_len, :].copy_(state.v_q[..., : state.v_q_len, :])
            state.k_q = k_q_buf
            state.v_q = v_q_buf
        kp_cap = int(state.k_params.scale.shape[-2])
        if needed_kp > kp_cap:
            new_kp_cap = kp_cap
            while new_kp_cap < needed_kp:
                new_kp_cap *= 2
            k_scale = torch.empty(*state.k_params.scale.shape[:-2], new_kp_cap, 1, dtype=state.k_params.scale.dtype, device=state.k_params.scale.device)
            k_zero = torch.empty(*state.k_params.zero_point.shape[:-2], new_kp_cap, 1, dtype=state.k_params.zero_point.dtype, device=state.k_params.zero_point.device)
            if state.k_param_len > 0:
                k_scale[..., : state.k_param_len, :].copy_(state.k_params.scale[..., : state.k_param_len, :])
                k_zero[..., : state.k_param_len, :].copy_(state.k_params.zero_point[..., : state.k_param_len, :])
            state.k_params.scale = k_scale
            state.k_params.zero_point = k_zero
        vp_cap = int(state.v_params.scale.shape[-3])
        if needed_vp > vp_cap:
            new_vp_cap = vp_cap
            while new_vp_cap < needed_vp:
                new_vp_cap *= 2
            v_scale = torch.empty(*state.v_params.scale.shape[:-3], new_vp_cap, *state.v_params.scale.shape[-2:], dtype=state.v_params.scale.dtype, device=state.v_params.scale.device)
            v_zero = torch.empty(*state.v_params.zero_point.shape[:-3], new_vp_cap, *state.v_params.zero_point.shape[-2:], dtype=state.v_params.zero_point.dtype, device=state.v_params.zero_point.device)
            if state.v_param_len > 0:
                v_scale[..., : state.v_param_len, :, :].copy_(state.v_params.scale[..., : state.v_param_len, :, :])
                v_zero[..., : state.v_param_len, :, :].copy_(state.v_params.zero_point[..., : state.v_param_len, :, :])
            state.v_params.scale = v_scale
            state.v_params.zero_point = v_zero

    def _append_quantized(self, state: KiviCacheState, k_flush: torch.Tensor, v_flush: torch.Tensor) -> None:
        k_q_new, k_p_new = affine_quantize_per_group_token_dim(
            k_flush, bits=self.k_bits, group_size=self.group_size, diagnostics=self.diagnostics
        )
        v_q_new, v_p_new = affine_quantize_per_group_last_dim(
            v_flush, bits=self.v_bits, group_size=self.group_size, diagnostics=self.diagnostics
        )
        if self.diagnostics:
            k_stats = self._quant_error_stats(k_flush, k_q_new, k_p_new, token_axis=True)
            v_stats = self._quant_error_stats(v_flush, v_q_new, v_p_new, token_axis=False)
        else:
            k_stats = {}
            v_stats = {}
        self._reserve_quant_storage(state, k_q_new, v_q_new, k_p_new, v_p_new)
        assert state.k_q is not None and state.v_q is not None and state.k_params is not None and state.v_params is not None
        q_s = state.k_q_len
        q_e = q_s + int(k_q_new.shape[-2])
        state.k_q[..., q_s:q_e, :].copy_(k_q_new)
        state.v_q[..., q_s:q_e, :].copy_(v_q_new)
        kp_s = state.k_param_len
        kp_e = kp_s + int(k_p_new.scale.shape[-2])
        state.k_params.scale[..., kp_s:kp_e, :].copy_(k_p_new.scale)
        state.k_params.zero_point[..., kp_s:kp_e, :].copy_(k_p_new.zero_point)
        vp_s = state.v_param_len
        vp_e = vp_s + int(v_p_new.scale.shape[-3])
        state.v_params.scale[..., vp_s:vp_e, :, :].copy_(v_p_new.scale)
        state.v_params.zero_point[..., vp_s:vp_e, :, :].copy_(v_p_new.zero_point)
        state.k_q_len = q_e
        state.v_q_len = q_e
        state.k_param_len = kp_e
        state.v_param_len = vp_e
        state.quant_append_count += 1
        state.total_flushed_tokens += int(k_flush.shape[-2])
        state.last_flush_len = int(k_flush.shape[-2])
        if state.cache_dtype is not None:
            k_deq_new = affine_dequantize_per_group_token_dim(
                k_q_new, k_p_new, self.group_size, out_dtype=state.cache_dtype, diagnostics=False
            )
            v_deq_new = affine_dequantize_per_group_last_dim(
                v_q_new, v_p_new, self.group_size, out_dtype=state.cache_dtype, diagnostics=False
            )
            if state.cache_k_deq_prefix is None:
                state.cache_k_deq_prefix = k_deq_new
                state.cache_v_deq_prefix = v_deq_new
            else:
                state.cache_k_deq_prefix = torch.cat([state.cache_k_deq_prefix, k_deq_new], dim=-2)
                state.cache_v_deq_prefix = torch.cat([state.cache_v_deq_prefix, v_deq_new], dim=-2)
            state.cache_q_len = state.k_q_len
        self._emit_telemetry(
            state,
            {
                "event": "quant_append",
                "flush_len": int(k_flush.shape[-2]),
                "k_mean_abs_err": k_stats.get("mean_abs_err"),
                "k_max_abs_err": k_stats.get("max_abs_err"),
                "v_mean_abs_err": v_stats.get("mean_abs_err"),
                "v_max_abs_err": v_stats.get("max_abs_err"),
            },
        )

    def _prefill_partition_official_like(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        t = int(k.shape[-2])
        # Official-like target:
        #   K: quantized prefix in residual blocks + fp remainder
        #   V: quantized prefix [:-R_v] + fp tail [-R_v:]
        k_quant_len = (t // self.k_residual_length) * self.k_residual_length
        v_quant_len = max(0, t - self.v_residual_length)
        # To keep K/V quantized prefix aligned in this cache format, use common prefix.
        common_quant_len = min(k_quant_len, v_quant_len)
        if common_quant_len > 0:
            common_quant_len = (common_quant_len // self.group_size) * self.group_size
        k_q = k[..., :common_quant_len, :] if common_quant_len > 0 else None
        k_fp = k[..., common_quant_len:, :]
        v_q = v[..., :common_quant_len, :] if common_quant_len > 0 else None
        v_fp = v[..., common_quant_len:, :]
        return k_q, k_fp, v_q, v_fp

    def _flush_if_full_legacy(self, state: KiviCacheState, *, out_dtype: torch.dtype) -> None:
        if state.k_fp is None or state.v_fp is None or state.fp_len == 0:
            return
        # Strict paper queue semantics: when residual reaches R, flush exactly R tokens
        # to quant storage and reset the fp residual (keeping only the remainder < R).
        while state.fp_len >= self.k_residual_length:
            s = state.fp_start
            e = s + self.k_residual_length
            k_flush = state.k_fp[..., s:e, :]
            v_flush = state.v_fp[..., s:e, :]
            state.fp_start += self.k_residual_length
            state.fp_len -= self.k_residual_length
            if state.fp_len == 0:
                state.fp_start = 0
            self._append_quantized(state, k_flush, v_flush)

    def _flush_if_full_official_like(self, state: KiviCacheState, *, out_dtype: torch.dtype) -> None:
        if state.k_fp is None or state.v_fp is None or state.fp_len == 0:
            return
        # Keep sliding fp tails while respecting group constraints.
        # For asymmetric residual settings, enforce both bounds by flushing enough
        # grouped tokens so neither K nor V tail drifts unbounded.
        while state.fp_len > 0:
            k_len = int(state.fp_len)
            v_len = int(state.fp_len)
            if k_len != v_len:
                raise RuntimeError(f"K/V fp tail length mismatch: k_len={k_len} v_len={v_len}")
            if k_len <= self.k_residual_length and v_len <= self.v_residual_length:
                break
            overflow_k = max(0, k_len - self.k_residual_length)
            overflow_v = max(0, v_len - self.v_residual_length)
            # Flush enough to satisfy the tighter side; group alignment is applied below.
            overflow = max(overflow_k, overflow_v)
            flush_len = (overflow // self.group_size) * self.group_size
            if flush_len <= 0:
                # If either side violates bound but cannot flush due to group granularity,
                # keep warning telemetry and stop; next append may make it flushable.
                if overflow_k > 0:
                    self._record_parity_warning(
                        state,
                        "k_tail_pending_group_flush",
                        {"k_len": k_len, "k_residual_length": self.k_residual_length, "overflow_k": overflow_k},
                    )
                if overflow_v > 0:
                    self._record_parity_warning(
                        state,
                        "v_tail_pending_group_flush",
                        {"v_len": v_len, "v_residual_length": self.v_residual_length, "overflow_v": overflow_v},
                    )
                break

            s = state.fp_start
            e = s + flush_len
            k_flush = state.k_fp[..., s:e, :]
            v_flush = state.v_fp[..., s:e, :]
            state.fp_start += flush_len
            state.fp_len -= flush_len
            if state.fp_len == 0:
                state.fp_start = 0
            self._append_quantized(state, k_flush, v_flush)

        if state.fp_len > (self.k_residual_length + self.group_size - 1):
            self._record_parity_warning(
                state,
                "k_tail_bound",
                {
                    "k_fp_len": int(state.fp_len),
                    "k_tail_bound": int(self.k_residual_length + self.group_size - 1),
                },
            )
        if state.fp_len > (self.v_residual_length + self.group_size - 1):
            self._record_parity_warning(
                state,
                "v_tail_bound",
                {
                    "v_fp_len": int(state.fp_len),
                    "v_tail_bound": int(self.v_residual_length + self.group_size - 1),
                },
            )

    def _flush_if_full(self, state: KiviCacheState, *, out_dtype: torch.dtype) -> None:
        if self.kivi_mode == "legacy":
            self._flush_if_full_legacy(state, out_dtype=out_dtype)
        else:
            self._flush_if_full_official_like(state, out_dtype=out_dtype)

    def append(self, state: KiviCacheState, k: torch.Tensor, v: torch.Tensor) -> KiviCacheState:
        """Append new kv for current step.

        k, v: (b, kvh, t_new, d)
        """
        if k.ndim != 4 or v.ndim != 4:
            raise ValueError(f"expected 4D KV tensors, got k={tuple(k.shape)} v={tuple(v.shape)}")
        if k.shape != v.shape:
            raise ValueError(f"K/V shape mismatch: k={tuple(k.shape)} v={tuple(v.shape)}")
        if k.shape[-2] <= 0:
            raise ValueError("append expects at least one token on KV token axis")
        needed = state.fp_len + int(k.shape[-2])
        self._ensure_fp_capacity(state, k, v, needed)
        assert state.k_fp is not None and state.v_fp is not None
        self._fp_compact_if_needed(state)
        s = state.fp_len
        e = s + int(k.shape[-2])
        state.k_fp[..., s:e, :].copy_(k)
        state.v_fp[..., s:e, :].copy_(v)
        state.fp_len = e

        state.total_len += k.shape[-2]
        self._flush_if_full(state, out_dtype=k.dtype)
        if state.k_q is not None and state.v_q is not None and int(state.k_q_len) != int(state.v_q_len):
            self._record_parity_warning(
                state,
                "kv_q_len_mismatch",
                {"k_q_len": int(state.k_q_len), "v_q_len": int(state.v_q_len)},
            )
        return state

    def append_prefill_storage(self, state: KiviCacheState, k: torch.Tensor, v: torch.Tensor) -> KiviCacheState:
        """Storage-only append used after fp16 prefill compute."""
        if self.kivi_mode == "official_like":
            k_q, k_fp, v_q, v_fp = self._prefill_partition_official_like(k, v)
            prefill_quant_len = int(k_q.shape[-2]) if k_q is not None else 0
            if k_q is not None and v_q is not None:
                self._append_quantized(state, k_q, v_q)
            if int(k_fp.shape[-2]) > 0:
                needed = state.fp_len + int(k_fp.shape[-2])
                self._ensure_fp_capacity(state, k_fp, v_fp, needed)
                assert state.k_fp is not None and state.v_fp is not None
                self._fp_compact_if_needed(state)
                s = state.fp_len
                e = s + int(k_fp.shape[-2])
                state.k_fp[..., s:e, :].copy_(k_fp)
                state.v_fp[..., s:e, :].copy_(v_fp)
                state.fp_len = e
            state.total_len += int(k.shape[-2])
            self._emit_telemetry(
                state,
                {
                    "event": "prefill_partition",
                    "prefill_len": int(k.shape[-2]),
                    "prefill_quant_len": prefill_quant_len,
                    "prefill_fp_len": int(state.fp_len),
                },
            )
            if state.k_q is not None and state.v_q is not None and int(state.k_q_len) != int(state.v_q_len):
                self._record_parity_warning(
                    state,
                    "kv_q_len_mismatch_after_prefill",
                    {"k_q_len": int(state.k_q_len), "v_q_len": int(state.v_q_len)},
                )
            return state
        return self.append(state, k, v)

    def materialize(self, state: KiviCacheState, *, out_dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return full (K,V) in out_dtype for attention.

        Shapes:
          K: (b, kvh, t_total, d)
          V: (b, kvh, t_total, d)
        """
        parts_k = []
        parts_v = []
        if state.k_q is not None and state.k_q_len > 0:
            assert state.k_params is not None and state.v_q is not None and state.v_params is not None
            q_len = int(state.k_q_len)
            can_reuse = (
                state.cache_k_deq_prefix is not None
                and state.cache_v_deq_prefix is not None
                and state.cache_q_len == q_len
                and state.cache_dtype == out_dtype
            )
            if can_reuse:
                k_deq = state.cache_k_deq_prefix
                v_deq = state.cache_v_deq_prefix
            else:
                k_q_view = state.k_q[..., : state.k_q_len, :]
                v_q_view = state.v_q[..., : state.v_q_len, :]
                k_params = AffineQuantParams(
                    scale=state.k_params.scale[..., : state.k_param_len, :],
                    zero_point=state.k_params.zero_point[..., : state.k_param_len, :],
                    qmin=state.k_params.qmin,
                    qmax=state.k_params.qmax,
                )
                v_params = AffineQuantParams(
                    scale=state.v_params.scale[..., : state.v_param_len, :, :],
                    zero_point=state.v_params.zero_point[..., : state.v_param_len, :, :],
                    qmin=state.v_params.qmin,
                    qmax=state.v_params.qmax,
                )
                # Must mirror _flush_if_full: K token-axis dequant, V last-dim dequant.
                k_deq = affine_dequantize_per_group_token_dim(
                    k_q_view, k_params, self.group_size, out_dtype=out_dtype, diagnostics=self.diagnostics
                )
                v_deq = affine_dequantize_per_group_last_dim(
                    v_q_view, v_params, self.group_size, out_dtype=out_dtype, diagnostics=self.diagnostics
                )
                state.cache_k_deq_prefix = k_deq
                state.cache_v_deq_prefix = v_deq
                state.cache_q_len = q_len
                state.cache_dtype = out_dtype
            parts_k.append(k_deq)
            parts_v.append(v_deq)
        k_fp_view, v_fp_view = self._fp_view(state)
        if k_fp_view is not None and v_fp_view is not None:
            parts_k.append(k_fp_view.to(out_dtype))
            parts_v.append(v_fp_view.to(out_dtype))
        if not parts_k:
            raise RuntimeError("cache is empty")
        return torch.cat(parts_k, dim=-2), torch.cat(parts_v, dim=-2)

