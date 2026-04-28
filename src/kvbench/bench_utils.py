import time
from typing import Callable

import torch


def _to_device_obj(device: str | torch.device) -> torch.device:
    return device if isinstance(device, torch.device) else torch.device(device)


def reset_cuda_peak_stats(device: str | torch.device) -> None:
    dev = _to_device_obj(device)
    if dev.type != "cuda":
        return
    torch.cuda.reset_peak_memory_stats(dev)


def cuda_peak_memory_gb(device: str | torch.device) -> dict[str, float]:
    dev = _to_device_obj(device)
    if dev.type != "cuda":
        return {"peak_allocated_gb": 0.0, "peak_reserved_gb": 0.0}
    gb = float(1024**3)
    return {
        "peak_allocated_gb": float(torch.cuda.max_memory_allocated(dev) / gb),
        "peak_reserved_gb": float(torch.cuda.max_memory_reserved(dev) / gb),
    }


def time_cuda_callable_ms(fn: Callable[[], None], *, device: str | torch.device, steps: int = 1, warmup: int = 0) -> float:
    dev = _to_device_obj(device)
    for _ in range(max(0, warmup)):
        fn()
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    for _ in range(max(1, steps)):
        fn()
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t1 = time.perf_counter()
    return float((t1 - t0) * 1000.0 / max(1, steps))

