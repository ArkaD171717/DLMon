from __future__ import annotations

import copy
import math
from typing import Any, TYPE_CHECKING

import torch
from torch.utils.data import Dataset

from dlmon.invariants import CHECK_WINDOW, Violation, ViolationType

_DTYPE_PRECISION = {
    "torch.float64": 6, "torch.float32": 5, "torch.bfloat16": 4,
    "torch.float16": 3,
    "torch.int64": 6, "torch.int32": 5, "torch.int16": 3,
    "torch.int8": 2, "torch.uint8": 1,
}

_SAFE_DTYPE_TRANSFORMS = {
    "ToTensor", "ToDtype", "ConvertImageDtype", "ConvertDtype",
    "ToImage", "ToPILImage", "PILToTensor",
}

if TYPE_CHECKING:
    from dlmon.guard_state import GuardState

def _equal(a: Any, b: Any) -> tuple[bool, str]:
    try:
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            return (torch.equal(a, b), "torch.equal")
    except Exception:
        return (False, "UNCHECKED")

    try:
        import numpy as np
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            return (bool(np.array_equal(a, b)), "np.array_equal")
    except Exception:
        pass

    try:
        from PIL import Image as PILImage
        if isinstance(a, PILImage.Image) and isinstance(b, PILImage.Image):
            eq = (a.mode == b.mode and a.size == b.size and a.tobytes() == b.tobytes())
            return (eq, "PIL.tobytes")
    except Exception:
        pass

    return (False, "UNCHECKED")


def _snapshot(obj: Any) -> Any:
    try:
        if isinstance(obj, torch.Tensor):
            return obj.clone()
    except Exception:
        pass

    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.copy()
    except Exception:
        pass

    try:
        from PIL import Image as PILImage
        if isinstance(obj, PILImage.Image):
            return obj.copy()
    except Exception:
        pass

    try:
        return copy.deepcopy(obj)
    except Exception:
        return None


class MonitoredCompose:

    WARMUP = CHECK_WINDOW
    SAMPLE_RATE = 100

    def __init__(self, transforms: list, always_monitor: bool = False):
        self.transforms = transforms
        self._always_monitor = always_monitor
        self._call_count = 0
        self._step_stats: list[dict] = []
        self._dtype_truncation_fired: set[int] = set()
        for t in transforms:
            p = getattr(t, "p", None)
            self._step_stats.append({
                "name": type(t).__name__,
                "noop_count": 0,
                "total_count": 0,
                "unchecked_count": 0,
                "p": p if isinstance(p, (int, float)) else None,
            })

    def __call__(self, img: Any) -> Any:
        self._call_count += 1
        should_monitor = (
            self._always_monitor
            or self._call_count <= self.WARMUP
            or (self._call_count % self.SAMPLE_RATE == 0)
        )

        if not should_monitor:
            for t in self.transforms:
                img = t(img)
            return img

        for i, t in enumerate(self.transforms):
            before = _snapshot(img)
            before_dtype = _get_dtype(img)
            img = t(img)
            after_dtype = _get_dtype(img)
            stats = self._step_stats[i]
            stats["total_count"] += 1
            if before is None:
                stats["unchecked_count"] += 1
            else:
                eq, method = _equal(before, img)
                if method == "UNCHECKED":
                    stats["unchecked_count"] += 1
                elif eq:
                    stats["noop_count"] += 1
            if (i not in self._dtype_truncation_fired
                    and before_dtype is not None and after_dtype is not None
                    and stats["name"] not in _SAFE_DTYPE_TRANSFORMS):
                bp = _DTYPE_PRECISION.get(before_dtype, 0)
                ap = _DTYPE_PRECISION.get(after_dtype, 0)
                if bp > 0 and ap > 0 and ap < bp:
                    self._dtype_truncation_fired.add(i)
                    stats.setdefault("dtype_truncations", []).append(
                        {"from": before_dtype, "to": after_dtype})
        return img

    def get_step_reports(self) -> list[dict]:
        return [dict(s) for s in self._step_stats]


class DatasetGuard(Dataset):

    WARMUP = 64
    SAMPLE_RATE = 100

    def __init__(self, inner: Dataset, state: GuardState):
        object.__setattr__(self, "inner", inner)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "_call_count", 0)

    def __getitem__(self, idx: int) -> Any:
        raw = self.inner[idx]
        cc = self._call_count + 1
        object.__setattr__(self, "_call_count", cc)

        should_monitor = cc <= self.WARMUP or (idx % self.SAMPLE_RATE == 0)
        if should_monitor:
            self.state.on_getitem(idx, raw)
            _check_nan_inf(raw, idx, self.state)

        return raw

    def __getitems__(self, indices: list[int]) -> list[Any]:
        results = []
        for idx in indices:
            results.append(self[idx])
        return results

    def __len__(self) -> int:
        return len(self.inner)

    def __getattr__(self, name: str) -> Any:
        # Only called for attributes not on DatasetGuard itself
        return getattr(object.__getattribute__(self, "inner"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        # Forward attribute writes to inner for transparency (e.g. transform replacement)
        if name in ("inner", "state", "_call_count"):
            object.__setattr__(self, name, value)
        else:
            setattr(self.inner, name, value)


def _get_dtype(obj: Any) -> str | None:
    if isinstance(obj, torch.Tensor):
        return str(obj.dtype)
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return f"numpy.{obj.dtype}"
    except ImportError:
        pass
    return None


def _check_nan_inf(raw: Any, idx: int, state: "GuardState") -> None:
    items = raw if isinstance(raw, (tuple, list)) else (raw,)
    for i, item in enumerate(items):
        if not isinstance(item, torch.Tensor):
            continue
        has_nan = torch.isnan(item).any().item()
        has_inf = torch.isinf(item).any().item()
        if has_nan or has_inf:
            problems = []
            if has_nan:
                problems.append("NaN")
            if has_inf:
                problems.append("Inf")
            state.violations.append(Violation(
                type=ViolationType.NAN_INF_OUTPUT,
                message=f"Sample {idx} element {i} contains {'+'.join(problems)}",
                epoch=state.epoch,
                sample_ids=[idx],
                details={"idx": idx, "element": i,
                         "has_nan": has_nan, "has_inf": has_inf},
            ))
