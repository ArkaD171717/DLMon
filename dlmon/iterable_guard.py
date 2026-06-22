from __future__ import annotations

from typing import Any, Iterator, TYPE_CHECKING

import xxhash
from torch.utils.data import IterableDataset

if TYPE_CHECKING:
    from dlmon.guard_state import GuardState


def _content_bytes(item: Any) -> bytes | None:
    import torch

    if isinstance(item, (tuple, list)) and len(item) >= 2:
        x = item[0]
    else:
        x = item

    if isinstance(x, torch.Tensor):
        try:
            return x.detach().cpu().numpy().tobytes()
        except Exception:
            return None

    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            return x.tobytes()
    except ImportError:
        pass

    if isinstance(x, bytes):
        return x

    return None


class IterableGuard(IterableDataset):

    def __init__(
        self,
        inner: IterableDataset,
        state: GuardState,
        dup_threshold: float = 0.05,
        min_samples: int = 1000,
    ):
        self.inner = inner
        self.state = state
        self.dup_threshold = dup_threshold
        self.min_samples = min_samples

    def __iter__(self) -> Iterator:
        self.state.on_epoch_boundary()
        seen: set[int] = set()
        dups = 0
        n = 0
        for item in self.inner:
            b = _content_bytes(item)
            if b is not None:
                h = xxhash.xxh64_intdigest(b)
                if h in seen:
                    dups += 1
                seen.add(h)
            n += 1
            yield item
        self.state.on_iterable_epoch_end(n, dups)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)
