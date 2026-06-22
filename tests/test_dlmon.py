import warnings

import pytest
import torch
import torch.nn as nn
from torch.utils.data import (
    DataLoader, Dataset, IterableDataset,
    RandomSampler, Sampler, TensorDataset,
)

from dlmon import DLMonitor
from dlmon.invariants import (
    BatchDistribution, EpochCompleteness, CompletenessMode,
    StochasticDiversity, TransformApplied, ViolationType,
)


class SimpleDataset(Dataset):
    def __init__(self, n=200, dim=10, num_classes=3):
        self.x = torch.randn(n, dim)
        self.targets = [i % num_classes for i in range(n)]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.targets[i]


class DuplicateSampler(Sampler):
    def __init__(self, n, num_dupes=15):
        self.n = n
        self.num_dupes = num_dupes

    def __iter__(self):
        indices = list(range(self.n))
        indices[:self.num_dupes] = indices[:self.num_dupes] + indices[:self.num_dupes]
        return iter(indices[:self.n])

    def __len__(self):
        return self.n


class FixedOrderSampler(Sampler):
    def __init__(self, n):
        self.n = n

    def __iter__(self):
        return iter(range(self.n))

    def __len__(self):
        return self.n


def test_detect_duplicate_samples():
    ds = SimpleDataset(100)
    loader = DataLoader(ds, batch_size=16, sampler=DuplicateSampler(100, num_dupes=10))
    monitor = DLMonitor()
    monitor.guard(loader, role="train")
    for batch in loader:
        pass
    dupes = [v for v in monitor.get_violations() if v.type == ViolationType.DUPLICATE_SAMPLE]
    assert len(dupes) >= 5


def test_completeness_catches_missing_samples():
    comp = EpochCompleteness(dataset_size=100, mode=CompletenessMode.EXACT_SET,
                             expected_set=set(range(100)))
    comp.start_epoch(1)
    for i in range(0, 100, 2):
        comp.record_sample(i)
    vs = comp.end_epoch()
    missing = [v for v in vs if v.type == ViolationType.MISSING_SAMPLE]
    assert len(missing) == 1
    assert missing[0].details["missing_count"] == 50


def test_detect_partition_overlap():
    monitor = DLMonitor()
    monitor.register_partition("train", list(range(0, 80)))
    monitor.register_partition("val", list(range(70, 100)))
    leaks = [v for v in monitor.check_partitions() if v.type == ViolationType.PARTITION_OVERLAP]
    assert len(leaks) == 1
    assert leaks[0].details["overlap_count"] == 10


def test_detect_no_shuffle():
    ds = SimpleDataset(100)
    gen = torch.Generator()
    loader = DataLoader(ds, batch_size=16, shuffle=True, generator=gen)
    monitor = DLMonitor()
    monitor.guard(loader, role="train")
    for epoch in range(3):
        gen.manual_seed(1234)
        for batch in loader:
            pass
    bugs = [v for v in monitor.get_violations() if v.type == ViolationType.NO_SHUFFLE]
    assert len(bugs) >= 1


def test_no_false_shuffle_on_custom_sampler():
    ds = SimpleDataset(100)
    loader = DataLoader(ds, batch_size=16, sampler=FixedOrderSampler(100))
    monitor = DLMonitor()
    monitor.guard(loader, role="train")
    for _ in range(3):
        for batch in loader:
            pass
    assert len([v for v in monitor.get_violations() if v.type == ViolationType.NO_SHUFFLE]) == 0


def test_detect_transform_noop():
    checker = TransformApplied(noop_threshold=0.9)
    checker.start_epoch(0)
    for _ in range(100):
        data = torch.randn(3, 32, 32)
        checker.check("identity", data, data)
    noops = [v for v in checker.end_epoch() if v.type == ViolationType.TRANSFORM_NOOP]
    assert len(noops) == 1


def test_detect_frozen_stochastic():
    checker = StochasticDiversity(min_unique_ratio=0.1)
    checker.start_epoch(0)
    fixed = torch.randn(3, 32, 32)
    for _ in range(50):
        checker.record_output(fixed)
    frozen = [v for v in checker.end_epoch() if v.type == ViolationType.FROZEN_STOCHASTIC]
    assert len(frozen) == 1


def test_no_false_positive_on_diverse_outputs():
    checker = StochasticDiversity(min_unique_ratio=0.1)
    checker.start_epoch(0)
    for _ in range(50):
        checker.record_output(torch.randn(3, 32, 32))
    assert len([v for v in checker.end_epoch() if v.type == ViolationType.FROZEN_STOCHASTIC]) == 0


def test_batch_distribution_anomaly():
    checker = BatchDistribution()
    checker.start_epoch(0)
    for i in range(20):
        checker.observe("class_0", 48.0 + 4.0 * (i % 3), i)
    checker.start_epoch(1)
    assert len(checker.observe("class_0", 50.0, 0)) == 0
    assert len(checker.observe("class_0", 95.0, 1)) >= 1


def test_integration_clean_pipeline():
    torch.manual_seed(42)
    ds = SimpleDataset(200, dim=10, num_classes=2)
    loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(42))
    model = nn.Linear(10, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    monitor = DLMonitor()
    monitor.guard(loader, role="train")
    for epoch in range(3):
        for x, y in loader:
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()

    assert len(monitor.get_violations()) == 0, "false positives on clean pipeline"


def test_summary():
    ds = SimpleDataset(200)
    loader = DataLoader(ds, batch_size=16, sampler=RandomSampler(ds))
    monitor = DLMonitor()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        monitor.guard(loader)
    s = monitor.summary()
    assert "DLMon summary" in s
    assert "dataset_monitoring=" in s


def test_status_fields():
    loader = DataLoader(SimpleDataset(64), batch_size=16)
    monitor = DLMonitor()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        monitor.guard(loader)
    st = list(monitor.status().values())[0]
    assert "role" in st
    assert "completeness_mode" in st
    assert "shuffle_detector_active" in st
    assert "dataset_monitoring_status" in st


def test_two_loaders():
    ds = SimpleDataset(200)
    train = DataLoader(ds, batch_size=16, sampler=RandomSampler(ds))
    val = DataLoader(ds, batch_size=16, shuffle=False)
    monitor = DLMonitor()
    monitor.guard(train, role="train")
    monitor.guard(val, role="val")
    status = monitor.status()
    assert len(status) == 2
    roles = {st["role"] for st in status.values()}
    assert roles == {"train", "val"}


def test_drop_last_no_false_positive():
    ds = SimpleDataset(100)
    loader = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True)
    monitor = DLMonitor()
    monitor.guard(loader, role="train")
    for batch in loader:
        pass
    missing = [v for v in monitor.get_violations() if v.type == ViolationType.MISSING_SAMPLE]
    assert len(missing) == 0
