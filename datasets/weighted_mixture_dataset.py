"""按源权重从多个子数据集中随机抽样（有放回），供多数据集训练与标准 ``DataLoader`` + ``Accelerator.prepare`` 共用。"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset


class WeightedMixtureDataset(Dataset):
    """对若干子数据集按 **源级** 归一化权重抽样源，再在源内均匀抽样一条样本。

    虚拟长度与原先 ``ConcatDataset`` + ``DistributedWeightedSampler`` 一致：
    ``ceil(target/world_size) * world_size``（分布式）或 ``target``（单进程）。
    ``__getitem__(idx)`` 使用 ``(seed, epoch, idx)`` 初始化 RNG，与 ``DistributedSampler``
    按 rank 划分互不重叠的 ``idx`` 相容；训练循环需在每 epoch 调用 ``set_epoch``（见 ``utils.set_sampler_epoch``）。
    """

    def __init__(
        self,
        datasets: List[Any],
        source_names: List[str],
        normalized_weights: Dict[str, float],
        *,
        samples_per_epoch: Optional[int],
        batch_size: int,
        total_concat_len: int,
        world_size: int,
        seed: int = 42,
    ) -> None:
        if len(datasets) != len(source_names):
            raise ValueError("datasets and source_names length mismatch")
        if not datasets:
            raise ValueError("WeightedMixtureDataset requires at least one sub-dataset")

        ws = max(1, int(world_size))
        target = max(int(samples_per_epoch or total_concat_len), int(batch_size))
        nspr = max(1, int(math.ceil(target / ws)))  # 每 rank 本 epoch 目标步数（与旧 WeightedSampler 一致）
        # 全局虚拟下标总数；DistributedSampler 再按 rank 划分 → 每 rank 约 nspr 个 idx，不会二次除以 ws
        self._virtual_len = nspr * ws if ws > 1 else target

        self.datasets = list(datasets)
        self.source_names = list(source_names)
        self.epoch = 0
        self.seed = int(seed)

        w_list = [float(normalized_weights[n]) for n in source_names]
        if any(x <= 0 for x in w_list):
            raise ValueError(f"Weights must be positive, got {w_list}")
        self._w = torch.tensor(w_list, dtype=torch.double)
        self._w = self._w / self._w.sum()

        self._ds_lens = [len(d) for d in self.datasets]
        if any(L <= 0 for L in self._ds_lens):
            raise ValueError("Every sub-dataset must be non-empty")

    def __len__(self) -> int:
        return self._virtual_len

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, idx: int) -> Any:
        g = torch.Generator()
        s = (self.seed + self.epoch * 1_000_003 + int(idx)) % (2**32 - 1)
        g.manual_seed(s if s > 0 else 1)
        src = int(torch.multinomial(self._w, 1, generator=g).item())
        j = int(torch.randint(0, self._ds_lens[src], (1,), generator=g).item())
        return self.datasets[src][j]
