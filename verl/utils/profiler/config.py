
from dataclasses import dataclass, field
from typing import ClassVar

from verl.base_config import BaseConfig


@dataclass
class ProfilerConfig(BaseConfig):
    """Worker profiler config. Currently only support Nsight system profiler.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        discrete (bool): True for each task has its own database, False for all tasks in one training step
          share one database.
        all_ranks (bool): Whether to profile all ranks.
        ranks (list[int]): The ranks that will be profiled. Defaults to [].
    """

    _frozen_fields: ClassVar[set[str]] = {"discrete", "all_ranks", "ranks"}

    discrete: bool = False

    all_ranks: bool = False

    ranks: list[int] = field(default_factory=list)

    def union(self, other: "ProfilerConfig") -> "ProfilerConfig":
        return ProfilerConfig(
            all_ranks=self.all_ranks or other.all_ranks,
            ranks=list(set(self.ranks or []) | set(other.ranks or [])),
            discrete=self.discrete or other.discrete,
        )

    def intersect(self, other: "ProfilerConfig") -> "ProfilerConfig":
        return ProfilerConfig(
            all_ranks=self.all_ranks and other.all_ranks,
            ranks=list(set(self.ranks or []) & set(other.ranks or [])),
            discrete=self.discrete and other.discrete,
        )

    def __post_init__(self) -> None:
        """config validation logics go here"""
        assert isinstance(self.ranks, set | list | tuple), (
            f"Profiler ranks must be of type list, got {type(self.ranks)}"
        )
