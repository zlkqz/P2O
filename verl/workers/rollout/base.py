
from abc import ABC, abstractmethod

from verl import DataProto

__all__ = ["BaseRollout"]


class BaseRollout(ABC):
    """Base class for rollout."""

    @abstractmethod
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Generate sequences"""
        pass
