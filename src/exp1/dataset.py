from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

from exp1.pointer_chase import ChaseInstance, ChaseSpec, encode_batch


class PointerChaseDataset(Dataset):
    """PyTorch Dataset adapter for pointer chase task."""

    def __init__(
        self,
        instances: Sequence[ChaseInstance],
        spec: ChaseSpec,
        num_silent: int = 0,
        silent_kind: Optional[str] = None,
        neutral_vector: Optional[torch.Tensor] = None,
    ):
        self.instances = instances
        self.spec = spec
        # Pre-encode all instances
        self.x, self.y = encode_batch(
            instances,
            spec,
            num_silent=num_silent,
            silent_kind=silent_kind,
            neutral_vector=neutral_vector,
        )

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_tuples": self.x[idx],
            "targets": self.y[idx],
        }


def exp1_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for pointer chase dataset.

    Returns:
        Dict containing:
            - 'input_tuples': [B, T, d_input]
            - 'targets': [B]
    """
    input_tuples = torch.stack([item["input_tuples"] for item in batch])
    targets = torch.stack([item["targets"] for item in batch])
    return {
        "input_tuples": input_tuples,
        "targets": targets,
    }
