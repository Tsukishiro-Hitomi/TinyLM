"""Training-data batch sampling."""

import numpy.typing as npt
import torch

def get_batch(x: npt.NDArray,
              batch_size: int, 
              context_length: int, 
              device: str
              ) -> tuple[torch.Tensor, torch.Tensor]:
    assert device == "cpu" or device.startswith("cuda:"), "device must be 'cpu' or a CUDA device"
    total_length = x.shape[0]
    assert total_length >= context_length + 1, "the length of dataset must be greater than context_length"
    starts = torch.randint(0, total_length - context_length, (batch_size,))
    offsets = torch.arange(context_length)
    indices = starts[:, None] + offsets[None, :]

    # Constructing with ``torch.Tensor(..., device='cuda')`` uses a legacy
    # constructor that only accepts CPU devices. Index in NumPy first and move
    # the completed batch to the requested device in one transfer.
    pre = torch.as_tensor(x[indices.numpy()], device=device, dtype=torch.long)
    next = torch.as_tensor(x[(indices + 1).numpy()], device=device, dtype=torch.long)
    return pre, next
