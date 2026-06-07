import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import os
import platform
def get_optimal_dataloader_settings():
    system = platform.system()
    if system == "Windows":
        return {
            'num_workers': 0,
            'pin_memory': False,
            'persistent_workers': False,
            'prefetch_factor': None,
        }
    else:
        return {
            'num_workers': min(4, os.cpu_count()),
            'pin_memory': True,
            'persistent_workers': True,
            'prefetch_factor': 2,
        }
def create_safe_dataloader(dataset, batch_size, shuffle=True, collate_fn=None, **kwargs):
    optimal_settings = get_optimal_dataloader_settings()
    final_kwargs = {**optimal_settings, **kwargs}
    max_batch_size = 256 if platform.system() == "Windows" else 256
    actual_batch_size = min(batch_size, max_batch_size)
    if actual_batch_size != batch_size:
        print(f"Warning: Batch size reduced from {batch_size} to {actual_batch_size} to avoid memory issues")
    dataloader = DataLoader(
        dataset,
        batch_size=actual_batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        **final_kwargs
    )
    return dataloader
def setup_multiprocessing():
    if platform.system() == "Windows":
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
    if hasattr(torch, 'multiprocessing'):
        torch.multiprocessing.set_sharing_strategy('file_system')
setup_multiprocessing()
