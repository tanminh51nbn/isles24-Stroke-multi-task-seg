import torch
from torch.utils.data import Sampler
import numpy as np
import random

class TaskBalancedBatchSampler(Sampler):
    """
    Custom BatchSampler to ensure each batch contains a specific ratio of tasks.
    Target: 3 LVO, 1 Lesion, 2 CoW, 1 Neg (Total 7 fixed slots in BS=24).
    The remaining slots are filled randomly from all positive cases.
    """
    def __init__(
        self, 
        task_indices: dict, 
        batch_size: int, 
        num_batches: int = 500,
        rank: int = 0,
        world_size: int = 1
    ):
        self.task_indices = task_indices
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.rank = rank
        self.world_size = world_size
        
        # Define fixed slots
        self.n_lvo = 3
        self.n_lesion = 1
        self.n_cow = 2
        self.n_neg = 1
        self.n_fixed = self.n_lvo + self.n_lesion + self.n_cow + self.n_neg
        self.n_random = batch_size - self.n_fixed
        
        # All indices for random slots to maintain diversity and prevent FP
        self.all_pool = []
        for pool in task_indices.values():
            self.all_pool.extend(pool)
        
        # Seed the RNG differently per rank
        self.rng = random.Random(42 + rank)

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            
            # 1. Fill fixed slots
            batch.extend(self.rng.sample(self.task_indices["lvo"], self.n_lvo))
            batch.extend(self.rng.sample(self.task_indices["lesion"], self.n_lesion))
            batch.extend(self.rng.sample(self.task_indices["cow"], self.n_cow))
            batch.extend(self.rng.sample(self.task_indices["neg"], self.n_neg))
            
            # 2. Fill random slots from all-pool
            batch.extend(self.rng.sample(self.all_pool, self.n_random))
            
            # 3. Final internal shuffle
            self.rng.shuffle(batch)
            
            yield batch

    def __len__(self):
        return self.num_batches
