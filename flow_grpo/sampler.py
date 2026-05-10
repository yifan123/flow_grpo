import os
import math
import json
import torch
from torch.utils.data import Sampler, Dataset, DataLoader
from collections import Counter

class TextPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}.txt')
        with open(self.file_path, 'r') as f:
            self.prompts = [line.strip() for line in f.readlines()]
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas
    
class DistributedKRepeatSampler(Sampler):
    """
    This class is a new implementation of the distributed K-repeat sampler, used in current code.
    Where the number of unique samples is determined by the total number of replicas and the batch size,
    and k can be not divisible by n*b.
    """
    def __init__(self, dataset : Dataset, batch_size : int, k : int, m : int, num_replicas : int, rank : int, seed :int = 0):
        self.dataset = dataset
        self.batch_size = batch_size  # Batch size per replica
        self.k = k                    # Number of repetitions per sample
        self.m = m                    # `Least` number of unique sample per epoch
        self.num_replicas = num_replicas  # Total number of replicas, process num, gpu num
        self.rank = rank              # Current replica rank
        self.seed = seed              # Random seed for synchronization
        
        # Compute the number of samples for each batch iteration
        self.sample_num_per_iteration = self.num_replicas * self.batch_size
        # Compute the lcm, total sample number per epoch
        self.sample_num_per_epoch = math.lcm(self.k * self.m, self.sample_num_per_iteration)
        # Compute the minimal iteration number to include k repetitions for all prompts,
        # i.e., the number of batches per epoch per replica
        self.num_batches_per_epoch = int(self.sample_num_per_epoch // self.sample_num_per_iteration)

        # Update m - number of unique sample per epoch
        self.m = self.sample_num_per_epoch // self.k

        self.epoch = 0

    def __iter__(self):
        while True:
            # Generate a deterministic random sequence to ensure all replicas are synchronized
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            
            # Randomly select m unique samples
            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()

            # Repeat each sample k times to generate m*k total samples.
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            
            # Shuffle to ensure uniform distribution
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
            for i in range(self.num_batches_per_epoch):
                # Offset for current iteration
                offset = i * self.sample_num_per_iteration
                # Compute start and end indices for current replica
                start = offset + self.rank * self.batch_size
                end = start + self.batch_size
                yield shuffled_samples[start:end]

    def set_epoch(self, epoch : int):
        self.epoch = epoch  # Used to synchronize random state across epochs


if __name__ == "__main__":
    # This is a simple test for basic functionality.
    num_processes = 4 # 3
    train_batch_size = 1 # 5
    num_image_per_prompt = 16 # 2
    unique_sample_per_epoch = 30 # 13
    sample_num_per_batch = num_processes * train_batch_size
    sample_num_per_epoch = math.lcm(num_image_per_prompt * unique_sample_per_epoch, sample_num_per_batch)
    num_batches_per_epoch = int(sample_num_per_epoch // sample_num_per_batch)
    unique_sample_per_epoch = sample_num_per_epoch // num_image_per_prompt

    for var_name in ['train_batch_size', 'num_image_per_prompt', 'num_processes', 'sample_num_per_epoch', 'sample_num_per_batch', 'num_batches_per_epoch', 'unique_sample_per_epoch']:
        var_value = vars()[var_name]
        print(f"{var_name:>30}: {var_value:<10}")
    

    # assert num_batches_per_epoch % 2 == 0, f"Please set config.sample.num_batches_per_epoch {num_batches_per_epoch} to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."


    dataset = 'dataset/ocr'
    train_dataset = TextPromptDataset(dataset, 'train')
    train_samplers = [
        DistributedKRepeatSampler(
        dataset=train_dataset,
        batch_size=train_batch_size,
        k=num_image_per_prompt,
        m=unique_sample_per_epoch,
        num_replicas=num_processes,
        rank=i,
        seed=0)
        for i in range(num_processes)
    ]

    train_dataloaders = [
        DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=1,
            collate_fn=TextPromptDataset.collate_fn,
            # persistent_workers=True
        ) for train_sampler in train_samplers
    ]


    train_iters = [iter(loader) for loader in train_dataloaders]

    # No bug here, for non-dist env.
    epoch_num = 100
    error_epoch = []
    for epoch in range(epoch_num):
        all_samples = []
        for j, sampler in enumerate(train_samplers):
            sampler.set_epoch(epoch)
            loader_iter = train_iters[j]
            for i in range(num_batches_per_epoch):
                samples = next(loader_iter)
                all_samples.extend(samples[0])

        # for k,v in sorted(counter.items(), key=lambda item: item[0]):
        #     short_k = k[:10] if len(k) > 10 else k
        #     print(f"{short_k:>12}:{v:<6}")
        counter = Counter(all_samples)
        all_unique_group_size = set(Counter(list(counter.values())).values())
        # print(f"\nTotal unique samples: {len(counter)}")
        # print(f"Sample length: {Counter(list(counter.values()))}")
        # print(f"All Unique group sizes: {all_unique_group_size}")
        if len(all_unique_group_size) > 1:
            error_epoch.append((epoch, all_unique_group_size))

    
    print(error_epoch)
