import torch
import os

from .dataset_factory import dataset_factory


def worker_init_fn(worker_id):
    os.sched_setaffinity(0, range(os.cpu_count())) 

def get_dataloader(dset_root, 
                   dset_names, 
                   if_train, 
                   batch_size, 
                   num_workers, 
                   clip_length=25, 
                   shuffle=True, 
                   image_height=None, 
                   image_width=None, 
                   non_overlapping_clips=False, 
                   ego_only=False,
                   bbox_masking_prob=0.0, 
                   specific_samples=None,
                   specific_categories=None,
                   force_clip_type=None):
    
    dataset = dataset_factory(dset_names, 
                              root=dset_root, 
                              train=if_train, 
                              clip_length=clip_length, 
                              resize_height=image_height, 
                              resize_width=image_width, 
                              non_overlapping_clips=non_overlapping_clips, 
                              bbox_masking_prob=bbox_masking_prob,
                              ego_only=ego_only,
                              specific_samples=specific_samples,
                              specific_categories=specific_categories,
                              force_clip_type=force_clip_type)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn
    )

    return dataset, dataloader

