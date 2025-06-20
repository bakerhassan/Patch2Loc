#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import os
import yaml
from omegaconf import OmegaConf
from src.utils.utils import  extract_volume_patches
import torchio as tio
import torch.nn.functional as F
import numpy as np

os.chdir("/lustre/scratch/bakerh/cvpr/CVPR2025/") 

from src.datamodules.Datamodules_train import IXI
from src.models.patch2loc_model import Patch2Loc

            
def remove_underscore_keys(d):
    return {k: v for k, v in d.items() if not k.startswith("_")}


with open("./configs/datamodule/IXI.yaml", "r") as file:
    config = yaml.safe_load(file)
config = remove_underscore_keys(config)
config = config['cfg']
config['mode'] = 't1'
config['data_dir'] = '/lustre/scratch/bakerh/cvpr/data/Data/'
config['sample_set'] = False
config['num_workers'] = 0
config['rescaleFactor'] = 1
config['imageDim'] = [192,192,100]

import torch
import torch.nn.functional as F

def compute_local_voxel_quantiles(images, patch_size=3, quantile_levels=None, save_path="local_quantiles.pt"):
    """
    Computes the voxel-level quantile function using a local patch around each voxel.
    More memory-friendly implementation.
    Args:
        images (torch.Tensor): Input tensor of shape (B, C, H, W).
        patch_size (int): Size of the local patch (must be odd).
        quantile_levels (list or torch.Tensor, optional): Quantile levels (e.g., [0.1, 0.5, 0.9]).
        save_path (str): Path to save the quantile tensor.
    """
    B, C, H, W = images.shape
    images = (images - images.mean(dim=0,keepdim=True)) / images.std(dim=1,keepdim=True)

    images = torch.nan_to_num(images, nan=0.0, posinf=0.0, neginf=0.0)
    if patch_size % 2 == 0:
        patch_size -= 1
    pad = patch_size // 2

    if quantile_levels is None:
        quantile_levels = torch.linspace(0, 1, steps=100, device=images.device)

    # Pad the image
    padded = F.pad(images.float(), (pad, pad, pad, pad), mode='reflect')

    # We'll store the quantiles here
    local_quantiles = torch.zeros((len(quantile_levels), C, H, W), device=images.device)
    mean = torch.zeros((C, H, W), device=images.device)
    std = torch.zeros((C, H, W), device=images.device)

    # Slide a window over the image without using unfold
    for i in range(H):
        for j in range(W):
            # Extract local patch centered at (i, j)
            patch = padded[:, :, i:i+patch_size, j:j+patch_size]  # (B, C, patch_size, patch_size)
            patch = patch.reshape(B, C, -1)  # Flatten patch

            # Compute quantiles along the patch dimension
            # q = torch.from_numpy(np.quantile(patch, quantile_levels, axis=(0,2))) # (len(quantile_levels), C)
            # local_quantiles[:, :, i, j] = q
            mean[:,i,j] = patch.mean(dim=(0,2))
            std[:,i,j] = patch.std(dim=(0,2))
    torch.save({'mean': mean, 'std': std}, save_path)
    print(f"Local quantiles saved to {save_path}")
    return None

    torch.save(local_quantiles, save_path)
    print(f"Local quantiles saved to {save_path}")
    return local_quantiles

for fold in range(5):
    # ckpt_path = f'/lustre/scratch/bakerh/cvpr/patch2loc_t2_single_scale/checkpoints/last_fold-{fold + 1}.ckpt'
    ckpt_path = f'/lustre/scratch/bakerh/cvpr/patch2loc_t1/checkpoints/last_fold-{fold + 1}.ckpt'
    config = OmegaConf.create(config)
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    # lightningDataModule = Brats21(config)
    lightningDataModule = IXI(config,fold=fold)

    # config['num_folds'] = 5
    lightningDataModule.setup()
    # dataloader = lightningDataModule.test_dataloader()
    dataloader1 = lightningDataModule.val_eval_dataloader()
    dataloader2 = lightningDataModule.test_eval_dataloader()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    self = Patch2Loc.load_from_checkpoint(ckpt_path, strict=False)
    self.on_test_start()



    self.model.eval()
    final_volumes = []
    errors = []
    predicted_logvariances = []
    for dataloader in [dataloader1, dataloader2]:
        for batch in dataloader:
                self.dataset = batch['Dataset']
                input = batch['vol'][tio.DATA]
                data_orig = batch['vol_orig'][tio.DATA]
                data_seg = batch['seg_orig'][tio.DATA] if batch['seg_available'][0] else torch.zeros_like(data_orig)
                data_mask = batch['mask_orig'][tio.DATA]
                ID = batch['ID'][0]
                age = batch['age'][0]
                self.stage = batch['stage'][0]
                label = batch['label'][0]
                # self.plot_slice_error(batch)
                start_idx = 0
                data_seg = data_seg.squeeze(0).permute(3,0,1,2)
                input = input.squeeze(0).permute(3,0,1,2)
                # input = torch.from_numpy(exposure.equalize_hist(input.numpy()))
                brain_masks = data_mask.squeeze(0).permute(3,0,1,2)
                # Determine the number of chunks
                num_chunks = 7 # Adjust this based on memory constraints
                chunk_size = input.size(0) // num_chunks
                results_error = []
                results_logvariance = []
                stride = (1,1,1)
                for start in range(0, input.size(0), chunk_size):
                    # Slice the input tensor for the current chunk
                    chunk = input[start: start + chunk_size]

                    # Process the chunk
                    patches, locations = extract_volume_patches(chunk, self.patch_size,slice_start=start + start_idx, total_slices_num=input.shape[0],
                                                                    stride=stride,rejection_rate=self.rejection_rate)
                    with torch.no_grad():
                        predicted_locations, predicted_logvariance = self.model(patches.half().to(device), locations[:, -1].to(device))
                    predicted_locations = predicted_locations.cpu()
                    predicted_logvariance = predicted_logvariance.cpu()
                    
                    # Calculate error and reshape
                    error = torch.linalg.norm(predicted_locations - locations[:, :-1],dim=1)
                    error = self.reshape_and_upsample(error, chunk.shape, stride)
                    predicted_logvariance = self.reshape_and_upsample(predicted_logvariance.mean(-1),chunk.shape, stride)
                    # predicted_logvariance = predicted_logvariance.mean(-1).reshape((input.shape[-2], input.shape[-1], input.shape[1], chunk.shape[0])).permute(3, 2, 0, 1)
                    
                    # Move results to CPU to free GPU memory
                    results_error.append(error.cpu())
                    results_logvariance.append(predicted_logvariance.cpu())

                # Concatenate results from all chunks
                error = torch.log(torch.cat(results_error, dim=0).squeeze().permute(1, 2, 0)+1e-10) #H,W,C
                predicted_logvariance = torch.cat(results_logvariance, dim=0).squeeze().permute(1, 2, 0)

                # Final computation
                final_volume = error * predicted_logvariance
                errors.append(error.permute(2,0,1).unsqueeze(1))
                predicted_logvariances.append(predicted_logvariance.permute(2,0,1).unsqueeze(1))

                # final_volume = final_volume.permute(2,0,1).unsqueeze(1)#C,H,W
                # final_volumes.append(final_volume)

    # final_volumes = torch.stack(final_volumes).squeeze()
    errors = torch.stack(errors).squeeze()
    predicted_logvariances = torch.stack(predicted_logvariances).squeeze()

    compute_local_voxel_quantiles(errors, self.patch_size[0],save_path=f"errors_{ fold + 1}_{config['mode']}.pt")
    compute_local_voxel_quantiles(predicted_logvariances, self.patch_size[0],save_path=f"logvariances_{ fold + 1}_{config['mode']}.pt")

