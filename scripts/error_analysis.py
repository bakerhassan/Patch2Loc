#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch
import os
import yaml
from omegaconf import OmegaConf

os.chdir("/lustre/scratch/bakerh/cvpr/CVPR2025/") 

from src.datamodules.Datamodules_eval import Brats21
from src.datamodules.Datamodules_train import IXI
from src.models.patch2loc_model import Patch2Loc

            
def remove_underscore_keys(d):
    return {k: v for k, v in d.items() if not k.startswith("_")}

ckpt_path = '/lustre/scratch/bakerh/cvpr/patch2loc_t2_single_scale/checkpoints/last_fold-5.ckpt'
# ckpt_path = '/lustre/scratch/bakerh/cvpr/patch2loc_t1/checkpoints/last_fold-1.ckpt'
with open("./configs/datamodule/IXI.yaml", "r") as file:
    config = yaml.safe_load(file)
config = remove_underscore_keys(config)
config = config['cfg']
config['mode'] = 't2'
config['data_dir'] = '/lustre/scratch/bakerh/cvpr/data/Data/'
config['sample_set'] = True
config['num_workers'] = 0
config['rescaleFactor'] = 1
config['imageDim'] = [192,192,100]

config = OmegaConf.create(config)
config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
# lightningDataModule = Brats21(config)
lightningDataModule = IXI(config,1)

# config['num_folds'] = 5
lightningDataModule.setup()
# dataloader = lightningDataModule.test_dataloader()
dataloader = lightningDataModule.val_eval_dataloader()

# Ensure the output folder exists
output_folder = 'notebook_figures'
os.makedirs(output_folder, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

self = Patch2Loc.load_from_checkpoint(ckpt_path, strict=False)
self.on_test_start()


# In[15]:


from matplotlib import colors
from src.utils.utils import  extract_volume_patches
import torchio as tio

from skimage import exposure
import torch
import matplotlib.pyplot as plt
from torchvision.transforms.functional import equalize

def plot_images(input_images, final_volumes, brain_masks, num_rows=3):
    """
    Plots rotated input images, final volumes (heatmaps), and brain masks (GT) side by side.

    Parameters:
        input_images (list of torch.Tensor): List of input images.
        final_volumes (list of torch.Tensor): List of heatmaps.
        brain_masks (list of torch.Tensor): List of ground truth masks.
        num_rows (int): Number of rows to display.
    """
    num_samples = min(len(input_images), num_rows)  # Ensure we don't exceed available images
    fig, axes = plt.subplots(num_samples, 3, figsize=(9, 3 * num_samples))  # Adjust figure size

    for i in range(num_samples):
        # Rotate each image by 270 degrees (-90)
        input_rotated = torch.rot90(input_images[i].squeeze(), k=3, dims=(0, 1))
        heatmap_rotated = torch.rot90(final_volumes[i].squeeze(), k=3, dims=(0, 1))
        mask_rotated = torch.rot90(brain_masks[i].squeeze(), k=3, dims=(0, 1))

        # Convert tensors to numpy for visualization
        input_np = input_rotated.numpy()
        heatmap_np = heatmap_rotated.numpy()
        mask_np = mask_rotated.numpy()

        # First column: Rotated Input Image
        axes[i, 0].imshow(input_np, cmap="gray")
        axes[i, 0].axis("off")

        # Second column: Rotated Heatmap
        axes[i, 1].imshow(heatmap_np, cmap="jet", norm=colors.Normalize(vmin=0, vmax=final_volumes.max().item()+.01))  # Heatmap visualization
        axes[i, 1].axis("off")

        # Third column: Rotated Ground Truth Mask
        axes[i, 2].imshow(mask_np, cmap="gray")  # Transparent overlay if needed
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.show()

self.model.eval()
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
        start_idx = 60
        data_seg = data_seg.squeeze(0).permute(3,0,1,2)[start_idx:70]
        input = input.squeeze(0).permute(3,0,1,2)[start_idx:70]
        # input = torch.from_numpy(exposure.equalize_hist(input.numpy()))
        brain_masks = data_mask.squeeze(0).permute(3,0,1,2)[start_idx:70]
        # Determine the number of chunks
        num_chunks = 10 # Adjust this based on memory constraints
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
            # error = error.reshape((input.shape[-2], input.shape[-1], input.shape[1], chunk.shape[0])).permute(3, 2, 0, 1)
            predicted_logvariance = self.reshape_and_upsample(predicted_logvariance.mean(-1),chunk.shape, stride)
            # predicted_logvariance = predicted_logvariance.mean(-1).reshape((input.shape[-2], input.shape[-1], input.shape[1], chunk.shape[0])).permute(3, 2, 0, 1)
            
            # Move results to CPU to free GPU memory
            results_error.append(error.cpu())
            results_logvariance.append(predicted_logvariance.cpu())

        # Concatenate results from all chunks
        error = torch.log(torch.cat(results_error, dim=0).squeeze().permute(1, 2, 0)+1e-10)
        predicted_logvariance = torch.cat(results_logvariance, dim=0).squeeze().permute(1, 2, 0)

        # Final computation
        final_volume = error * predicted_logvariance
        final_volume = final_volume.permute(2,0,1).unsqueeze(1)
        plot_images(input, final_volume * brain_masks,data_seg)


# In[9]:


import torch
import umap
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Placeholder lists for features and labels
all_features = []
all_labels = []

self.model.eval()

for batch in dataloader:
    input = batch['vol'][tio.DATA]
    data_seg = batch['seg_orig'][tio.DATA] if batch['seg_available'][0] else torch.zeros_like(input)

    # Process slices
    start_idx = 60
    data_seg = data_seg.squeeze(0).permute(3, 0, 1, 2)[60:70]
    input = input.squeeze(0).permute(3, 0, 1, 2)[60:70]

    num_chunks = 10
    chunk_size = input.size(0) // num_chunks
    stride = (self.patch_size[0],self.patch_size[1],1)

    for start in range(0, input.size(0), chunk_size):
        chunk = input[start: start + chunk_size]
        patches, _ = extract_volume_patches(chunk, self.patch_size, slice_start=start + start_idx,
                                            total_slices_num=input.shape[0], stride=stride,
                                            rejection_rate=self.rejection_rate)

        patches_seg, _ = extract_volume_patches(data_seg[start: start + chunk_size], self.patch_size, slice_start=start,
                                                total_slices_num=input.shape[0], stride=stride,
                                                rejection_rate=self.rejection_rate)

        # Compute mean intensity per patch
        mean_intensities = patches_seg.mean(dim=(1, 2, 3))  # Assuming (batch, H, W, D)

        # Classify patches
        mask_abnormal = mean_intensities > 0.9  # Abnormal patches
        mask_normal = mean_intensities < 0.1    # Normal patches
        selected_patches = mask_abnormal | mask_normal

        # Keep only selected patches
        patches = patches[selected_patches]
        labels = mask_abnormal[selected_patches].float()  # 1 for abnormal, 0 for normal

        if patches.size(0) > 0:  # Ensure we have valid data
            with torch.no_grad():
                features = self.model.encoder(patches.half().to(device))
                all_features.append(features.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

# Convert to numpy arrays
all_features = np.concatenate(all_features, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

# Apply UMAP
reducer = umap.UMAP(n_components=2, random_state=42)
umap_embedding = reducer.fit_transform(all_features)

# Plot UMAP
plt.figure(figsize=(8, 6))
plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=all_labels, cmap="coolwarm", alpha=0.7)
plt.colorbar(label="Class (0: Normal, 1: Abnormal)")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.title("UMAP Visualization of Extracted Features")
plt.show()

