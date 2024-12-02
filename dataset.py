from PIL import Image
import numpy as np
from torchvision import transforms
import os
import torch
from torch.utils.data import Dataset

class CarImageDataset(Dataset):
    def __init__(self, root_dir="cropped_images", split="train", upscale_factor=2):
        self.root_dir = os.path.join(root_dir, split)
        self.image_paths = []
        self.upscale_factor = upscale_factor
        
        # Get all image paths from subdirectories
        for crop in os.listdir(self.root_dir):
            if crop.endswith(('.jpg', '.jpeg', '.png')):
                self.image_paths.append(os.path.join(self.root_dir, crop))
                
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        while True:
            try:
                img_path = self.image_paths[idx]
                hr = Image.open(img_path)
                hr = torch.from_numpy(np.array(hr)).permute(2, 0, 1)  # Move channels to the first dimension
                
                if hr.dim() != 3 or hr.shape[0] != 3:
                    raise ValueError(f"Unexpected tensor shape: {hr.shape}")
                    
                # Normalize the HR image
                hr = hr.float() / 255.0
                
                hr_size = (hr.shape[1], hr.shape[2])
                lr_size = (hr_size[0] // self.upscale_factor, hr_size[1] // self.upscale_factor)
                lr = transforms.Resize(lr_size, interpolation=Image.BICUBIC)(Image.fromarray((hr.permute(1, 2, 0).numpy() * 255).astype(np.uint8)))
                lr = torch.from_numpy(np.array(lr)).permute(2, 0, 1)  # Move channels to the first dimension
                
                # Normalize the LR image
                lr = lr.float() / 255.0
                
                return lr, hr
            except (ValueError, IndexError) as e:
                idx = (idx + 1) % len(self.image_paths)
