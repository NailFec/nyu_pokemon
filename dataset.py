# dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class PokemonDataset(Dataset):
    def __init__(self, data_path, labels_path, transform=None):
        """
        Args:
            data_path (string): .npy image data file path.
            labels_path (string): .npy label data file path.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        print("Loading data, this may take a while...")
        self.data = np.load(data_path)
        self.labels = np.load(labels_path)
        self.transform = transform
        print("Data loaded.")

        # Data type check and conversion
        if self.data.dtype != np.float32:
            print(f"Image data type is {self.data.dtype}, converting to float32...")
            self.data = self.data.astype(np.float32)
        
        if self.labels.dtype != np.int64:
            print(f"Label data type is {self.labels.dtype}, converting to int64...")
            self.labels = self.labels.astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Get image and label from numpy arrays
        image_np = self.data[idx]
        label = self.labels[idx]

        # Convert (C, H, W) to (H, W, C) for PyTorch transforms
        image_np_hwc = np.transpose(image_np, (1, 2, 0))
        
        # Convert [0, 1] float to [0, 255] uint8
        image_np_uint8 = (image_np_hwc * 255).astype(np.uint8)
        
        # Convert to PIL Image
        image_pil = Image.fromarray(image_np_uint8)

        if self.transform:
            image_pil = self.transform(image_pil)
        
        return image_pil, torch.tensor(label, dtype=torch.long)