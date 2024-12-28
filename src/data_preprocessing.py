import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
from torchvision import transforms
import torch
from torch.utils.data import Dataset

# Add Gaussian noise and transform images
def transform_images(img, std):
    to_tensor = transforms.ToTensor()
    img = to_tensor(img)
    noise = torch.randn(size=img.shape) * std
    noisy_img = img + noise
    noisy_img = torch.clip(noisy_img, 0, 1)
    return noisy_img, img

# Create PyTorch dataset class
class LFWDataset(Dataset):
    def __init__(self, processed, split='train'):
        self.total = len(processed)
        if split == 'train':
            self.x = [img[0] for img in processed[:round(self.total * 0.8)]]
            self.y = [img[1] for img in processed[:round(self.total * 0.8)]]
        elif split == 'validation':
            self.x = [img[0] for img in processed[round(self.total * 0.8):round(self.total * 0.9)]]
            self.y = [img[1] for img in processed[round(self.total * 0.8):round(self.total * 0.9)]]
        elif split == 'test':
            self.x = [img[0] for img in processed[round(self.total * 0.9):]]
            self.y = [img[1] for img in processed[round(self.total * 0.9):]]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

# Load and process images
def load_data(dataset_path, std=0.1):
    img_list = os.listdir(dataset_path)
    processed = []
    for img in img_list:
        pic = image.imread(os.path.join(dataset_path, img))
        noisy_img, clean_img = transform_images(pic, std)
        processed.append((noisy_img, clean_img))
    return processed

# Utility to plot examples
def plot_processed_images(processed_images):
    fig, axes = plt.subplots(2, 5, figsize=(15, 8))
    for i in range(5):
        clean_img, noisy_img = processed_images[i]
        clean_img, noisy_img = clean_img.permute(1, 2, 0), noisy_img.permute(1, 2, 0)
        axes[0, i].imshow(clean_img)
        axes[1, i].imshow(noisy_img)
    plt.show()
