import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import json

class COCODataset(Dataset):
    def __init__(self, root, transform=None, mode='train'):
        self.root = root
        self.transform = transform
        self.mode = mode
        self.images = sorted(os.listdir(os.path.join(root, f'{mode}2017')))
        with open(os.path.join(root, 'annotations', f'instances_{mode}2017.json')) as f:
            self.annotations = json.load(f)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, f'{self.mode}2017', self.images[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        label = self._get_label(self.images[idx])
        return image, label

    def _get_label(self, image_id):
        labels = []
        for ann in self.annotations['annotations']:
            if ann['image_id'] == image_id:
                labels.append(ann)
        return labels
