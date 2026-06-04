import torch
import torchvision
import numpy as np
import os
from PIL import Image

class MyDataset(torch.utils.data.Dataset):
    # for Airplanes, Cars and Ships dataset
    # train or test folders further have subfolders containing images
    # for each category

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.all_image_paths = []
        self.all_labels = []

        for planecarship_dir in os.listdir(data_dir): # each category folder  
            category_path = os.path.join(data_dir, planecarship_dir)
            if not os.path.isdir(category_path):
                continue
            if planecarship_dir == "airplanes":
                label = 0
            elif planecarship_dir == "cars":
                label = 1
            elif planecarship_dir == "ships":
                label = 2
            image_paths = [os.path.join(category_path, f) for f in os.listdir(category_path) if f.endswith('.jpg')] 
            # list of image filenames for a category, e.g., airplane, car, or ship
            labels = [label for i in range(len(image_paths))] 
            self.all_image_paths += image_paths
            self.all_labels += labels
        self.num_classes = len(set(self.all_labels))

    def __len__(self):
        return len(self.all_image_paths)

    def __getitem__(self, index):
        img_path = self.all_image_paths[index]
        label = self.all_labels[index]

        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label
