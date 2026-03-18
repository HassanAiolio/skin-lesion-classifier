import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os

class SkinLesionDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df.UNK == 0]
        self.classes = self.df.columns.tolist()[1:-1]
        self.df['label'] = self.df[self.classes].idxmax(axis=1)
        self.df['label'] = self.df['label'].map({cls: i for i, cls in enumerate(self.classes)})
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, self.classes[row['label']], row['image']+'.jpg')
        image = Image.open(img_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, row['label']

    def get_class_weights(self):
        weights = []
        total = len(self.df)
        num_classes = 8
        for idx, cls in enumerate(self.classes):
            nb_samples = len(self.df[self.df['label'] == idx])
            weight = total / (num_classes * nb_samples)
            weights.append(weight)
        return torch.tensor(weights)