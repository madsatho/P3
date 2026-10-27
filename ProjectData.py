import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from skimage import io
from torchvision import transforms

# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

class RealVsAIDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        filename = self.annotations.iloc[index, 1]
        label = int(self.annotations.iloc[index, 2])

        # 1 = real, 0 = AI
        subfolder = "real" if label == 1 else "ai_images"
        img_path = os.path.join(self.root_dir, subfolder, filename)

        image = io.imread(img_path)
        y_label = label #Hvis denne skal være tensor så brug torch.tensor(label)

        if self.transform:
            image = self.transform(image)

        return image, y_label

from torchvision import transforms

root = "data/my_real_vs_ai_dataset/my_real_vs_ai_dataset"

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

train_set = RealVsAIDataset("data/train_labels.csv", root, transform=transform)

# Step 1: basic checks on one item
print("Number of items:", len(train_set))
image, label = train_set[0]
print("Image shape:", image.shape)   # expect torch.Size([3, 256, 256])
print("Image dtype:", image.dtype)   # expect torch.float32
print(f" Laveste farve værdi: {image.min().item()}, største: {image.max().item()}, Mean: {image.mean().item()}")  # within 0 and 1
print(f"Label type: {type(label)}, Label værdi: {label}")

# Step 2: one batch from a DataLoader
loader = DataLoader(train_set, batch_size=32, shuffle=True)
images, labels = next(iter(loader))
print("Batch images:", images.shape)  # expect [32, 3, 256, 256]
print("Batch labels:", labels)        # a mix of 0s and 1s