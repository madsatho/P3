from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image # <-- Denne burde være hurtigere end skimage
import matplotlib.pyplot as plt


#Importering af data:
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

        image = Image.open(img_path).convert("RGB")  # altid 3 kanaler
        y_label = label #Hvis denne skal være tensor så brug torch.tensor(label). JEG KAN OGSÅ LAVE DENNE TIL FLOAT

        if self.transform:
            image = self.transform(image)

        return image, y_label

def data_loaded_test(train_set, test_set, show_picture=False):

    # Basic checks on one item:
    print(f"Number of items {train_set} :", len(train_set))
    print(f"Number of items {test_set} :", len(test_set))

    image, label = train_set[0]
    print("Image shape:", image.shape)  # expect torch.Size([3, 256, 256])
    print("Image dtype:", image.dtype)  # expect torch.float32
    print(f"Mean color value: {image.mean().item()}")  # within 0 and 1
    print(f"Label type: {type(label)}, Label værdi: {label}")
    print(f"IMG type: {type(image)}")

    if show_picture:
        plt.imshow(image.permute(1, 2, 0).detach().cpu().numpy())
        plt.show()

def image_dimensions(image_set, CNN=False):
    image, label = image_set[0]
    if CNN:
        return image.shape[0]
    else:
        tot_tensor_value=image.numel()
        if type(tot_tensor_value) == int:
            return tot_tensor_value
        else:
            print(f"Billedets værdi er ikke den rigtige datatype")
            exit(1)