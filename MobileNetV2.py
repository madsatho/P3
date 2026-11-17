import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import MobileNet_V2_Weights
from torch.utils.data import DataLoader

from Modellen.DataImport import RealVsAIDataset
from Modellen.ProjectData import BATCH_SIZE

BATCH_SIZE = 32
weights = MobileNet_V2_Weights.DEFAULT

train_dataset = RealVsAIDataset("data/train_labels.csv", "data/my_real_vs_ai_dataset/my_real_vs_ai_dataset", transform=weights.transforms())
test_dataset = RealVsAIDataset("data/test_labels.csv", "data/my_real_vs_ai_dataset/my_real_vs_ai_dataset", transform=weights.transforms())
val_dataset = RealVsAIDataset("data/val_labels.csv", "data/my_real_vs_ai_dataset/my_real_vs_ai_dataset", transform=weights.transforms())


train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

model = models.mobilenet_v2(weights=weights)
model.classifier = nn.Identity()  # fjerner klassifikationslaget -> embedding
model.eval()

i = random.randrange(len(train_dataset))
j = random.randrange(len(test_dataset))


x = torch.stack([train_dataset[i][0], test_dataset[j][0]])

with torch.no_grad():
    e1, e2 = model(x)

print("Embedding shape:", e1.shape)  # 1280 værdier
print("Cosine similarity:", F.cosine_similarity(e1, e2, dim=0).item())



plt.subplot(1, 2, 1)
plt.imshow(train_dataset[i][0].permute(1, 2, 0))
plt.subplot(1, 2, 2)
plt.imshow(test_dataset[j][0].permute(1, 2, 0))
plt.show()

def compare(dataset, compare_dataset):
    comparisons = []
    for i in range(len(dataset)):
        for j in range(len(compare_dataset)):

            x = torch.stack([dataset[i][0], compare_dataset[j][0]])

            with torch.no_grad():
                e1, e2 = model(x)

            comparison = F.cosine_similarity(e1, e2, dim=0).item()

            comparisons.append(comparison)
    return comparisons

r = compare(val_dataset,val_dataset)
print(r)