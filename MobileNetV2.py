import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import MobileNet_V2_Weights

from Modellen.DataImport import RealVsAIDataset

weights = MobileNet_V2_Weights.DEFAULT

train_dataset = RealVsAIDataset("data/test_labels.csv", "data/my_real_vs_ai_dataset/my_real_vs_ai_dataset", transform=weights.transforms())
test_dataset = RealVsAIDataset("data/test_labels.csv", "data/my_real_vs_ai_dataset/my_real_vs_ai_dataset", transform=weights.transforms())


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

