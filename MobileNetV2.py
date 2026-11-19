import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import MobileNet_V2_Weights
from torch.utils.data import DataLoader
from Modellen.DataImport import RealVsAIDataset

BATCH_SIZE = 32
weights = MobileNet_V2_Weights.DEFAULT
device = "cuda" if torch.cuda.is_available() else "cpu"

combined_dataset = RealVsAIDataset("data/train_labels.csv", "data/my_real_vs_ai_dataset/my_real_vs_ai_dataset", transform=weights.transforms())
combined_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)


model = models.mobilenet_v2(weights=weights)
model.classifier = nn.Identity()  # fjerner klassifikationslaget -> embedding
model.eval()
model.to(device)


def embed(dataloader):
    with torch.no_grad():
        return F.normalize(torch.cat([model(x.to(device)) for x, l in dataloader]), dim=1)

def compare(dataloader, compare_dataloader):
    return embed(dataloader) @ embed(compare_dataloader).T

if __name__ == "__main__":
        r = compare(combined_dataloader, combined_dataloader)
        print(r.shape)

