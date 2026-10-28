import os
import pandas as pd
import torch
from torch import nn #<--- FOrdi jeg ik vil skrive torch.nn.Module hver gang
import torchmetrics
from torch.utils.data import Dataset, DataLoader
from skimage import io
from PIL import Image # <-- Denne burde være hurtigere end skimage
from torchvision import transforms
import matplotlib.pyplot as plt


# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


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



root = "data/my_real_vs_ai_dataset/my_real_vs_ai_dataset"

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

train_set = RealVsAIDataset("data/train_labels.csv", root, transform=transform)
test_set = RealVsAIDataset("data/test_labels.csv", root, transform=transform)


# Basic checks on one item:
print("Number of items:", len(train_set))
image, label = train_set[0]
print("Image shape:", image.shape)   # expect torch.Size([3, 256, 256])
print("Image dtype:", image.dtype)   # expect torch.float32
print(f"Laveste farve værdi: {image.min().item()}, største: {image.max().item()}, Mean: {image.mean().item()}")  # within 0 and 1
print(f"Label type: {type(label)}, Label værdi: {label}")
print(f"IMG type: {type(image)}")

#Vis  første billede
#plt.imshow(image.permute(1, 2, 0).detach().cpu().numpy())
#plt.show()


#Vi bruger nu dataloader der deler de storer "lister" op i små bidder (batches) så de er nemere at arbejde med!

# Setup the batch size hyperparameter
BATCH_SIZE = 32

# Turn datasets into iterables (batches)
train_dataloader = DataLoader(train_set, # dataset to turn into iterable
    batch_size=BATCH_SIZE, # how many samples per batch?
    shuffle=True # shuffle data every epoch?
)

test_dataloader = DataLoader(test_set,
    batch_size=BATCH_SIZE,
    shuffle=False # don't necessarily have to shuffle the testing data
)

# Let's check out what we've created
print(f"Length of train dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}")
print(f"Length of test dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}") #<--- 12.5% af data btw :(


class GraveXModelV0(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),  # neural networks like their inputs in vector form
            nn.Linear(in_features=input_shape, out_features=hidden_units), # in_features = number of features in a data sample (784 pixels)
            #nn.ReLU(), # For at gør lortet ikke-lineær
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )

    def forward(self, x):
        return self.layer_stack(x)


# Need to set up model with input parameters
model_0 = GraveXModelV0(input_shape=3*256*256, # 1 input for hver pixel (skud ud claude)
    hidden_units=10, # how many units in the hidden layer
    output_shape=1) # one for every class eller 1 for binary

model_0.to(device) # Sætter modellen på min GPU


# Setup loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss() # BCE står for binary cross entropy (Claude was here...{to fix bug})
acc_fn = torchmetrics.Accuracy(task='binary').to(device) #<-- Denne er ik med i tutorial men burde virke :)
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.00007) #StatQuest har et tutorial på hvordan vi finder den bedste learning rate hvis det er.



# Import tqdm for progress bar
from tqdm.auto import tqdm

# Set the number of epochs (we'll keep this small for faster training times)
epochs = 3

# Create training and testing loop
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n-------")
    ### Training
    train_loss = 0
    # Add a loop to loop through training batches
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)  # <-- Flytter data
        model_0.train()
        # 1. Forward pass
        y_pred = model_0(X).squeeze(1) #Fordi vi kun har 1 output

        # 2. Calculate loss (per batch)
        loss = loss_fn(y_pred, y.float())
        train_loss += loss  # accumulatively add up the loss per epoch

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Print out how many samples have been seen
        if batch % 200 == 0:
            print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples")



    # Divide total train loss by length of train dataloader (average loss per batch per epoch)
    train_loss /= len(train_dataloader)

    ### Testing
    # Setup variables for accumulatively adding up loss and accuracy
    test_loss, test_acc = 0, 0
    model_0.eval()
    with torch.inference_mode():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)  # <-- Flytter data
            # 1. Forward pass
            test_pred = model_0(X).squeeze(1)

            # 2. Calculate loss (accumulatively)
            test_loss += loss_fn(test_pred, y.float())  # accumulatively add up the loss per epoch

            # 3. Calculate accuracy (preds need to be same as y_true)
            test_acc += acc_fn((torch.sigmoid(test_pred) > 0.5).int(), y)

        # Calculations on test metrics need to happen inside torch.inference_mode()
        # Divide total test loss by length of test dataloader (per batch)
        test_loss /= len(test_dataloader)

        # Divide total accuracy by length of test dataloader (per batch)
        test_acc /= len(test_dataloader)

    ## Print out what's happening
    print(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}\n")

