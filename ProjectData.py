import torch
from torch import nn #<--- Fordi jeg ik vil skrive torch.nn.Module hver gang
import torchmetrics
from torch.utils.data import DataLoader
from torchvision import transforms

from DataImport import RealVsAIDataset, data_loaded_test, image_dimensions
from Models import GraveXModelV0, GraveXModelV2


# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
root = "data/my_real_vs_ai_dataset/my_real_vs_ai_dataset"
epochs = 3
BATCH_SIZE = 128

transform = transforms.Compose([
    #transforms.Resize((256, 256)), <--- Ikke nødvendig
    transforms.ToTensor(),
])

def main():
    print(device)

    train_set = RealVsAIDataset("data/train_labels.csv", root, transform=transform)
    test_set = RealVsAIDataset("data/test_labels.csv", root, transform=transform)

    data_loaded_test(train_set, test_set)
    network_input=image_dimensions(train_set, True)


    #Vi bruger nu dataloader der deler de storer "lister" op i små bidder (batches) så de er nemere at arbejde med!
    # Turn datasets into iterables (batches)
    train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    test_dataloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    # Let's check out what we've created
    print(f"Length of train dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}")
    print(f"Length of test dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}") #<--- 12.5% af data btw :(


    model = GraveXModelV2(input_shape=network_input, hidden_units=10, output_shape=1)
    model.to(device) # Sætter modellen på min GPU


    # Setup loss function and optimizer
    loss_fn = nn.BCEWithLogitsLoss() # BCE står for binary cross entropy (Claude was here...{to fix bug})
    acc_fn = torchmetrics.Accuracy(task='binary').to(device) #<-- Denne er ik med i tutorial men burde virke :)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3) #Adam er mere tilgivende end SGD, men StatQuest har et tutorial på at finde learning rate hvis nødvendigt.



    # Import tqdm for progress bar
    from tqdm.auto import tqdm

    # Set the number of epochs (we'll keep this small for faster training times)


    # Create training and testing loop
    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch}\n-------")
        ### Training
        train_loss = 0
        # Add a loop to loop through training batches
        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)  # <-- Flytter data
            model.train()
            # 1. Forward pass
            y_pred = model(X).squeeze(1) #Fordi vi kun har 1 output

            # 2. Calculate loss (per batch)
            loss = loss_fn(y_pred, y.float())
            train_loss += loss.item()  # accumulatively add up the loss per epoch

            # 3. Optimizer zero grad
            optimizer.zero_grad()

            # 4. Loss backward
            loss.backward()

            # 5. Optimizer step
            optimizer.step()

            # Print out how many samples have been seen
            if batch % 50 == 0:
                print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples")



        # Divide total train loss by length of train dataloader (average loss per batch per epoch)
        train_loss /= len(train_dataloader)

        ### Testing
        # Setup variables for accumulatively adding up loss and accuracy
        test_loss, test_acc = 0, 0
        model.eval()
        with torch.inference_mode():
            for X, y in test_dataloader:
                X, y = X.to(device), y.to(device)  # <-- Flytter data
                # 1. Forward pass
                test_pred = model(X).squeeze(1)

                # 2. Calculate loss (accumulatively)
                test_loss += loss_fn(test_pred, y.float()).item()  # accumulatively add up the loss per epoch

                # 3. Calculate accuracy (preds need to be same as y_true)
                test_acc += acc_fn((torch.sigmoid(test_pred) > 0.5).int(), y).item()

            # Calculations on test metrics need to happen inside torch.inference_mode()
            # Divide total test loss by length of test dataloader (per batch)
            test_loss /= len(test_dataloader)

            # Divide total accuracy by length of test dataloader (per batch)
            test_acc /= len(test_dataloader)

        ## Print out what's happening
        print(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}\n")

if __name__ == "__main__":
    main()