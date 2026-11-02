import torch
from torch import nn #<--- Fordi jeg ik vil skrive torch.nn.Module hver gang
import torchmetrics
from torch.utils.data import DataLoader
from torchvision import transforms

from DataImport import RealVsAIDataset, data_loaded_test, image_dimensions
from Models import GraveXModelV0, GraveXModelV2
from TrainTest import train_n_test


# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
root = "data/my_real_vs_ai_dataset/my_real_vs_ai_dataset"
epochs = 1
BATCH_SIZE = 128

transform = transforms.Compose([
    #transforms.Resize((256, 256)), <--- Ikke nødvendig
    transforms.ToTensor(),
])

def main():
    print(device)

    # Check available GPU memory and total GPU memory
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        total_free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
        print(f"Total free GPU memory: {round(total_free_gpu_memory * 1e-9, 3)} GB")
        print(f"Total GPU memory: {round(total_gpu_memory * 1e-9, 3)} GB \n")


    train_set = RealVsAIDataset("data/train_labels.csv", root, transform=transform)
    test_set = RealVsAIDataset("data/test_labels.csv", root, transform=transform)

    #data_loaded_test(train_set, test_set)
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


    train_n_test(train_dataloader, test_dataloader, model, loss_fn, acc_fn, optimizer, device, epochs)


if __name__ == "__main__":
    main()