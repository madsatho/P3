import torch
from tqdm.auto import tqdm
from timeit import default_timer as timer



def train_n_test(train_dataloader, test_dataloader, model, loss_fn, acc_fn, optimizer, device, epochs):
    train_time_start = timer()

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
            y_pred = model(X).squeeze(1)  # Fordi vi kun har 1 output

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
            if batch % 25 == 0:
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
        print(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.3f}\n")

        train_time_end = timer()
        print(f"Time taken: {train_time_end - train_time_start:.3f} seconds")