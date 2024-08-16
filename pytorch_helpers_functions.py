import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchmetrics import F1Score
import matplotlib.pyplot as plt


# Data Preparation
# =========================================
## calculate_mean_std - calculates mean and std for the images in folder

# Model training
# =========================================
## train_step - runs model learning for one epoch
## val_step - evaluates model for epoch
## train_model - combines previous two ones

def calculate_mean_std(images_dir:str):

    """
    Calculates mean and std for tensors from images in folder.
    Normaly with images we can use ImagesNet's mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]. 
    But with other types of images we need to count own values.

    Returns: 
    tuple: (mean, std) for the dataset. 
    
    Example usage:
    mean, std = calculate_mean_std('images')
    """

    transform = transforms.Compose([transforms.Resize((128, 128)),
                                    transforms.ToTensor()])

    # Load the dataset
    dataset = datasets.ImageFolder(root=images_dir, transform=transform)

    # Create a DataLoader to iterate through the dataset
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)


    mean = 0.0
    std = 0.0
    total_images = 0
    
    for images, _ in loader:
        batch_samples = images.size(0)  # batch size (the last batch can have smaller size)
        images = images.view(batch_samples, images.size(1), -1)  # reshape to (batch_size, channels, height*width)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_samples
    
    mean /= total_images
    std /= total_images
    
    return mean, std


# ==============================================================
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               num_classes: int):

    """
    Trains PyTorch model for one epoch.
    Args:
    model: PyTorch model
    dataloader: A DataLoader instance
    loss_fn: Loss function
    optimizer: Optimizer

    Returns:
    A tuple of training loss and accuracy
    """
    train_loss, train_acc, train_f1 = 0, 0, 0
    # Initialize F1 metric
    f1_metric = F1Score(num_classes=num_classes, 
                        average='weighted', 
                        task='multiclass').to(device)  # 'macro', 'micro', 'weighted', etc.

    # Turn on train mode
    model.train()
    for batch, (X, y) in enumerate(dataloader):

        # Send data to device
        X, y = X.to(device), y.to(device)

        # Forward pass
        y_pred = model(X)
        # Calculate the losss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        # Optimizer zero grad
        optimizer.zero_grad()
        # Loss backward
        loss.backward()
        # Optimizer step
        optimizer.step()

        # Calculate accuracy
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)
        train_f1 += f1_metric(y_pred_class, y)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    train_f1 = train_f1 / len(dataloader)

    # Choose what to return
    return train_loss, train_f1



# ==============================================================
def val_step(model: torch.nn.Module,
             dataloader: torch.utils.data.DataLoader,
             loss_fn: torch.nn.Module,
             device: torch.device,
             num_classes: int):

    """
    Performs PyTorch model's validation during one epoch.
    Args:
    model: PyTorch model
    dataloader: A DataLoader instance
    loss_fn: Loss function
    device: Device "cpu" or "cuda"

    Returns:
    A tuple of val loss and accuracy
    """
    val_loss, val_acc, val_f1 = 0, 0, 0
    # Initialize F1 metric
    f1_metric = F1Score(num_classes=num_classes, 
                        average='weighted', 
                        task='multiclass').to(device)  # 'macro', 'micro', 'weighted', etc.
    # Turn on eval mode
    model.eval()

    with torch.inference_mode():

        for batch, (X, y) in enumerate(dataloader):

            # Send data to device
            X, y = X.to(device), y.to(device)

            # Forward pass
            y_pred = model(X)
            # Calculate the losss
            loss = loss_fn(y_pred, y)

            val_loss += loss.item()

            # Calculate accuracy
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            val_acc += (y_pred_class == y).sum().item()/len(y_pred)
            val_f1 += f1_metric(y_pred_class, y)

    val_loss = val_loss / len(dataloader)
    val_acc = val_acc / len(dataloader)
    val_f1 = val_f1 / len(dataloader)

    # Choose what to return
    return val_loss, val_f1

# ==============================================================
def train_model(model: torch.nn.Module,
                train_dataloader: torch.utils.data.DataLoader,
                val_dataloader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                loss_fn: torch.nn.Module,
                epochs: int,
                device: torch.device,
                num_classes: int):

    """
    Trains PyTorch model for N epochs
    Returns:
    Dictionaries of Train Loss, Val Loss, Train Acc, Val Accuracy.
    If test dataloader in input it also returns final test Loss and Test accuracy
    """
    # Move model to device
    model.to(device)
    print(f"Device: {device}")

    train_f1, val_f1, train_loss, val_loss = [], [], [], []

    for epoch in range(epochs):
        # Run train and collect results in list
        train_loss_ep, train_f1_ep = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device,
                                            num_classes=num_classes)
        train_loss.append(train_loss_ep)
        train_f1.append(train_f1_ep)

        # Run validation and collect metrics in list
        val_loss_ep, val_f1_ep = val_step(model=model,
                                         dataloader=val_dataloader,
                                         loss_fn=loss_fn,
                                         device=device,
                                         num_classes=num_classes)
        val_loss.append(val_loss_ep)
        val_f1.append(val_f1_ep)

        print(f"""Epoch {epoch+1} of {epochs}_ _ _ _ _ _ _ _ _ _
              Train loss: {train_loss_ep:.5f}
              Val   loss: {val_loss_ep:.5f}
              Train f1: {train_f1_ep:.5f}
              Val   f1: {val_f1_ep:.5f}""")

    return train_f1, val_f1, train_loss, val_loss


# ==============================================================
def plot_loss_curves(train_metric, val_metric, train_loss, val_loss):
    """
    Plots metric and loss on two plots
    """
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_metric, c="green", label="train_metric")
    plt.plot(val_metric, c="green", linestyle="dashed", label="val_metric")
    plt.title("Metric")
    plt.legend()

    plt.subplot(1, 2, 2)

    plt.plot(train_loss, c="blue", label="train_loss")
    plt.plot(val_loss, c="blue", linestyle="dashed", label="val_loss")
    plt.legend()
    plt.title("Loss")
    plt.show()
