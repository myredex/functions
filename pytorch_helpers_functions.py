import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter


# Data Preparation
# =========================================
## calculate_mean_std - calculates mean and std for the images in folder
## split_pytorch_dataset - splits dataset into two stratified parts

# Model training
# =========================================
## train_step - runs model learning for one epoch
## val_step - evaluates model for epoch
## get_predictions - performs predictions using test_dataloader (for kaggle usage)
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


from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset
from collections import Counter

def split_pytorch_dataset(dataset:torch.utils.data.Dataset,
                          test_size:float=0.2,
                          seed=42):
    """
    Splits highly unballanced dataset into two parts.
    Args:
        dataset: pytorch dataset
        test_size: float value to define size of test set
    Returns:
        two pytorch subsets
    Example:
        train_dataset, test_dataset = split_pytorch_dataset(dataset=dataset, test_size=0.2)
    """
    # Get targets
    targets = dataset.targets
    
    # Split only once
    sss = StratifiedShuffleSplit(n_splits=1,
                                 test_size=test_size,
                                 random_state=seed)
    
    for train_idx, test_idx in sss.split(list(range(len(targets))), targets):
        train_dataset = Subset(dataset, train_idx)
        test_dataset = Subset(dataset, test_idx)
        
    # Count number of classes in each subset
    train_targets = [dataset.targets[i] for i in train_idx]
    test_targets = [dataset.targets[i] for i in test_idx]
    
    train_classes = Counter(train_targets)
    test_classes = Counter(test_targets)
    print(f"Dataset splitted into:")
    print(f"Train dataset contains of: {train_classes}")
    print(f"Test dataset contains of: {test_classes}")
    print(f"Lenth of datasets: {len(train_dataset)}, {len(test_dataset)}")
        
    return train_dataset, test_dataset

# ==============================================================
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device):

    """
    Trains PyTorch model for one epoch.
    Args:
        model: PyTorch model
        dataloader: A DataLoader instance
        metric_function: for example accuracy, f1, etc. or custom 
        gets y_true and y_pred arrays 
        loss_fn: Loss function
        optimizer: Optimizer

    Returns:
        A tuple of training loss and predictions
    """
    train_loss = 0
    
    y_preds, y_trues = [], []
    
    # Turn on train mode
    model.train()
    for batch, (X, y) in tqdm(enumerate(dataloader)):
        
        # Send data to device
        X, y = X.to(device), y.to(torch.float32).to(device)

        # Forward pass
        y_preds_batch = model(X)
        
        # Calculate the losss
        loss = loss_fn(torch.squeeze(y_preds_batch, dim=1), y)
        
        train_loss += loss.item()
        # Optimizer zero grad
        optimizer.zero_grad()
        # Loss backward
        loss.backward()
        # Optimizer step
        optimizer.step()
        
        # Collect preds and trues
        y_trues.extend(y.detach().cpu().numpy())
        y_preds.extend(torch.squeeze(y_preds_batch, dim=1).detach().cpu().numpy())

    train_loss = train_loss / len(dataloader)

    return train_loss, np.array(y_preds), np.array(y_trues)



# ==============================================================
def val_step(model: torch.nn.Module,
             dataloader: torch.utils.data.DataLoader,
             loss_fn: torch.nn.Module,
             device: torch.device):

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
    val_loss = 0
    y_trues, y_preds = [], []
    
    # Turn on eval mode
    model.eval()
    
    with torch.inference_mode():

        for batch, (X, y) in enumerate(dataloader):

            # Send data to device
            X, y = X.to(device), y.to(torch.float32).to(device)

            # Forward pass
            y_preds_batch = model(X) 
             
            # Calculate the losss
            loss = loss_fn(torch.squeeze(y_preds_batch, dim=1), y)

            val_loss += loss.item()
            
            y_trues.extend(y.detach().cpu().numpy())
            y_preds.extend(torch.squeeze(y_preds_batch, dim=1).detach().cpu().numpy())
        
    val_loss = val_loss / len(dataloader)

    return val_loss, np.array(y_preds), np.array(y_trues)

# ==============================================================

def get_predictions(model: torch.nn.Module,
                    dataloader: torch.utils.data.DataLoader,
                    device: torch.device):

    """
    Performs PyTorch model's predictions
    Args:
        model: PyTorch model
        dataloader: A DataLoader instance 

    Returns:
        predictons in np.array
    """
    y_preds = []
    
    # Turn on eval mode
    model.eval()
    
    with torch.inference_mode():

        for batch, (X) in enumerate(dataloader):

            # Send data to device
            X = X.to(device)

            # Make predictions
            y_preds_batch = model(X) 
            
            # Put preds into list in np format
            y_preds.extend(torch.squeeze(y_preds_batch, dim=1).detach().cpu().numpy())

    return np.array(y_preds)
                 

# ==============================================================
def train_model(model: torch.nn.Module,
                metric_function,
                train_dataloader: torch.utils.data.DataLoader,
                val_dataloader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                loss_fn: torch.nn.Module,
                epochs: int,
                device: torch.device):

    """
    Trains PyTorch model for N epochs
    Args: 
        model: Pytorch model
        metric_function: Some function that gets y_true and y_pred as input and returns score
        train_dataloader: Pytorch dataloader
        val_dataloader: Pytorch validation dataloader
        optimizer: Pytorch optimizer
        loss_fn: Loss function
        epochs: number of epochs to train
        device: CPU or CUDA
        
    Returns:
        Dictionaries of Train Loss, Val Loss, Train Acc, Val Accuracy.
        If test dataloader in input it also returns final test Loss and Test accuracy
    """
    # Move model to device
    model.to(device)
    print(f"Device: {device}")

    train_metrics, val_metrics, train_loss, val_loss = [], [], [], []

    for epoch in tqdm(range(epochs)):
        # Run train and collect results in list
        train_loss_ep, train_preds_ep, train_true_labels = train_step(model=model,
                                                    dataloader=train_dataloader,
                                                    loss_fn=loss_fn,
                                                    optimizer=optimizer,
                                                    device=device)
        train_loss.append(train_loss_ep)

        # Run validation and collect preds in list
        val_loss_ep, val_preds_ep, val_true_labels = val_step(model=model,
                                              dataloader=val_dataloader,
                                              loss_fn=loss_fn,
                                              device=device)
        val_loss.append(val_loss_ep)
        
        # Count metric for epoch
        train_metric_ep = metric_function(y_true=train_true_labels,
                                          y_pred=train_preds_ep)
        val_metric_ep = metric_function(y_true=val_true_labels,
                                        y_pred=val_preds_ep)
        
        train_metrics.append(train_metric_ep)
        val_metrics.append(val_metric_ep)
        
        print(f"""Epoch {epoch+1}_ _ _ _ _ _ _ _ _ _
              Train loss: {train_loss_ep:.5f}
              Val   loss: {val_loss_ep:.5f}
              Train metric: {train_metric_ep:.5f}
              Val   metric: {val_metric_ep:.5f}""")
    
    return train_metrics, val_metrics, train_loss, val_loss


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
