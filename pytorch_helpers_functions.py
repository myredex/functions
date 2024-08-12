# Data Preparation
# =========================================
## calculate_mean_std - calculates mean and std for the images in folder

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

    from torchvision import datasets
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    import numpy as np

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

mean, std = calculate_mean_std('images')
