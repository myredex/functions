# Classes to create custom dataset
# Content 
# class image_text_dataset(Dataset):

from PIL import Image
from torch.utils.data import Dataset


class image_text_dataset(Dataset):
    """
    This class creates pytorch dataset from dataframe. Output is Image/Text pair
    Args & Kwargs:
        dataframe: pandas dataframe with targets column
        images_path: str path to the images
        image_name_column: str name of the column with names of images
        text_column: name of the column with targets if empy doesnt return target
        transform: if transform of the data is needed
    Example:
        dataset = image_text_dataset(dataframe=train_dataframe,
                             images_path="/content/dawlance_images",
                             image_name_column="image_name",
                             text_column="text",
                             transform=transform)
    """
    
    def __init__(self, 
                 images_path: str,
                 dataframe: pd.DataFrame,
                 image_name_column: str,
                 text_column: str,
                 target_column: str,
                 transform=None):
        
        self.images_path = images_path
        self.df = dataframe
        self.image_name_column = image_name_column
        self.text_column = text_column
        self.target_column = target_column
        self.transform = transform
        
        self.images = dataframe[self.image_name_column].values
        self.texts = dataframe[self.text_column].values
        self.targets = dataframe[self.target_column].values
        
    def __len__(self):
        return len(self.df) # return lenth of the dataframe
    
    def __getitem__(self, index):
        
        # Get image path
        img_path = self.images_path + "/" + str(self.images[index])

        # Open image and decode it 
        img = Image.open(img_path).convert('RGB')

        # Transform image
        if self.transform:
            img = self.transform(img)
        else:
            # Convert to tensor if no transform is provided
            img = transforms.ToTensor()(img)
            
        text = self.texts[index]
        target = self.targets[index]
        return img, text, target       
