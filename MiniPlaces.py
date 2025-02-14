import os
from torch.utils.data import Dataset
from PIL import Image

class MiniPlaces(Dataset):
    def __init__(self, root_dir, split, transform=None, label_dict=None):
        """
        Args:
            root_dir (str): Root directory for the MiniPlaces images.
            split (str): Split to use ('train' or 'val').
            transform (callable, optional): Optional data transformation to
                                            apply to the images.
            label_dict (dict, optional): Optional dictionary mapping integer
                                         labels to class names.
        """
        assert split in ['train', 'val']
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.filenames = []
        self.labels = []

        # valid.txt does not contain class information. This allows passing
        # label_dict from training set to validation set.
        self.label_dict = label_dict if label_dict is not None else {}

        with open(f'{root_dir}/{split}.txt') as file: # Load text file
          for line in file: # Read line-by-line
            split_line = line.rstrip().split(' ') # rstrip removes newline
            file_name = split_line[0] # Extract file name from text file line
            label_int = int(split_line[1]) # Extract label int from split line

            self.filenames.append(file_name) # Store the image filenames
            self.labels.append(label_int)    # and labels

            if (split == 'train'):
              class_name = file_name.split('/')[2] # Extract class name
              self.label_dict[label_int] = class_name # Store in self.label_dict

    def __len__(self):
        """
        Returns:
            int: Number of images in the dataset.
        """
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            tuple: Tuple containing the image and its label.
        """
        image = Image.open(f'{self.root_dir}/images/{self.filenames[idx]}') # Load image
        if (self.transform is not None):
          image = self.transform(image) # Preprocess image

        label = self.labels[idx] # Retrieve label

        return image, label
    
class MiniPlacesTest(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Root directory for the MiniPlaces images.
            transform (callable, optional): Optional data transformation to
                                            apply to the images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.filenames = []
        self.ids = []

        for image in os.listdir(f'{root_dir}/images/test'):
          filename = os.fsdecode(image) # Get filename of image
          self.filenames.append(filename) # Store the image filenames
          self.ids.append(filename.split('.')[0]) # Remove file extension

    def __len__(self):
        """
        Returns:
            int: Number of images in the dataset.
        """
        return len(self.ids)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            tuple: Tuple containing the image and its id.
        """
        image = Image.open(f'{self.root_dir}/images/test/{self.filenames[idx]}')
        if (self.transform is not None):
          image = self.transform(image) # Preprocess image

        id = self.ids[idx] # Retrieve id

        return image, id
    
class MiniPlacesSR(Dataset):
    def __init__(self, root_dir, split, base_transform=None, downscale=None):
        """
        Args:
            root_dir (str): Root directory for the MiniPlaces images.
            split (str): Split to use ('train' or 'val').
            base_transform (callable, optional): Transformation to apply to the
                                                 images and ground truths.
            downscale (callable, optional): Downscale transformation to apply
                                            to the images.
        """
        assert split in ['train', 'val']
        self.root_dir = root_dir
        self.split = split
        self.base_transform = base_transform
        self.downscale = downscale
        self.filenames = []

        with open(f'{root_dir}/{split}.txt') as file: # Load text file
          for line in file: # Read line-by-line
            split_line = line.rstrip().split(' ') # rstrip removes newline
            file_name = split_line[0] # Extract file name from text file line
            self.filenames.append(file_name) # Store the image filenames

    def __len__(self):
        """
        Returns:
            int: Number of images in the dataset.
        """
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            tuple: Tuple containing the image and its ground truth image.
        """
        image = Image.open(f'{self.root_dir}/images/{self.filenames[idx]}') # Load image

        # Ground truth is the image without downscaling
        if (self.base_transform is not None):
            ground_truth = self.base_transform(image)

        if (self.downscale is not None):
          image = self.downscale(ground_truth) # Downscale image

        return image, ground_truth