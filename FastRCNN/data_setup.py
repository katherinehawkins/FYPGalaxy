# Import Files from one directory above
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
sys.path.append('../') 
from base_dataset import RadioGalaxyNET

def collate_fn(batch):
    images = []
    annotations = []
    for img, ann in batch:
        images.append(img)
        annotations.append(ann)
    return images, annotations

def create_dataloaders(
        train_dir:str,
        val_dir:str,
        test_dir:str,
        train_ann_dir: str,
        val_ann_dir: str,
        test_ann_dir: str,
        transform: transforms.Compose,
        batch_size: int,
        num_worker: int = 0
):
    """Creates training and testing DataLoaders.

  Takes in a training directory and testing directory path and turns
  them into PyTorch Datasets and then into PyTorch DataLoaders.

  Args:
    train_dir: Path to training directory.
    val_dir: Path to validating directory.
    test_dir: Path to testing directory.
    train_ann_dir: Path to training COCO annotations.
    val_ann_dir: Path to validation COCO annotations.
    test_ann_dir: Path to testing COCO annotations.
    transform: torchvision transforms to perform on training and testing data.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of workers per DataLoader.

  Returns:
    A tuple of (train_dataloader, test_dataloader, class_names).
    Where class_names is a list of the target classes.
    Example usage:
      train_dataloader, val_dataloader, test_dataloader = \
        = def create_dataloaders(
        train_dir:path/to/train_dir,
        val_dir:path/to/val_dir,
        test_dir:path/to/test_dir,
        train_ann_dir: path/to/train_ann_dir,
        val_ann_dir: path/to/val_ann_dir,
        test_ann_dir: path/to/test_ann_dir,
        transform: transforms.Compose,
        batch_size: int,
        num_worker: int = 0
)
  """
    # Define the dataset class
    train_dataset = RadioGalaxyNET(root=train_dir,
                          annFile=train_ann_dir,
                          transforms=transform
                          ) 
    val_dataset = RadioGalaxyNET(root=val_dir,
                          annFile=val_ann_dir,
                          transforms=transform
                          )
    test_dataset = RadioGalaxyNET(root=test_dir,
                          annFile=test_ann_dir,
                          transforms=transform
                          )
     # Turn images into data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_worker,
        collate_fn=collate_fn,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_worker,
        collate_fn=collate_fn,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False, # don't need to shuffle test data
        num_workers=num_worker,
        collate_fn=collate_fn,
    )
    return train_dataloader, val_dataloader, test_dataloader

