"""
Tiny-ImageNet dataset preparation utilities
"""
import os
import shutil
from torchvision.datasets import ImageFolder
import torchvision.transforms as T


def prepare_tiny_imagenet_val(data_dir='tiny-imagenet/tiny-imagenet-200'):
    """
    Adjust the format of the val split of the dataset to be used with ImageFolder.
    
    Args:
        data_dir: Root directory of the Tiny-ImageNet dataset
    """
    val_annotations_path = os.path.join(data_dir, 'val/val_annotations.txt')
    val_images_dir = os.path.join(data_dir, 'val/images')
    
    if not os.path.exists(val_annotations_path):
        print(f"Val annotations file not found at {val_annotations_path}")
        return
    
    with open(val_annotations_path) as f:
        for line in f:
            fn, cls, *_ = line.split('\t')
            class_dir = os.path.join(data_dir, f'val/{cls}')
            os.makedirs(class_dir, exist_ok=True)
            
            src = os.path.join(val_images_dir, fn)
            dst = os.path.join(class_dir, fn)
            
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copyfile(src, dst)
    
    # Remove the old images directory if it exists
    if os.path.exists(val_images_dir):
        shutil.rmtree(val_images_dir)
    
    print("Validation split reorganized successfully!")


def get_transforms(input_size=224):
    """
    Get standard transforms for Tiny-ImageNet
    
    Args:
        input_size: Input image size (default: 224)
    
    Returns:
        torchvision.transforms.Compose object
    """
    transform = T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform


def get_tiny_imagenet_datasets(data_dir='tiny-imagenet/tiny-imagenet-200', input_size=224):
    """
    Get Tiny-ImageNet train and validation datasets
    
    Args:
        data_dir: Root directory of the Tiny-ImageNet dataset
        input_size: Input image size (default: 224)
    
    Returns:
        tuple: (train_dataset, val_dataset)
    """
    transform = get_transforms(input_size)
    
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    train_dataset = ImageFolder(root=train_dir, transform=transform)
    val_dataset = ImageFolder(root=val_dir, transform=transform)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Number of classes: {len(train_dataset.classes)}")
    
    return train_dataset, val_dataset
