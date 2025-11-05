"""
Evaluation script for Tiny-ImageNet classification with CustomNet
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.custom_net import CustomNet
from dataset.tiny_imagenet import get_tiny_imagenet_datasets
from utils.train_utils import validate, load_checkpoint
from utils.visualization import visualize_predictions


def evaluate(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load datasets
    print("Loading datasets...")
    _, val_dataset = get_tiny_imagenet_datasets(args.data_dir, args.input_size)
    
    # Create data loader
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Initialize model
    print("Initializing model...")
    model = CustomNet(num_classes=args.num_classes).to(device)
    
    # Load checkpoint
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)  # Dummy optimizer for loading
        epoch, best_acc = load_checkpoint(model, optimizer, args.checkpoint, device)
    else:
        print("No checkpoint provided. Evaluating with random weights.")
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Evaluate
    print("\nEvaluating model...")
    print("=" * 80)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    print("=" * 80)
    print(f'\nFinal Validation Loss: {val_loss:.6f}')
    print(f'Final Validation Accuracy: {val_acc:.2f}%')
    
    # Visualize predictions
    if args.visualize:
        print("\nGenerating prediction visualizations...")
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Get class names if available
        classes = val_dataset.classes if hasattr(val_dataset, 'classes') else None
        
        visualize_predictions(
            model, val_dataset, device, 
            num_images=args.num_vis_images,
            classes=classes,
            save_path=os.path.join(args.output_dir, 'predictions.png')
        )
        
        print(f"Visualizations saved to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate CustomNet on Tiny-ImageNet')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='tiny-imagenet/tiny-imagenet-200',
                        help='Path to Tiny-ImageNet dataset')
    parser.add_argument('--input_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--num_classes', type=int, default=200,
                        help='Number of classes')
    
    # Model parameters
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    
    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Visualization parameters
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize model predictions')
    parser.add_argument('--num_vis_images', type=int, default=16,
                        help='Number of images to visualize')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save outputs')
    
    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
