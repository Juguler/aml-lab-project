"""
Main training script for Tiny-ImageNet classification with CustomNet
Integrated with Weights & Biases for experiment tracking
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.custom_net import CustomNet
from dataset.tiny_imagenet import get_tiny_imagenet_datasets, prepare_tiny_imagenet_val
from utils.train_utils import train_one_epoch, validate, save_checkpoint
from utils.visualization import plot_training_curves, show_sample_images

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")


def main(args):
    # Initialize Weights & Biases
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                'epochs': args.num_epochs,
                'batch_size': args.batch_size,
                'learning_rate': args.lr,
                'momentum': args.momentum,
                'weight_decay': args.weight_decay,
                'architecture': 'CustomNet',
                'dataset': 'Tiny-ImageNet',
                'num_classes': args.num_classes,
                'input_size': args.input_size,
            },
            tags=['lab03', 'tiny-imagenet', 'custom-cnn']
        )
        # Watch the model (logs gradients and parameters)
        print("✓ Weights & Biases initialized")
    else:
        wandb = None
        if args.use_wandb:
            print("Warning: Wandb requested but not available")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Prepare dataset if needed
    if args.prepare_data:
        print("Preparing validation dataset...")
        prepare_tiny_imagenet_val(args.data_dir)
    
    # Load datasets
    print("Loading datasets...")
    train_dataset, val_dataset = get_tiny_imagenet_datasets(args.data_dir, args.input_size)
    
    # Show sample images if requested
    if args.show_samples:
        print("Displaying sample images...")
        show_sample_images(train_dataset, num_images=16, 
                          save_path=os.path.join(args.output_dir, 'sample_images.png'))
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True
    )
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
    
    # Watch model with wandb
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.watch(model, log='all', log_freq=100)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=args.lr, 
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler (optional)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=args.lr_step_size, 
        gamma=args.lr_gamma
    )
    
    # Training history
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_acc = 0
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Training loop
    print(f"\nStarting training for {args.num_epochs} epochs...")
    print("=" * 80)
    
    for epoch in range(1, args.num_epochs + 1):
        print(f"\nEpoch {epoch}/{args.num_epochs}")
        print("-" * 80)
        
        # Train
        train_loss, train_acc = train_one_epoch(
            epoch, model, train_loader, criterion, optimizer, device,
            wandb_log=wandb if args.use_wandb and WANDB_AVAILABLE else None
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device,
            wandb_log=wandb if args.use_wandb and WANDB_AVAILABLE else None,
            epoch=epoch
        )
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Update learning rate
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        # Log learning rate to wandb
        if args.use_wandb and WANDB_AVAILABLE:
            wandb.log({'learning_rate': current_lr, 'epoch': epoch})
        
        # Save checkpoint if best model
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch, best_acc, checkpoint_path)
            
            # Log best accuracy to wandb
            if args.use_wandb and WANDB_AVAILABLE:
                wandb.run.summary['best_val_accuracy'] = best_acc
                wandb.save(checkpoint_path)
        
        # Save latest checkpoint
        if epoch % args.save_freq == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(model, optimizer, epoch, best_acc, checkpoint_path)
    
    print("\n" + "=" * 80)
    print(f'Training completed! Best validation accuracy: {best_acc:.2f}%')
    
    # Plot training curves
    print("\nGenerating training curves...")
    plot_training_curves(
        train_losses, train_accs, val_losses, val_accs,
        save_path=os.path.join(args.output_dir, 'training_curves.png')
    )
    
    # Log final training curves to wandb
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.log({
            'training_curves': wandb.Image(os.path.join(args.output_dir, 'training_curves.png'))
        })
        wandb.finish()
        print("✓ Wandb run completed and logged")
    
    print(f"\nAll outputs saved to {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CustomNet on Tiny-ImageNet')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='tiny-imagenet/tiny-imagenet-200',
                        help='Path to Tiny-ImageNet dataset')
    parser.add_argument('--prepare_data', action='store_true',
                        help='Prepare validation dataset')
    parser.add_argument('--input_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--num_classes', type=int, default=200,
                        help='Number of classes')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='Weight decay')
    parser.add_argument('--lr_step_size', type=int, default=5,
                        help='Learning rate decay step size')
    parser.add_argument('--lr_gamma', type=float, default=0.1,
                        help='Learning rate decay factor')
    
    # System parameters
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save outputs')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='Save checkpoint every N epochs')
    
    # Visualization
    parser.add_argument('--show_samples', action='store_true',
                        help='Show sample images before training')
    
    # Weights & Biases parameters
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for experiment tracking')
    parser.add_argument('--wandb_project', type=str, default='tiny-imagenet-lab03',
                        help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Wandb run name (optional)')
    
    args = parser.parse_args()
    main(args)
