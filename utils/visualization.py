"""
Visualization utilities for training and evaluation
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid


def plot_training_curves(train_losses, train_accs, val_losses, val_accs, save_path=None):
    """
    Plot training and validation loss/accuracy curves
    
    Args:
        train_losses: List of training losses
        train_accs: List of training accuracies
        val_losses: List of validation losses
        val_accs: List of validation accuracies
        save_path: Path to save the plot (optional)
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f'Training curves saved to {save_path}')
    
    plt.show()


def show_sample_images(dataset, num_images=16, classes=None, save_path=None):
    """
    Display sample images from the dataset
    
    Args:
        dataset: PyTorch dataset
        num_images: Number of images to display
        classes: List of class names (optional)
        save_path: Path to save the plot (optional)
    """
    # Get random samples
    indices = np.random.choice(len(dataset), num_images, replace=False)
    
    images = []
    labels = []
    
    for idx in indices:
        img, label = dataset[idx]
        images.append(img)
        labels.append(label)
    
    # Create grid
    images = torch.stack(images)
    grid = make_grid(images, nrow=4, normalize=True, padding=2)
    
    # Convert to numpy and transpose
    grid_np = grid.cpu().numpy().transpose((1, 2, 0))
    
    # Denormalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    grid_np = std * grid_np + mean
    grid_np = np.clip(grid_np, 0, 1)
    
    # Plot
    plt.figure(figsize=(12, 12))
    plt.imshow(grid_np)
    plt.axis('off')
    
    if classes:
        title = 'Sample Images\n' + ', '.join([classes[labels[i]] if i < len(labels) else '' 
                                               for i in range(min(8, len(labels)))])
        plt.title(title)
    else:
        plt.title('Sample Images from Dataset')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f'Sample images saved to {save_path}')
    
    plt.show()


def visualize_predictions(model, dataset, device='cuda', num_images=16, classes=None, save_path=None):
    """
    Visualize model predictions on sample images
    
    Args:
        model: PyTorch model
        dataset: PyTorch dataset
        device: Device to run inference on
        num_images: Number of images to display
        classes: List of class names (optional)
        save_path: Path to save the plot (optional)
    """
    model.eval()
    
    # Get random samples
    indices = np.random.choice(len(dataset), num_images, replace=False)
    
    images = []
    true_labels = []
    pred_labels = []
    
    with torch.no_grad():
        for idx in indices:
            img, label = dataset[idx]
            images.append(img)
            true_labels.append(label)
            
            # Predict
            img_batch = img.unsqueeze(0).to(device)
            output = model(img_batch)
            pred = output.argmax(dim=1).item()
            pred_labels.append(pred)
    
    # Create grid
    images = torch.stack(images)
    grid = make_grid(images, nrow=4, normalize=True, padding=2)
    
    # Convert to numpy and transpose
    grid_np = grid.cpu().numpy().transpose((1, 2, 0))
    
    # Denormalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    grid_np = std * grid_np + mean
    grid_np = np.clip(grid_np, 0, 1)
    
    # Plot
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(grid_np)
    ax.axis('off')
    
    # Add predictions as text
    if classes:
        title = 'Model Predictions (Green=Correct, Red=Wrong)\n'
        for i in range(min(num_images, len(true_labels))):
            correct = true_labels[i] == pred_labels[i]
            color = 'green' if correct else 'red'
            true_class = classes[true_labels[i]] if true_labels[i] < len(classes) else str(true_labels[i])
            pred_class = classes[pred_labels[i]] if pred_labels[i] < len(classes) else str(pred_labels[i])
            
            if i < 8:  # Only show first 8 in title
                title += f'{i+1}: True={true_class}, Pred={pred_class} | '
        
        ax.set_title(title, color='black', fontsize=10)
    else:
        ax.set_title('Model Predictions')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f'Predictions visualization saved to {save_path}')
    
    plt.show()
