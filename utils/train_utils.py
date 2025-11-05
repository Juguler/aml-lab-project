"""
Training and validation utilities
"""
import torch


def train_one_epoch(epoch, model, train_loader, criterion, optimizer, device='cuda'):
    """
    Train the model for one epoch
    
    Args:
        epoch: Current epoch number
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on (default: 'cuda')
    
    Returns:
        tuple: (train_loss, train_accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()  # Zero the gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, targets)  # Compute loss
        loss.backward()  # Backpropagate
        optimizer.step()  # Update weights

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')
    
    return train_loss, train_accuracy


def validate(model, val_loader, criterion, device='cuda'):
    """
    Validate the model
    
    Args:
        model: PyTorch model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on (default: 'cuda')
    
    Returns:
        tuple: (val_loss, val_accuracy)
    """
    model.eval()
    val_loss = 0
    correct, total = 0, 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, targets)  # Compute loss

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100. * correct / total

    print(f'Validation Loss: {val_loss:.6f} Acc: {val_accuracy:.2f}%')
    
    return val_loss, val_accuracy


def save_checkpoint(model, optimizer, epoch, best_acc, filepath):
    """
    Save model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        best_acc: Best validation accuracy
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
    }
    torch.save(checkpoint, filepath)
    print(f'Checkpoint saved to {filepath}')


def load_checkpoint(model, optimizer, filepath, device='cuda'):
    """
    Load model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        filepath: Path to checkpoint
        device: Device to load on (default: 'cuda')
    
    Returns:
        tuple: (epoch, best_acc)
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    best_acc = checkpoint['best_acc']
    
    print(f'Checkpoint loaded from {filepath}')
    print(f'Resuming from epoch {epoch} with best accuracy {best_acc:.2f}%')
    
    return epoch, best_acc
