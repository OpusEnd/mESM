# train.py - Training script for UNetSLSM

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from datetime import datetime
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# Add the directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from our modular package
from src.model import AttentionUNet, device
from src.data import CellDataset


class DiceLoss(nn.Module):
    """Dice Loss for segmentation tasks"""
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target, weight=None):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        if weight is not None:
            weight_flat = weight.view(-1)
            intersection = (pred_flat * target_flat * weight_flat).sum()
            union = (pred_flat * weight_flat).sum() + (target_flat * weight_flat).sum()
        else:
            intersection = (pred_flat * target_flat).sum()
            union = pred_flat.sum() + target_flat.sum()
        
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice_score


class BCEDiceLoss(nn.Module):
    """Combined BCE and Dice Loss for segmentation tasks"""
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)
        
    def forward(self, pred, target, weight=None):
        bce_loss = self.bce(pred, target)
        pred_sigmoid = torch.sigmoid(pred)
        dice_loss = self.dice(pred_sigmoid, target, weight)
        loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        return loss


def get_transforms(config):
    """Get data augmentation transforms"""
    if config.get('data', {}).get('augmentations', False):
        train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            ToTensorV2(),
        ])
        
        val_transform = A.Compose([
            ToTensorV2(),
        ])
    else:
        train_transform = A.Compose([
            ToTensorV2(),
        ])
        val_transform = A.Compose([
            ToTensorV2(),
        ])
    
    return train_transform, val_transform


def validate(model, val_loader, criterion, device, threshold=0.5):
    """Validate the model on the validation set"""
    model.eval()
    val_loss = 0.0
    val_dice = 0.0
    
    with torch.no_grad():
        for images, labels, weight_maps in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            weight_maps = weight_maps.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels, weight_maps)
            val_loss += loss.item()
            
            # Calculate Dice score
            pred = torch.sigmoid(outputs) > threshold
            pred = pred.float()
            dice_score = 2.0 * (pred * labels).sum() / (pred.sum() + labels.sum() + 1e-8)
            val_dice += dice_score.item()
    
    return val_loss / len(val_loader), val_dice / len(val_loader)


def train_model(config_path):
    """Train the model based on the configuration"""    # Load config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"Loaded configuration from {config_path}")
    
    # Set device
    device_name = config.get('hardware', {}).get('device', 'auto')
    if device_name == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_name)
    print(f"Using device: {device}")
    
    # Create model
    model = AttentionUNet(
        in_channels=config.get('model', {}).get('in_channels', 3),
        out_channels=config.get('model', {}).get('out_channels', 1)
    ).to(device)
    
    # Create optimizer
    optimizer_name = config.get('train', {}).get('optimizer', 'adam').lower()
    if optimizer_name == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.get('train', {}).get('learning_rate', 0.001),
            weight_decay=config.get('train', {}).get('weight_decay', 0.0001)
        )
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.get('train', {}).get('learning_rate', 0.001),
            momentum=0.9,
            weight_decay=config.get('train', {}).get('weight_decay', 0.0001)
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    # Create loss function
    loss_name = config.get('train', {}).get('loss_function', 'bce_dice_loss').lower()
    if loss_name == 'bce_dice_loss':
        criterion = BCEDiceLoss()
    elif loss_name == 'dice_loss':
        criterion = DiceLoss()
    elif loss_name == 'bce_loss':
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")
    
    # Create learning rate scheduler
    scheduler_name = config.get('train', {}).get('lr_scheduler', 'reduce_on_plateau').lower()
    if scheduler_name == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.get('train', {}).get('lr_scheduler_factor', 0.5),
            patience=config.get('train', {}).get('lr_scheduler_patience', 10),
            verbose=True
        )
    elif scheduler_name == 'step_lr':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.5
        )
    else:
        scheduler = None
    
    # Data transforms
    train_transform, val_transform = get_transforms(config)
    
    # Create datasets
    train_img_dir = config.get('data', {}).get('train_img_dir', 'data/cell_dataset/img_dir/train')
    train_mask_dir = config.get('data', {}).get('train_mask_dir', 'data/cell_dataset/mask_dir/train')
    val_img_dir = config.get('data', {}).get('val_img_dir', 'data/cell_dataset/img_dir/val')
    val_mask_dir = config.get('data', {}).get('val_mask_dir', 'data/cell_dataset/mask_dir/val')
    
    # Get image and mask paths
    train_img_paths = [os.path.join(train_img_dir, filename) for filename in os.listdir(train_img_dir) if os.path.isfile(os.path.join(train_img_dir, filename))]
    train_mask_paths = [os.path.join(train_mask_dir, filename) for filename in os.listdir(train_mask_dir) if os.path.isfile(os.path.join(train_mask_dir, filename))]
    val_img_paths = [os.path.join(val_img_dir, filename) for filename in os.listdir(val_img_dir) if os.path.isfile(os.path.join(val_img_dir, filename))]
    val_mask_paths = [os.path.join(val_mask_dir, filename) for filename in os.listdir(val_mask_dir) if os.path.isfile(os.path.join(val_mask_dir, filename))]
    
    # Sort paths to ensure correspondence
    train_img_paths.sort()
    train_mask_paths.sort()
    val_img_paths.sort()
    val_mask_paths.sort()
    
    # Create datasets
    train_dataset = CellDataset(train_img_paths, train_mask_paths, transform=train_transform, mode='train')
    val_dataset = CellDataset(val_img_paths, val_mask_paths, transform=val_transform, mode='val')
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('train', {}).get('batch_size', 8),
        shuffle=True,
        num_workers=config.get('hardware', {}).get('num_workers', 4),
        pin_memory=config.get('hardware', {}).get('pin_memory', True)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('val', {}).get('batch_size', 8),
        shuffle=False,
        num_workers=config.get('hardware', {}).get('num_workers', 4),
        pin_memory=config.get('hardware', {}).get('pin_memory', True)
    )
    
    # Create checkpoint directory
    experiment_name = config.get('logging', {}).get('experiment_name', 'unet_cell_segmentation')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = config.get('model', {}).get('checkpoint_dir', 'runs/unet_training')
    checkpoint_dir = os.path.join(checkpoint_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize training variables
    epochs = config.get('train', {}).get('epochs', 100)
    best_val_loss = float('inf')
    patience = config.get('train', {}).get('early_stopping_patience', 15)
    patience_counter = 0
    save_best_only = config.get('model', {}).get('save_best_only', True)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        # Initialize progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        # Training step
        for images, labels, weight_maps in pbar:
            images = images.to(device)
            labels = labels.to(device)
            weight_maps = weight_maps.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels, weight_maps)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        # Calculate average training loss
        train_loss /= len(train_loader)
        
        # Validate every validation_interval epochs
        val_interval = config.get('val', {}).get('validation_interval', 1)
        if (epoch + 1) % val_interval == 0:
            val_loss, val_dice = validate(
                model, val_loader, criterion, device,
                threshold=config.get('val', {}).get('threshold', 0.5)
            )
            
            # Print validation metrics
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
            
            # Update learning rate scheduler if using ReduceLROnPlateau
            if scheduler_name == 'reduce_on_plateau':
                scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save model
                if save_best_only:
                    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))
                    print(f"Saved best model with validation loss: {best_val_loss:.4f}")
                else:
                    torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth"))
                    print(f"Saved model at epoch {epoch+1}")
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Update learning rate scheduler if not using ReduceLROnPlateau
        if scheduler is not None and scheduler_name != 'reduce_on_plateau':
            scheduler.step()
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "final_model.pth"))
    print(f"Model training completed. Final model saved to {checkpoint_dir}/final_model.pth")
    
    return model, checkpoint_dir


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train UNetSLSM Cell Segmentation Model")
    parser.add_argument("--config", type=str, default="config/train_config.yaml", help="Path to training config file")
    args = parser.parse_args()
    
    train_model(args.config)
