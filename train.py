import torch
import torch.optim as optim
import torch.cuda.amp as amp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import os
from pathlib import Path
import importlib
import logging
import random

import model
from config import Config
from dataset import RadarDataset
from loss import RadarPillarsLoss
from maps import calculate_3d_map_for_folder, post_process_predictions, save_predictions
from torch.cuda.amp import autocast, GradScaler


def set_seed(seed):
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_criterion(config):
    
    return RadarPillarsLoss(config)

def freeze_layers(model, layers_to_freeze):
    
    for name, param in model.named_parameters():
        for layer_name in layers_to_freeze:
            if layer_name in name:
                param.requires_grad = False
                logging.info(f"Freezing layer: {name}")

class EarlyStopping:

    def __init__(self, patience=7, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return False
        
        if self.mode == 'min':
            if val_loss < self.best_loss - self.min_delta:
                self.best_loss = val_loss
                self.counter = 0
            else:
                self.counter += 1
        else:  # mode == 'max'
            if val_loss > self.best_loss + self.min_delta:
                self.best_loss = val_loss
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
            return True
        return False

def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, epoch, config):
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_reg_loss = 0
    
    # AMP 초기화
    scaler = amp.GradScaler(enabled=config.use_amp)
    
    progress_bar = tqdm(train_loader, desc=f'Training Epoch {epoch}')
    
    for batch_idx, (points, center_coords, mask, cls_targets, reg_targets, reg_masks, _) in enumerate(progress_bar):
        # Move data to device
        points = points.to(config.device)
        center_coords = center_coords.to(config.device)
        mask = mask.to(config.device)
        cls_targets = cls_targets.to(config.device)
        reg_targets = reg_targets.to(config.device)
        reg_masks = reg_masks.to(config.device)
        
        '''
        # AMP forward pass
        with amp.autocast(enabled=config.use_amp):
            cls_preds, reg_preds = model(points, center_coords, mask)
            loss, loss_dict = criterion(
                cls_preds, reg_preds,
                cls_targets, reg_targets, reg_masks
            )
        #print("cls_target\n -----", cls_targets)
        #print("cls_preds\n -----", cls_preds)
        #print("reg_targets\n -----", reg_targets)
        #print("reg_preds\n -----", reg_preds)
        
        # AMP backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        '''
        
        if config.use_amp:
            scaler = GradScaler()
        
        with autocast(enabled=config.use_amp, dtype=torch.float16):
            cls_preds, reg_preds = model(points, center_coords, mask)
            loss, loss_dict = criterion(
                cls_preds, reg_preds,
                cls_targets, reg_targets, reg_masks
            )
        
        if config.use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()   
        
        # Update statistics
        total_loss += loss_dict['total_loss'].item()
        total_cls_loss += loss_dict['cls_loss'].item()
        total_reg_loss += loss_dict['reg_loss'].item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': total_loss / (batch_idx + 1),
            'cls_loss': total_cls_loss / (batch_idx + 1),
            'reg_loss': total_reg_loss / (batch_idx + 1),
            'lr': optimizer.param_groups[0]['lr']
        })
    
    num_batches = len(train_loader)
    avg_loss = total_loss / num_batches
    avg_cls_loss = total_cls_loss / num_batches
    avg_reg_loss = total_reg_loss / num_batches
    
    return avg_loss, avg_cls_loss, avg_reg_loss

def validate(model, val_loader, criterion, config, writer, epoch):
    model.eval()
    total_loss = 0
    total_cls_loss = 0
    total_reg_loss = 0
    all_predictions = []
    
    with torch.no_grad():
        with amp.autocast(enabled=config.use_amp):
            pbar = tqdm(val_loader, desc='Validation')
            for batch_idx, (points, center_coords, mask, cls_targets, reg_targets, reg_masks, file_names) in enumerate(pbar):
                # Move data to device
                points = points.to(config.device)
                center_coords = center_coords.to(config.device)
                mask = mask.to(config.device)
                cls_targets = cls_targets.to(config.device)
                reg_targets = reg_targets.to(config.device)
                reg_masks = reg_masks.to(config.device)
                
                # Forward pass
                cls_preds, reg_preds = model(points, center_coords, mask)
                
                # Calculate loss
                loss, loss_dict = criterion(
                    cls_preds, reg_preds,
                    cls_targets, reg_targets, reg_masks
                )
                
                # Update metrics
                total_loss += loss.item()
                total_cls_loss += loss_dict['cls_loss'].item()
                total_reg_loss += loss_dict['reg_loss'].item()
                
                # Post-process predictions
                batch_predictions = post_process_predictions(cls_preds, reg_preds, config)
                all_predictions.extend(batch_predictions)
    
                # Save predictions
                save_predictions(all_predictions, os.path.join(config.save_dir, 'predictions'), file_names)
    
    # Calculate mAP
    mAP_3d = calculate_3d_map_for_folder(
        config.val_label_path,
        os.path.join(config.save_dir, 'predictions'),
        iou_threshold=0.5
    )
    
    # Log validation metrics to tensorboard
    writer.add_scalar('Val/Loss', total_loss / len(val_loader), epoch)
    writer.add_scalar('Val/Cls_Loss', total_cls_loss / len(val_loader), epoch)
    writer.add_scalar('Val/Reg_Loss', total_reg_loss / len(val_loader), epoch)
    writer.add_scalar('Val/mAP_3D', mAP_3d, epoch)
    
    return total_loss / len(val_loader), total_cls_loss / len(val_loader), total_reg_loss / len(val_loader), mAP_3d

def save_checkpoint(model, optimizer, scheduler, epoch, config, is_best=False):
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config': config
    }
    
    checkpoint_path = os.path.join(
        config.save_dir,
        f"{config.model_name}_epoch_{epoch}.pth"
    )
    torch.save(checkpoint, checkpoint_path)
    
    if is_best:
        best_path = os.path.join(
            config.save_dir,
            f"{config.model_name}_best.pth"
        )
        torch.save(checkpoint, best_path)

def main():
    # Set random seed
    set_seed(42)  
    
    # Load configuration
    config = Config()
    config.print_config()
    
    # Setup tensorboard
    writer = SummaryWriter(Path(config.save_dir) / 'tensorboard')
    
    # Create model and move to device
    radar_pillars = model.create_model(config)
    radar_pillars = radar_pillars.to(config.device)
    
    # Freeze layers if specified
    if hasattr(config, 'layers_to_freeze'):
        freeze_layers(radar_pillars, config.layers_to_freeze)
    
    # Create datasets
    train_dataset = RadarDataset(config, split='train')
    val_dataset = RadarDataset(config, split='val')
    
    # Create dataloaders with custom collate_fn
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=train_dataset.collate_fn,
        pin_memory=config.pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=val_dataset.collate_fn,
        pin_memory=config.pin_memory
    )
    
    # Learning rate scheduler
    initial_lr = 0.0003
    max_lr = 0.003
    
    optimizer = torch.optim.Adam(radar_pillars.parameters(), lr=initial_lr)
    
    steps_per_epoch = len(train_loader)
    total_steps = config.num_epochs * steps_per_epoch
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy='cos',
        cycle_momentum=False,
        div_factor=max_lr/initial_lr,
        final_div_factor=1e4
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_min_delta
    )
    
    criterion = get_criterion(config)
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(config.num_epochs):
        logging.info(f"\nEpoch {epoch+1}/{config.num_epochs}")
        
        # Train
        train_loss, train_cls_loss, train_reg_loss = train_one_epoch(
            radar_pillars, train_loader, criterion, optimizer, scheduler, epoch, config
        )
        
        # Log training metrics
        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Train/Cls_Loss', train_cls_loss, epoch)
        writer.add_scalar('Train/Reg_Loss', train_reg_loss, epoch)
        writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Validate
        val_loss, val_cls_loss, val_reg_loss, mAP_3d = validate(
            radar_pillars, val_loader, criterion, config, writer, epoch
        )
        
        # Check for best model
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        
        # Save checkpoint
        save_checkpoint(
            radar_pillars, 
            optimizer, 
            scheduler, 
            epoch,
            config,
            is_best=is_best
        )
        
        # Early stopping check
        if early_stopping(val_loss):
            logging.info(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # Print epoch results
        print(f"Train Loss: {train_loss:.4f} (cls: {train_cls_loss:.4f}, reg: {train_reg_loss:.4f})")
        print(f"Val Loss: {val_loss:.4f} (cls: {val_cls_loss:.4f}, reg: {val_reg_loss:.4f})")
        print(f"mAP 3D: {mAP_3d:.4f}")
    
    writer.close()

if __name__ == '__main__':
    main()