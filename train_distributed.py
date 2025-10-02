import argparse
import os, sys
import torch
import numpy as np
from tqdm import tqdm
import datetime, random
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets_load_rawboost import SVDD2024

# Distributed training imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# you can import the model from Models directory any model you want to use.
from Models.model_attention import Model

from utils import seed_worker, set_seed, compute_eer

class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, use_logits=True):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.use_logits = use_logits
        
    def forward(self, logits, targets):
        if self.use_logits:
            bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        else:
            bce_loss = F.binary_cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

def setup_distributed(args):
    """Setup distributed training"""
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl')
    
    if args.distributed:
        device = torch.device(f"cuda:{args.local_rank}")
    else:
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    return device

def create_distributed_dataloaders(train_dataset, dev_dataset, args):
    """Create distributed dataloaders"""
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory
        )
        
        dev_sampler = DistributedSampler(dev_dataset, shuffle=False)
        dev_loader = DataLoader(
            dev_dataset,
            batch_size=args.batch_size,
            sampler=dev_sampler,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory
        )
        return train_loader, dev_loader, train_sampler
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            pin_memory=args.pin_memory
        )
        dev_loader = DataLoader(
            dev_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker
        )
        return train_loader, dev_loader, None

def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

def main_distributed(args):
    """Main function for distributed training"""
    device = setup_distributed(args)
    set_seed(42)
    
    # Create datasets
    path = args.base_dir
    train_dataset = SVDD2024(path, partition="train", args=args, algo=args.algo)
    dev_dataset = SVDD2024(path, partition="dev", args=args, algo=args.algo)
    
    # Create dataloaders
    train_loader, dev_loader, train_sampler = create_distributed_dataloaders(train_dataset, dev_dataset, args)
    
    # Create model
    model = Model(args, device).to(device)
    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=False)
    
    # Create optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-6, weight_decay=1e-4, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-9)
    
    # Setup logging (only on main process)
    writer = None
    checkpoint_dir = None
    if not args.distributed or (args.distributed and args.local_rank == 0):
        log_dir = os.path.join(args.log_dir, args.placeHolder_name_Checkpoints)
        os.makedirs(log_dir, exist_ok=True)
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(log_dir, current_time)
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        checkpoint_dir = os.path.join(log_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save config
        with open(os.path.join(log_dir, "config.json"), "w") as f:
            f.write(str(vars(args)))
    
    criterion = BinaryFocalLoss()
    best_val_eer = 1.0
    
    # Training loop
    for epoch in range(args.epochs):
        # Set sampler epoch for distributed training
        if args.distributed and train_sampler:
            train_sampler.set_epoch(epoch)
        
        model.train()
        pos_samples, neg_samples = [], []
        total_loss = 0
        
        # Progress bar only on main process
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", 
                           disable=args.distributed and args.local_rank != 0)
        
        for i, batch in enumerate(progress_bar):
            if args.debug and i > 20:
                break
            x, label, _ = batch
            x = x.to(device)
            label = label.to(device)
            soft_label = label.float() * 0.9 + 0.05
            pred = model(x)
            loss = criterion(pred, soft_label.unsqueeze(1))
            
            pos_samples.append(pred[label == 1].detach().cpu().numpy())
            neg_samples.append(pred[label == 0].detach().cpu().numpy())
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            total_loss += loss.item()
            
            # Update progress bar
            if not args.distributed or args.local_rank == 0:
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Log to tensorboard (only main process)
            if writer and i % 100 == 0:
                writer.add_scalar("Loss/train", loss.item(), epoch * len(train_loader) + i)
        
        scheduler.step()
        
        # Compute training metrics (only on main process for logging)
        if not args.distributed or args.local_rank == 0:
            if pos_samples and neg_samples:
                train_eer = compute_eer(np.concatenate(pos_samples), np.concatenate(neg_samples))[0]
                if writer:
                    writer.add_scalar("LR/train", scheduler.get_last_lr()[0], epoch)
                    writer.add_scalar("EER/train", train_eer, epoch)
                    writer.add_scalar("Loss/train_epoch", total_loss / len(train_loader), epoch)
        
        # Validation
        model.eval()
        val_loss = 0
        pos_samples, neg_samples = [], []
        
        val_progress_bar = tqdm(dev_loader, desc=f"Validation", 
                               disable=args.distributed and args.local_rank != 0)
        
        with torch.no_grad():
            for i, batch in enumerate(val_progress_bar):
                if args.debug and i > 20:
                    break
                x, label, _ = batch
                x = x.to(device)
                label = label.to(device)
                pred = model(x)
                soft_label = label.float() * 0.9 + 0.05
                loss = criterion(pred, soft_label.unsqueeze(1))
                
                pos_samples.append(pred[label == 1].detach().cpu().numpy())
                neg_samples.append(pred[label == 0].detach().cpu().numpy())
                val_loss += loss.item()
                
                # Update progress bar
                if not args.distributed or args.local_rank == 0:
                    val_progress_bar.set_postfix({'val_loss': f'{loss.item():.4f}'})
        
        # Compute validation metrics and save models (only on main process)
        if not args.distributed or args.local_rank == 0:
            val_loss /= len(dev_loader)
            if pos_samples and neg_samples:
                val_eer = compute_eer(np.concatenate(pos_samples), np.concatenate(neg_samples))[0]
                
                if writer:
                    writer.add_scalar("Loss/val", val_loss, epoch)
                    writer.add_scalar("EER/val", val_eer, epoch)
                
                if val_eer < best_val_eer:
                    best_val_eer = val_eer
                    model_state = model.module.state_dict() if args.distributed else model.state_dict()
                    torch.save(model_state, os.path.join(checkpoint_dir, f"best_model.pt"))
                
                if epoch % 2 == 0:  # Save every 2 epochs
                    model_state = model.module.state_dict() if args.distributed else model.state_dict()
                    torch.save(model_state, os.path.join(checkpoint_dir, f"model_{epoch}_EER_{val_eer:.4f}.pt"))
                
                print(f"Epoch {epoch+1}/{args.epochs} - Val Loss: {val_loss:.4f}, Val EER: {val_eer:.4f}, Best EER: {best_val_eer:.4f}")
        
        # Synchronize all processes
        if args.distributed:
            dist.barrier()
    
    # Cleanup
    if writer:
        writer.close()
    
    if args.distributed:
        cleanup_distributed()

def main(args):
    # Set the seed for reproducibility
    set_seed(42)
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Create the dataset
    path = args.base_dir
    train_dataset = SVDD2024(path, partition="train",args=args, algo=args.algo)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, worker_init_fn=seed_worker,
                          pin_memory=args.pin_memory)
    
    dev_dataset = SVDD2024(path, partition="dev",args=args, algo=args.algo)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, worker_init_fn=seed_worker)
    
    # Create the model
    model = Model(args, device).to(device)

    # Create the optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-6, weight_decay=1e-4, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-9)

    # Create the directory for the logs
    log_dir = os.path.join(args.log_dir, args.placeHolder_name_Checkpoints)
    os.makedirs(log_dir, exist_ok=True)
    
    # get current time
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(log_dir, current_time)
    os.makedirs(log_dir, exist_ok=True)

    # Create the summary writer
    writer = SummaryWriter(log_dir=log_dir)
    
    # Create the directory for the checkpoints
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save config for reproducibility
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        f.write(str(vars(args)))
        
    criterion = BinaryFocalLoss()
    
    best_val_eer = 1.0

    # Train the model
    for epoch in range(args.epochs):
        model.train()
        pos_samples, neg_samples = [], []
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")):
            if args.debug and i > 20:
                break
            x, label, _ = batch
            x = x.to(device)
            label = label.to(device)
            soft_label = label.float() * 0.9 + 0.05
            pred = model(x)
            loss = criterion(pred, soft_label.unsqueeze(1))
            pos_samples.append(pred[label == 1].detach().cpu().numpy())
            neg_samples.append(pred[label == 0].detach().cpu().numpy())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            writer.add_scalar("Loss/train", loss.item(), epoch * len(train_loader) + i)
        scheduler.step()
        writer.add_scalar("LR/train", scheduler.get_last_lr()[0], epoch * len(train_loader) + i)
        writer.add_scalar("EER/train", compute_eer(np.concatenate(pos_samples), np.concatenate(neg_samples))[0], epoch)
        
        
        model.eval()
        val_loss = 0
        pos_samples, neg_samples = [], []
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dev_loader, desc=f"Validation")):
                if args.debug and i > 20:
                    break
                x, label, _ = batch
                x = x.to(device)
                label = label.to(device)
                pred = model(x)
                soft_label = label.float() * 0.9 + 0.05
                loss = criterion(pred, soft_label.unsqueeze(1))
                pos_samples.append(pred[label == 1].detach().cpu().numpy())
                neg_samples.append(pred[label == 0].detach().cpu().numpy())
                val_loss += loss.item()
            val_loss /= len(dev_loader)
            val_eer = compute_eer(np.concatenate(pos_samples), np.concatenate(neg_samples))[0]
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("EER/val", val_eer, epoch)
            if val_eer < best_val_eer:
                best_val_eer = val_eer
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"best_model.pt"))
            if epoch % 2 == 0: # Save every 2 epochs
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"model_{epoch}_EER_{val_eer}.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True, help="The base directory of the dataset.")
    parser.add_argument("--epochs", type=int, default=30, help="The number of epochs to train.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode.")
    parser.add_argument("--gpu", type=int, default=0, help="The GPU to use.")
    parser.add_argument("--placeHolder_name_Checkpoints", type=str, default="hemlata_rawboost5_wavlm", help="Name where checkpoints are saved.")
    parser.add_argument("--batch_size", type=int, default=40, help="The batch size for training.")
    parser.add_argument("--num_workers", type=int, default=12, help="The number of workers for the data loader.")
    parser.add_argument("--log_dir", type=str, default="logs", help="The directory for the logs.")

    # rawboost 
    parser.add_argument('--algo', type=int, default=8, 
                    help='Rawboost algos discriptions. 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
                          5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2) .[default=0]')

    # LnL_convolutive_noise parameters 
    parser.add_argument('--nBands', type=int, default=5, 
                    help='number of notch filters.The higher the number of bands, the more aggresive the distortions is.[default=5]')
    parser.add_argument('--minF', type=int, default=20, 
                    help='minimum centre frequency [Hz] of notch filter.[default=20] ')
    parser.add_argument('--maxF', type=int, default=8000, 
                    help='maximum centre frequency [Hz] (<sr/2)  of notch filter.[default=8000]')
    parser.add_argument('--minBW', type=int, default=100, 
                    help='minimum width [Hz] of filter.[default=100] ')
    parser.add_argument('--maxBW', type=int, default=1000, 
                    help='maximum width [Hz] of filter.[default=1000] ')
    parser.add_argument('--minCoeff', type=int, default=10, 
                    help='minimum filter coefficients. More the filter coefficients more ideal the filter slope.[default=10]')
    parser.add_argument('--maxCoeff', type=int, default=100, 
                    help='maximum filter coefficients. More the filter coefficients more ideal the filter slope.[default=100]')
    parser.add_argument('--minG', type=int, default=0, 
                    help='minimum gain factor of linear component.[default=0]')
    parser.add_argument('--maxG', type=int, default=0, 
                    help='maximum gain factor of linear component.[default=0]')
    parser.add_argument('--minBiasLinNonLin', type=int, default=5, 
                    help=' minimum gain difference between linear and non-linear components.[default=5]')
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20, 
                    help=' maximum gain difference between linear and non-linear components.[default=20]')
    parser.add_argument('--N_f', type=int, default=5, 
                    help='order of the (non-)linearity where N_f=1 refers only to linear components.[default=5]')

    # ISD_additive_noise parameters
    parser.add_argument('--P', type=int, default=10, 
                    help='Maximum number of uniformly distributed samples in [%].[defaul=10]')
    parser.add_argument('--g_sd', type=int, default=2, 
                    help='gain parameters > 0. [default=2]')

    # SSI_additive_noise parameters
    parser.add_argument('--SNRmin', type=int, default=10, 
                    help='Minimum SNR value for coloured additive noise.[defaul=10]')
    parser.add_argument('--SNRmax', type=int, default=40, 
                    help='Maximum SNR value for coloured additive noise.[defaul=40]')
    
    ##===================================================Rawboost data augmentation ======================================================================#
    
    # CUDA and training specific parameters
    parser.add_argument('--num_workers', type=int, default=4,
                    help='Number of workers for data loading')
    parser.add_argument('--pin_memory', action='store_true',
                    help='Pin memory for faster GPU transfer')
    parser.add_argument('--cudnn_benchmark', action='store_true',
                    help='Enable cudnn benchmark for faster training')
    
    # Distributed training arguments
    parser.add_argument('--distributed', action='store_true',
                    help='Enable distributed training')
    parser.add_argument('--local_rank', type=int, default=0,
                    help='Local rank for distributed training')
    
    args = parser.parse_args()
    
    # Set CUDA specific settings
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = args.cudnn_benchmark
        torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Ampere
    
    # Use distributed training if specified
    if args.distributed:
        main_distributed(args)
    else:
        main(args)