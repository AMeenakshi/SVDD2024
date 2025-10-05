# Improved train_distillation.py addressing architectural mismatch
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

# Import improved distillation model
from improved_knowledge_distillation import ImprovedKnowledgeDistillationModel, train_with_improved_distillation
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

def main(args):
    set_seed(42)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create datasets
    path = args.base_dir
    train_dataset = SVDD2024(path, partition="train", args=args, algo=args.algo)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers, worker_init_fn=seed_worker,
                             pin_memory=args.pin_memory)
    
    dev_dataset = SVDD2024(path, partition="dev", args=args, algo=args.algo)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, 
                           num_workers=args.num_workers, worker_init_fn=seed_worker)
    
    # Create improved distillation model (teacher + compressed student with same architecture)
    print("Creating distillation model...")
    model = ImprovedKnowledgeDistillationModel(args, device).to(device)
    
    # Load pre-trained teacher weights
    if args.teacher_checkpoint:
        print(f"Loading teacher from {args.teacher_checkpoint}")
        teacher_state = torch.load(args.teacher_checkpoint, map_location=device)
        model.teacher.load_state_dict(teacher_state)
        print("Teacher loaded successfully!")
    else:
        print("Warning: No teacher checkpoint provided. Using randomly initialized teacher.")
    
    # Freeze teacher parameters
    for param in model.teacher.parameters():
        param.requires_grad = False
    print(f"Teacher parameters frozen. Student parameters: {sum(p.numel() for p in model.student.parameters() if p.requires_grad):,}")
    
    # Only optimize student and projection parameters
    student_params = list(model.student.parameters()) + \
                    list(model.ssl_projector.parameters()) + \
                    list(model.attention_projector.parameters()) + \
                    list(model.graph_projector_s.parameters()) + \
                    list(model.graph_projector_t.parameters())
    
    optimizer = optim.AdamW(student_params, lr=args.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs//3, eta_min=1e-8)

    # Setup logging
    log_dir = os.path.join(args.log_dir, f"improved_distillation_{args.placeHolder_name_Checkpoints}")
    os.makedirs(log_dir, exist_ok=True)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(log_dir, current_time)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save config for reproducibility
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        import json
        config_dict = vars(args)
        config_dict['student_params'] = sum(p.numel() for p in model.student.parameters())
        config_dict['teacher_params'] = sum(p.numel() for p in model.teacher.parameters())
        json.dump(config_dict, f, indent=2)

    criterion = BinaryFocalLoss()
    best_val_eer = 1.0

    print(f"Starting training for {args.epochs} epochs...")
    
    # Training loop
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Training phase
        loss_dict, pos_samples, neg_samples = train_with_improved_distillation(
            model, train_loader, optimizer, criterion, device, epoch, writer
        )
        
        scheduler.step()
        
        # Log training metrics
        train_eer = compute_eer(np.concatenate(pos_samples), np.concatenate(neg_samples))[0]
        writer.add_scalar("EER/train", train_eer, epoch)
        writer.add_scalar("LR/train", scheduler.get_last_lr()[0], epoch)
        
        print(f"Training - EER: {train_eer:.4f}, "
              f"Total Loss: {loss_dict['total']:.4f}, "
              f"Hard Loss: {loss_dict['hard']:.4f}, "
              f"Soft Loss: {loss_dict['soft']:.4f}, "
              f"Feature Loss: {loss_dict['feature']:.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0
        pos_samples, neg_samples = [], []
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dev_loader, desc="Validation")):
                if args.debug and i > 20:
                    break
                    
                x, label, _ = batch
                x = x.to(device)
                label = label.to(device)
                
                # Only use student for validation
                pred = model.student(x)
                soft_label = label.float() * 0.9 + 0.05
                loss = criterion(pred, soft_label.unsqueeze(1))
                
                pos_samples.append(pred[label == 1].detach().cpu().numpy())
                neg_samples.append(pred[label == 0].detach().cpu().numpy())
                val_loss += loss.item()
                
        val_loss /= len(dev_loader)
        val_eer = compute_eer(np.concatenate(pos_samples), np.concatenate(neg_samples))[0]
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("EER/val", val_eer, epoch)
        
        print(f"Validation - EER: {val_eer:.4f}, Loss: {val_loss:.4f}")
        
        # Save best model
        if val_eer < best_val_eer:
            best_val_eer = val_eer
            torch.save(model.student.state_dict(), 
                      os.path.join(checkpoint_dir, f"best_student_model.pt"))
            print(f"New best model saved! EER: {val_eer:.4f}")
            
        # Save regular checkpoints
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            torch.save({
                'student_state_dict': model.student.state_dict(),
                'projector_states': {
                    'ssl': model.ssl_projector.state_dict(),
                    'attention': model.attention_projector.state_dict(),
                    'graph_s': model.graph_projector_s.state_dict(),
                    'graph_t': model.graph_projector_t.state_dict(),
                },
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_eer': val_eer
            }, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}_EER_{val_eer:.4f}.pt"))
    
    print(f"\nTraining completed! Best validation EER: {best_val_eer:.4f}")
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Improved Knowledge Distillation for SVDD2024")
    parser.add_argument("--base_dir", type=str, required=True, 
                       help="The base directory of the dataset.")
    parser.add_argument("--teacher_checkpoint", type=str, required=True, 
                       help="Path to pre-trained teacher model.")
    parser.add_argument("--epochs", type=int, default=25, 
                       help="The number of epochs to train.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, 
                       help="Learning rate for student training.")
    parser.add_argument("--debug", action="store_true", 
                       help="Run in debug mode.")
    parser.add_argument("--gpu", type=int, default=0, 
                       help="The GPU to use.")
    parser.add_argument("--placeHolder_name_Checkpoints", type=str, 
                       default="compressed_sea_distilled", 
                       help="Name where checkpoints are saved.")
    parser.add_argument("--batch_size", type=int, default=24, 
                       help="The batch size for training (reduced for distillation).")
    parser.add_argument("--num_workers", type=int, default=4, 
                       help="Number of workers for data loading.")
    parser.add_argument("--pin_memory", action="store_true", 
                       help="Pin memory for faster data loading.")
    parser.add_argument("--log_dir", type=str, default="./logs", 
                       help="Directory for logs.")
    parser.add_argument("--algo", type=int, default=5, 
                       help="RawBoost algorithm.")
    
    args = parser.parse_args()
    main(args)