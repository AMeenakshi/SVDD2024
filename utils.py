import random
import numpy as np
import torch
from sklearn.metrics import roc_curve

def set_seed(seed):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def seed_worker(worker_id):
    """Seed DataLoader workers."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def compute_eer(y_true, y_pred):
    """Compute Equal Error Rate (EER)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    fnr = 1 - tpr
    
    # Find the threshold where FAR (FPR) equals FRR (FNR)
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    
    return eer