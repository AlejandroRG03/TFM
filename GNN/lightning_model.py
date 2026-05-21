import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryAveragePrecision
from codex_gnn_model import CODEXVetoGNN

class WarmupReduceLROnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    """
    ReduceLROnPlateau preceded by a linear warmup phase.
    During the first `warmup_epochs`, the LR increases linearly
    from `warmup_start_lr` to the optimizer's base LR.
    After warmup, it delegates to the standard ReduceLROnPlateau logic.
    """
    def __init__(self, optimizer, warmup_epochs=2, warmup_start_lr=1e-6,
                 factor=0.3, patience=4, min_lr=1e-6):
        super().__init__(optimizer, mode='min', factor=factor,
                         patience=patience, min_lr=min_lr)
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self._warmup_step = 0

    # Added epoch=None to match official PyTorch signature
    def step(self, metrics=None, epoch=None): 
        self._warmup_step += 1
        if self._warmup_step <= self.warmup_epochs:
            frac = self._warmup_step / max(1, self.warmup_epochs)
            lr = self.warmup_start_lr + frac * (self.base_lr - self.warmup_start_lr)
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr
        else:
            super().step(metrics, epoch)

class CODEXLightning(pl.LightningModule):
    def __init__(self, pos_weight_val, learning_rate=1e-3, model_kwargs=None):
        super().__init__()
        self.save_hyperparameters() # Automatically saves lr, pos_weight and kwargs
        self.learning_rate = learning_rate
        
        if model_kwargs is None:
            model_kwargs = {}
            
        # Instantiate the pure PyTorch model passing dynamic arguments
        self.model = CODEXVetoGNN(**model_kwargs)
        
        # Loss function with background/signal weights
        self.register_buffer('pos_weight', torch.tensor([pos_weight_val]))
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        
        # Official PyTorch metrics (GPU optimized)
        self.train_acc = BinaryAccuracy()
        self.train_auc = BinaryAUROC()
        self.train_prc = BinaryAveragePrecision()  # <-- Precision-Recall Curve (Vital)
        
        self.val_acc = BinaryAccuracy()
        self.val_auc = BinaryAUROC()
        self.val_prc = BinaryAveragePrecision()    # <-- Precision-Recall Curve (Vital)

    def forward(self, data):
        # Forward pass only calls the underlying model
        return self.model(data).squeeze(-1)

    def training_step(self, batch, batch_idx):
        # 1. Prediction and Loss
        logits = self(batch)
        loss = self.criterion(logits, batch.y.float())
        
        # 2. Metrics
        probs = torch.sigmoid(logits)
        y_int = batch.y.long()
        self.train_acc(probs, y_int)
        self.train_auc(probs, y_int)
        self.train_prc(probs, y_int)
        
        # 3. Logging (automatically synced for TensorBoard / WandB)
        self.log('train_loss', loss, on_step=False, on_epoch=True, batch_size=batch.num_graphs, sync_dist=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, batch_size=batch.num_graphs, sync_dist=True)
        self.log('train_auc', self.train_auc, on_step=False, on_epoch=True, batch_size=batch.num_graphs, sync_dist=True)
        self.log('train_prc', self.train_prc, on_step=False, on_epoch=True, batch_size=batch.num_graphs, sync_dist=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        # Identical to training_step, but for validation
        logits = self(batch)
        loss = self.criterion(logits, batch.y.float())
        
        probs = torch.sigmoid(logits)
        y_int = batch.y.long()
        self.val_acc(probs, y_int)
        self.val_auc(probs, y_int)
        self.val_prc(probs, y_int)
        
        # prog_bar=True shows these metrics in the terminal progress bar
        self.log('val_loss', loss, prog_bar=True, batch_size=batch.num_graphs, sync_dist=True)
        self.log('val_acc', self.val_acc, prog_bar=True, batch_size=batch.num_graphs, sync_dist=True)
        self.log('val_auc', self.val_auc, prog_bar=True, batch_size=batch.num_graphs, sync_dist=True)
        self.log('val_prc', self.val_prc, prog_bar=True, batch_size=batch.num_graphs, sync_dist=True)

    def configure_optimizers(self):
        # AdamW Optimizer: Includes weight_decay to force small weights and regularize the GNN
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate, 
            weight_decay=1e-4 
        )
        
        # Custom Scheduler: Prevents the network from jumping uncontrollably in the first epochs
        scheduler = WarmupReduceLROnPlateau(
            optimizer, 
            warmup_epochs=2,
            warmup_start_lr=1e-6,
            factor=0.5, 
            patience=3
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }