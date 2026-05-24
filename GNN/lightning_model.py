import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingWarmRestarts, SequentialLR
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryAveragePrecision
from codex_gnn_model import CODEXVetoGNN

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
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate, 
            weight_decay=1e-4 
        )
        
        warmup = LinearLR(
            optimizer,
            start_factor=1e-6 / self.learning_rate,
            end_factor=1.0,
            total_iters=2
        )
        
        cosine = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=15,
            T_mult=2,
            eta_min=1e-6
        )
        
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[2]
        )
        
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler}}