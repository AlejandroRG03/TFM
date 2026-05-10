import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC
from codex_gnn_model import CODEXVetoGNN

class CODEXLightning(pl.LightningModule):
    def __init__(self, pos_weight_val, learning_rate=1e-3, k=8):
        super().__init__()
        self.save_hyperparameters() # Automatically saves lr and pos_weight
        self.learning_rate = learning_rate
        
        # Instantiate the pure PyTorch model
        self.model = CODEXVetoGNN(k=k)
        
        # Loss function with background/signal weights
        self.register_buffer('pos_weight', torch.tensor([pos_weight_val]))
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        
        # Official PyTorch metrics (GPU optimized)
        self.train_acc = BinaryAccuracy()
        self.train_auc = BinaryAUROC()
        self.val_acc = BinaryAccuracy()
        self.val_auc = BinaryAUROC()

    def forward(self, data):
        # The forward pass only calls the underlying model
        return self.model(data).squeeze(-1)

    def training_step(self, batch, batch_idx):
        # 1. Prediction and Loss
        logits = self(batch)
        loss = self.criterion(logits, batch.y.float())
        
        # 2. Metrics
        probs = torch.sigmoid(logits)
        self.train_acc(probs, batch.y)
        self.train_auc(probs, batch.y)
        
        # 3. Logging (automatically saved for TensorBoard)
        self.log('train_loss', loss, on_step=False, on_epoch=True, batch_size=batch.num_graphs, sync_dist=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, batch_size=batch.num_graphs, sync_dist=True)
        self.log('train_auc', self.train_auc, on_step=False, on_epoch=True, batch_size=batch.num_graphs, sync_dist=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        # Identical to training_step, but for validation
        logits = self(batch)
        loss = self.criterion(logits, batch.y.float())
        
        probs = torch.sigmoid(logits)
        self.val_acc(probs, batch.y)
        self.val_auc(probs, batch.y)
        
        # prog_bar=True displays these metrics in the terminal progress bar
        self.log('val_loss', loss, prog_bar=True, batch_size=batch.num_graphs, sync_dist=True)
        self.log('val_acc', self.val_acc, prog_bar=True, batch_size=batch.num_graphs, sync_dist=True)
        self.log('val_auc', self.val_auc, prog_bar=True, batch_size=batch.num_graphs, sync_dist=True)

    def configure_optimizers(self):
        # Adam optimizer with the provided learning rate
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        # Scheduler to reduce LR when validation loss plateaus
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
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