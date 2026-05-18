import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC
from codex_gnn_model import CODEXVetoGNN


class WarmupReduceLROnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    """
    ReduceLROnPlateau preceded by a linear warmup phase.
    For the first `warmup_epochs` epochs the LR increases linearly
    from `warmup_start_lr` to the base LR of the optimizer.
    After warmup, delegates to the standard ReduceLROnPlateau logic.
    """
    def __init__(self, optimizer, warmup_epochs=2, warmup_start_lr=1e-6,
                 factor=0.3, patience=4, min_lr=1e-6):
        super().__init__(optimizer, mode='min', factor=factor,
                         patience=patience, min_lr=min_lr)
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self._warmup_step = 0

    def step(self, metrics=None):
        self._warmup_step += 1
        if self._warmup_step <= self.warmup_epochs:
            frac = self._warmup_step / max(1, self.warmup_epochs)
            lr = self.warmup_start_lr + frac * (self.base_lr - self.warmup_start_lr)
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr
        else:
            super().step(metrics)


class CODEXLightning(pl.LightningModule):
    def __init__(self, pos_weight_val, learning_rate=5e-4,
                 model_kwargs=None):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        # Allow overriding model construction args
        if model_kwargs is None:
            model_kwargs = {}
        self.model = CODEXVetoGNN(**model_kwargs)

        self.register_buffer('pos_weight', torch.tensor([pos_weight_val]))
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

        self.train_acc = BinaryAccuracy()
        self.train_auc = BinaryAUROC()
        self.val_acc = BinaryAccuracy()
        self.val_auc = BinaryAUROC()

    def forward(self, data):
        return self.model(data).squeeze(-1)

    def training_step(self, batch, batch_idx):
        logits = self(batch)
        loss = self.criterion(logits, batch.y.float())

        probs = torch.sigmoid(logits)
        self.train_acc(probs, batch.y)
        self.train_auc(probs, batch.y)

        self.log('train_loss', loss, on_step=False, on_epoch=True,
                 batch_size=batch.num_graphs, sync_dist=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True,
                 batch_size=batch.num_graphs, sync_dist=True)
        self.log('train_auc', self.train_auc, on_step=False, on_epoch=True,
                 batch_size=batch.num_graphs, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        loss = self.criterion(logits, batch.y.float())

        probs = torch.sigmoid(logits)
        self.val_acc(probs, batch.y)
        self.val_auc(probs, batch.y)

        self.log('val_loss', loss, prog_bar=True,
                 batch_size=batch.num_graphs, sync_dist=True)
        self.log('val_acc', self.val_acc, prog_bar=True,
                 batch_size=batch.num_graphs, sync_dist=True)
        self.log('val_auc', self.val_auc, prog_bar=True,
                 batch_size=batch.num_graphs, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=1e-5
        )

        scheduler = WarmupReduceLROnPlateau(
            optimizer,
            warmup_epochs=2,
            warmup_start_lr=1e-6,
            factor=0.3,
            patience=4,
            min_lr=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }