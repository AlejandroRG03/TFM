import pytorch_lightning as pl
from torch.utils.data import IterableDataset, DataLoader
import torch

class MyDataset(IterableDataset):
    def __iter__(self):
        print("--> Dataset __iter__ called")
        yield torch.tensor([1])
        yield torch.tensor([2])

class MyCallback(pl.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        print("--> Callback on_train_epoch_start")

class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)
    def training_step(self, batch, batch_idx):
        return self.layer(batch.float()).sum()
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.1)

if __name__ == '__main__':
    ds = MyDataset()
    dl = DataLoader(ds, num_workers=2, persistent_workers=True)
    model = MyModel()
    trainer = pl.Trainer(max_epochs=2, callbacks=[MyCallback()], accelerator='cpu', logger=False, enable_checkpointing=False)
    trainer.fit(model, dl)
