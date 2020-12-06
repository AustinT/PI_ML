""" File to train neural network on path integral data """

import argparse
import torch
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
import utils


class RhoMLP(pl.LightningModule):
    """ class for neural network model """

    def __init__(
        self,
        hidden_sizes,
        linden_path,
        n_in=4,
        batch_size=128,
        n_train=320000,
        lr=1e-3,
        lr_factor: float = 0.3,
        lr_patience: int = 4,
    ):
        super().__init__()

        self.save_hyperparameters()

        # Initialize hidden network
        layers = []
        last = n_in
        for h in hidden_sizes:
            layers.append(torch.nn.Linear(last, h))
            layers.append(torch.nn.ELU())
            last = h
        layers = layers[:-1]  # pop off last activation
        self.net = torch.nn.Sequential(*layers)

        # Training stuff
        self.batch_size = batch_size
        self.n_train = n_train
        self.loss_func = torch.nn.MSELoss()
        self.lr = lr
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience

        # Data stuff
        self.linden_path = linden_path

    def forward(self, x):
        y = self.net(x)
        return torch.exp(y)

    def configure_optimizers(self):
        adam = torch.optim.Adam(self.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            adam, factor=self.lr_factor, patience=self.lr_patience, min_lr=1e-5
        )
        return dict(optimizer=adam, lr_scheduler=sched, monitor="loss/val")

    def _get_xy_data(self):
        # Get random data
        rho_eep = utils.get_rho_eep_func(str(self.linden_path))
        rand_nums = utils.get_random_z_phi(self.n_train)
        rho_vals = rho_eep(*rand_nums.T)
        assert rho_vals.shape[0] == rand_nums.shape[0]

        return rand_nums, rho_vals

    def _get_dataloader(self):

        rand_nums, rho_vals = self._get_xy_data()

        # Convert to tensor dataset for serving
        rand_nums = torch.as_tensor(rand_nums, dtype=torch.float)
        rho_vals = torch.as_tensor(rho_vals.reshape(-1, 1), dtype=torch.float)
        dset = TensorDataset(rand_nums, rho_vals)
        return DataLoader(
            dset, batch_size=self.batch_size, drop_last=True, num_workers=3
        )

    def train_dataloader(self):
        return self._get_dataloader()

    def val_dataloader(self):
        return self._get_dataloader()

    def _get_loss(self, batch):
        x, y = batch
        y_pred = self(x)
        return self.loss_func(y_pred, y)

    def training_step(self, batch, batch_idx):
        return self._get_loss(batch)

    def validation_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        self.log("loss/val", loss, prog_bar=True)
        return loss


if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 256, 128, 128, 64, 64])
    parser.add_argument("-l", "--learning_rate", type=float, default=1e-3)
    parser.add_argument("-b", "--batch_size", type=int, default=128)
    parser.add_argument("-e", "--num_epochs", type=int, default=100)
    parser.add_argument("-n", "--n_train", type=int, default=320000)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    # Sample neural network
    pl.seed_everything(0)
    model = RhoMLP(
        hidden_sizes=args.hidden + [1],
        linden_path="linden.out",
        lr=args.learning_rate,
        batch_size=args.batch_size,
        n_train=args.n_train,
    )

    # Create trainer object
    trainer = pl.Trainer(
        default_root_dir=str("logs/sample_model"),
        max_epochs=args.num_epochs,
        reload_dataloaders_every_epoch=True,
        callbacks=[pl.callbacks.LearningRateMonitor()],
        gpus=1 if args.gpu else 0,
    )

    # Run the fitting
    trainer.fit(model)
