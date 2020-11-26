""" File to train neural network on path integral data """

import argparse
import numpy as np
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
        batch_size=32,
        n_train=32000,
        lr=1e-3,
        output_power: int = 1,
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
        self.output_power = output_power
        assert self.output_power >= 1  # less than 1 not supported
        self._apply_power = True

        # Training stuff
        self.batch_size = batch_size
        self.n_train = n_train
        self.loss_func = torch.nn.MSELoss()
        self.lr = lr

        # Data stuff
        self.linden_path = linden_path

    def forward(self, x):
        y = self.net(x)
        if self.output_power > 1 and self._apply_power:
            return torch.pow(y, self.output_power)
        else:
            return y

    def configure_optimizers(self):
        adam = torch.optim.Adam(self.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(adam, factor=0.3)
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

        # Apply power of NN
        if self.output_power > 1 and not self._apply_power:
            rho_vals = np.sign(rho_vals) * np.power(
                np.abs(rho_vals), 1 / self.output_power
            )

        # Convert to tensor dataset for serving
        rand_nums = torch.as_tensor(rand_nums, dtype=torch.float)
        rho_vals = torch.as_tensor(rho_vals.reshape(-1, 1), dtype=torch.float)
        dset = TensorDataset(rand_nums, rho_vals)
        return DataLoader(dset, batch_size=self.batch_size, drop_last=True)

    def train_dataloader(self):
        self._apply_power = False
        return self._get_dataloader()

    def val_dataloader(self):
        self._apply_power = True
        return self._get_dataloader()

    def _get_loss(self, batch):
        x, y = batch
        y_pred = self(x)
        return self.loss_func(y_pred, y)

    def training_step(self, batch, batch_idx):
        self._apply_power = False
        return self._get_loss(batch)

    def validation_step(self, batch, batch_idx):
        self._apply_power = True
        loss = self._get_loss(batch)
        self.log("loss/val", loss, prog_bar=True)
        return loss


if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 64, 32])
    parser.add_argument("-p", "--output_power", type=int, default=1)
    args = parser.parse_args()

    # Sample neural network
    pl.seed_everything(0)
    model = RhoMLP(
        hidden_sizes=args.hidden + [1],
        linden_path="linden.out",
        lr=1e-3,
        output_power=args.output_power,
    )

    # Create trainer object
    trainer = pl.Trainer(
        default_root_dir=str("logs/sample_model"),
        max_epochs=200,
        reload_dataloaders_every_epoch=True,
        callbacks=[pl.callbacks.LearningRateMonitor()],
    )

    # Run the fitting
    trainer.fit(model)
