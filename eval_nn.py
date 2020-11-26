""" Script to evaluate neural network """

import argparse
import numpy as np
import torch
import pytorch_lightning as pl

from train_nn import RhoMLP


def get_model_predictions(model: RhoMLP, n_pred, batch_size=1024):

    # Get data
    model.n_train = n_pred
    rand_nums, rho_vals = model._get_xy_data()

    # Query model
    model.eval()
    outputs = []
    with torch.no_grad():
        for i in range(0, n_pred, batch_size):
            x = torch.as_tensor(
                rand_nums[i : i + batch_size], device=model.device, dtype=torch.float
            )
            y_pred = model(x).cpu().numpy()
            outputs.append(y_pred)
    outputs = np.concatenate(outputs, axis=0).flatten()
    return rand_nums, rho_vals, outputs


if __name__ == "__main__":

    pl.seed_everything(0)

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--model_path", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument("-n", "--num_predictions", type=int, default=10000)
    args = parser.parse_args()

    # Load neural network
    model = RhoMLP.load_from_checkpoint(args.model_path, map_location="cpu")

    # Do queries
    x, y, y_pred = get_model_predictions(model, args.num_predictions)
    err = y - y_pred

    # Get stats
    print(f"Error stats with {args.num_predictions} samples")
    print("="*50)
    print(f"MSE: {np.mean(err**2):.2e}")
    print(f"MAE: {np.mean(np.abs(err)):.2e}")
    percentiles = [0, 25, 50, 75, 90, 99, 99.5, 99.9, 99.99, 100]
    print("Absolute Error Percentiles:")
    for p in percentiles:
        print(f"\t{p:>6.2f}%: {np.percentile(np.abs(err), p):.2e}")
