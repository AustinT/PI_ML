# PI_ML
A python script to perform a path integral monte carlo simulation of a single linear rotor. 
The algorithms are take from the MoRiBS-PIMC project:
https://github.com/pnroy/MoRiBS-PIMC

## Machine learning of rho function

Install the packages in `requirements.txt`.
Then run `python train_nn.py` with default parameters.
This will train a neural network to approximate rho_eep given
the current `linden.out` file.
The current approximation is ok but could be
improved by changing the network parameters.

The script `eval_nn.py` can be used to check
the error on randomly selected points.
It requires a model checkpoint as an argument,
which can be found in the `logs` directory after training.
