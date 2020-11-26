""" Misc utility functions """

import numpy as np
from scipy import interpolate


def get_xyz(z, phi):
    """ get xyz vectors given z, phi """
    sint = np.sqrt(1.0 - z ** 2)
    x = sint * np.cos(phi)
    y = sint * np.sin(phi)
    return np.array([x, y, z])


def get_rho_eep_func(linden_path: str):
    """
    given data in a "linden" file, return a function that uses interpolation
    to calculate rho_eep(z1, phi1, z2, phi2)
    """

    # Read in contents of file
    linden = np.loadtxt(linden_path)
    cos_gamma = linden[:, 0]
    rho_arr = linden[:, 1]

    # Create interpolation function
    rho_f = interpolate.interp1d(cos_gamma, rho_arr, kind="cubic")

    # Function to correctly interpolate rho_eep elements
    def rho_eep(z1, phi1, z2, phi2):

        # calculate xyz vectors
        v1 = get_xyz(z1, phi1)
        v2 = get_xyz(z2, phi2)

        # Dot product (to get cos(gamma))
        dot = np.sum(v1 * v2, axis=0)
        dot = np.minimum(dot, 1.0)  # avoid numerical index errors
        return rho_f(dot)

    return rho_eep


def get_random_z_phi(n_data):
    """ sample random (z1, phi1, z2, phi2) vector uniformly """
    rand_nums = np.random.random(size=(n_data, 4))

    # convert to [-1, +1]
    rand_nums = 2 * rand_nums - 1

    # convert angles
    for i in [1, 3]:
        rand_nums[:, i] *= np.pi

    return rand_nums
