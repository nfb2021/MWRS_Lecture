import numpy as np


def gauss2d(x=0., y=0., mx=0., my=0., sx=1., sy=1.):
    """
    Computes a probability from user-defined 2D Gaussian probability distribution.

    Parameters
    ----------
    x: float or np.ndarray, optional
        x coordinate (0 is default).
    y: float or np.ndarray, optional
        y coordinate (0 is default).
    mx: float, optional
        mean value in x direction (0 is default).
    my: float, optional
        mean value in y direction (0 is default).
    sx: float, optional
        standard deviation in x direction (1 is default).
    sy: float, optional
        standard deviation in y direction (1 is default).

    Returns
    -------
    float
        Probability value.
    """
    return 1. / (2. * np.pi * sx * sy) * np.exp(-((x - mx) ** 2. / (2. * sx ** 2.) + (y - my) ** 2. / (2. * sy ** 2.)))


def _data2list(data):
    """
    Wraps a list around an object.

    Parameters
    ----------
    data: object
        Input data.

    Returns
    -------
    list
        Input data as list.
    """
    if not isinstance(data, list):
        data = [data]

    return data


def generate_dem(x_means=None, y_means=None, x_stds=None, y_stds=None,
                 x_min=-500., x_max=500., n_x=1000, y_min=-500., y_max=500., n_y=1000, z_min=0., z_max=140.):

    """
    Generates a Digital Elevation Model (DEM) from a superposition of hills/2D Gaussian PDF's.
    If no values are specified a default DEM will be generated.

    Parameters
    ----------
    x_means: float or list, optional
        List of center positions of a hill in x direction.
    y_means: float or list, optional
        List of center positions of a hill in y direction.
    x_stds: float or list, optional
        List of standard deviations of a hill in x direction.
    y_stds: float or list, optional
        List of standard deviations of a hill in y direction.
    x_min: float, optional
        Minimum x extent of the DEM (default is -500).
    x_max: : float, optional
        Maximum x extent of the DEM (default is 500).
    n_x: int, optional
        Number of x coordinates (default is 1000).
    y_min: float, optional
        Minimum y extent of the DEM (default is -500).
    y_max: float, optional
        Maximum y extent of the DEM (default is 500).
    n_y: int, optional
        Number of y coordinates (default is 1000).
    z_min: float, optional
        Minimum z extent of the DEM (default is 0).
    z_max: float, optional
        Maximum z extent of the DEM (default is 140).

    Returns
    -------
    x, y, z: np.array, np.array, np.array
        3D coordinates where each coordinate is represented as a 2D NumPy array.
    """
    if x_means is None:
        x_means = [-100, 300, -400, 400]
    else:
        x_means = _data2list(x_means)

    if y_means is None:
        y_means = [-100, 300, 200, -200]
    else:
        y_means = _data2list(y_means)

    if x_stds is None:
        x_stds = [100, 150, 125, 175]
    else:
        x_stds = _data2list(x_stds)

    if y_stds is None:
        y_stds = [100, 150, 125, 175]
    else:
        y_stds = _data2list(y_stds)

    xs = np.linspace(x_min, x_max, n_x)  # [m]
    ys = np.linspace(y_min, y_max, n_y)  # [m]
    x, y = np.meshgrid(xs, ys)

    probs = 0.
    for i in range(len(x_means)):
        x_mean = x_means[i]
        y_mean = y_means[i]
        x_std = x_stds[i]
        y_std = y_stds[i]
        probs += gauss2d(x, y, mx=x_mean, my=y_mean, sx=x_std, sy=y_std)

    z = ((probs - np.min(probs)) / (np.max(probs) - np.min(probs))) * (z_max - z_min) + z_min

    return x, y, z


if __name__ == '__main__':
    x, y, z = generate_dem()
