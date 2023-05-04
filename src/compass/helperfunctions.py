import numpy as np
from astropy.modeling.functional_models import Gaussian1D, Gaussian2D
from matplotlib.patches import Ellipse


def parallax_projection(time, host_star):
    t = time
    t_eqx = 80 / 365.25
    eps = np.deg2rad(23.5)
    ra = np.deg2rad(host_star.ra)
    dec = np.deg2rad(host_star.dec)
    s_ra_0, s_dec_0 = 0, 0
    s_ra_i = (
        s_ra_0
        + np.sin(ra) * np.cos(2 * np.pi * (t - t_eqx))
        - np.cos(eps) * np.cos(ra) * np.sin(2 * np.pi * (t - t_eqx))
    )
    s_dec_i = (
        s_dec_0
        + np.cos(ra) * np.sin(dec) * np.cos(2 * np.pi * (t - t_eqx))
        - (np.sin(eps) * np.cos(dec) - np.cos(eps) * np.sin(ra) * np.sin(dec))
        * np.sin(2 * np.pi * (t - t_eqx))
    )
    return s_ra_i, s_dec_i


def get_ellipse_props(cov, confidence):
    """
    Calculates the eigenvalues/eigenvectors of the covariance matrix
    and returns the major and minor axis length of the ellipse.
        Args:
            cov (numpy.array (2x2)): Covariance matrix.
            confidence (float): Probability between 0 and 1.
        Returns:
            major_axis (float): Semi-major axis.
            minor_axis (float): Semi-minor axis.
            angle (float): Angle of rotation of the ellipse.
    """
    # Compute the value of the chi-squared distribution
    # that corresponds to the
    # desired level of confidence
    r_prime = np.sqrt(-2 * np.log(1 - confidence))
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    major_axis = np.sqrt(eigenvalues[0]) * r_prime
    minor_axis = np.sqrt(eigenvalues[1]) * r_prime
    angle = np.arctan(eigenvectors[1, 0] / eigenvectors[1, 1])
    return major_axis, minor_axis, angle


def ellipse(x_mean, y_mean, cov, color, linestyle, axis):
    for confd in [0.5, 0.9, 0.99]:
        major, minor, angle = get_ellipse_props(cov, confidence=confd)
        add_ellp_patch2(
            x_mean,
            y_mean,
            major,
            minor,
            angle,
            color=color,
            linestyle=linestyle,
            axis=axis,
        )


def add_ellp_patch(pdf, major, minor, angle, color, axis):
    """
    Adds a patch to a figure. The patch includes a
    ellipse at the means of a pdf with a scalable
    magnitude in the both axis directions.
    Args:
        pdf (astropy.modeling.Gaussian2D): 2D Gaussian function.
        eigvalue (list): List of eigenvalues of the cov matrix.
        angle (float): Angle of the eigenvectors in radian.
        i (float): Scaling the magnitude of the eigenvalues.
        axis (figure axis): E.g. plt, axs[1,0]
    """
    axis.add_patch(
        Ellipse(
            (pdf.x_mean.value, pdf.y_mean.value),
            width=2 * major,
            height=2 * minor,
            angle=360 * angle / (2 * np.pi),
            facecolor="none",
            edgecolor=color,
            linewidth=1,
            # label=f'{i*100:.0f}'
        )
    )


def add_ellp_patch2(x_mean, y_mean, major, minor, angle, color, linestyle, axis):
    """
    Adds a patch to a figure. The patch includes a
    ellipse at the means of a pdf with a scalable
    magnitude in the both axis directions.
    Args:
        pdf (astropy.modeling.Gaussian2D): 2D Gaussian function.
        eigvalue (list): List of eigenvalues of the cov matrix.
        angle (float): Angle of the eigenvectors in radian.
        i (float): Scaling the magnitude of the eigenvalues.
        axis (figure axis): E.g. plt, axs[1,0]
    """
    axis.add_patch(
        Ellipse(
            (x_mean, y_mean),
            width=2 * major,
            height=2 * minor,
            angle=360 * angle / (2 * np.pi),
            facecolor="none",
            edgecolor=color,
            linewidth=1,
            linestyle=linestyle,
            alpha=0.6
            # label=f'{i*100:.0f}'
        )
    )


def convert_df_to_array(df, x_col_name, y_col_name):
    """
    Convert dataframe to two 1D arrays.
    """
    data = df[[x_col_name, y_col_name]].dropna()
    x_data = data[x_col_name].values
    y_data = data[y_col_name].values
    return x_data, y_data


def gaussian1D(g2d_astropy_func, x_or_y):
    stddev = f"{x_or_y}_stddev"
    mean = f"{x_or_y}_mean"
    g1d = Gaussian1D(
        (np.sqrt(2 * np.pi) * g2d_astropy_func.__getattribute__(stddev)) ** (-1),
        g2d_astropy_func.__getattribute__(mean),
        g2d_astropy_func.__getattribute__(stddev),
    )
    return g1d


def get_g2d_parameters(x_data, y_data):
    """
    Calculate 2D Gaussian parameters based on two arrays.
    """
    x_mean = np.mean(x_data)
    y_mean = np.mean(y_data)
    x_stddev = np.std(x_data)
    y_stddev = np.std(y_data)
    cov = np.cov(m=[x_data, y_data])
    rho = cov[1, 0] / (x_stddev * y_stddev)
    if rho < 1 and rho > -1:
        amplitude = 2 * np.pi * x_stddev * y_stddev * np.sqrt(1 - rho**2)
    else:
        amplitude = 2 * np.pi * x_stddev * y_stddev
    return amplitude, x_mean, y_mean, x_stddev, y_stddev, cov, rho


def get_g2d_func(x_mean, y_mean, x_std, y_std, rho):
    cov = np.array(
        [[x_std**2, rho * x_std * y_std], [rho * x_std * y_std, y_std**2]]
    )
    denom = 2 * np.pi * np.linalg.det(cov) ** (1 / 2)
    g2d = Gaussian2D(amplitude=1 / denom, x_mean=x_mean, y_mean=y_mean, cov_matrix=cov)
    return g2d, cov


def convolution2d(a, b, A, B):
    """
    Returns a Gaussian2D functions and its covariance matrix
    of the convolution from the two Gaussian2D.
    Args:
        a (numpy.ndarray): mean 2x1 vector of the first Gaussian2D.
        b (numpy.ndarray): mean 2x1 vector of the second Gaussian2D.
        A (numpy.ndarray): Covariance 2x2 matrix of the first Gaussian2D.
        B (numpy.ndarray): Covariance 2x2 matrix of the second Gaussian2D.
    """
    x_mean = (a + b)[0]
    y_mean = (a + b)[1]
    cov_c = A + B
    denom = 2 * np.pi * np.linalg.det(cov_c) ** (1 / 2)
    g2d_c = Gaussian2D(
        amplitude=1 / denom, x_mean=x_mean, y_mean=y_mean, cov_matrix=cov_c
    )
    return g2d_c, cov_c


def calc_prime_1(prime_0, pm, plx, time, plx_proj):
    prime_1 = prime_0 + time * pm + plx * plx_proj
    return prime_1


def func_exp(x, a, b, c):
    return a * np.exp(-b * x) + c


def func_exp_inc(x, a, b, c):
    return a * np.exp(b * x) + c


def func_lin(x, a, b):
    return a * x + b


def func_const(x, a):
    return a


def n_dim_gauss_evaluated(obs, mean, cov):
    denom = (2 * np.pi) ** (len(mean) / 4) * np.linalg.det(cov) ** (1 / 2)
    expo = np.dot(np.dot((obs - mean).T, np.linalg.inv(cov)), (obs - mean))
    return np.exp(-1 / 2 * expo) / denom
