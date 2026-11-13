"""Default model definitions; add custom subscriptions in register_fitting_models.

Set ``VECTORISED=True`` when subscribing a model that accepts 2D ``x`` arrays
and row-wise parameters shaped as ``(n_rows, 1)``. Set it to ``False`` for
models that must be evaluated one row at a time.
"""

import numpy as np

from mvesuvio.util.model_manager import model_manager


def _gauss(x, A, x0, sigma):
    return A / (2 * np.pi) ** 0.5 / sigma * np.exp(-((x - x0) ** 2) / 2 / sigma**2)


def _gauss_cntr(x, A, sigma):
    return A / (2 * np.pi) ** 0.5 / sigma * np.exp(-(x**2) / 2 / sigma**2)


def _gcc4c6(x, A, x0, sigma1, c4, c6):
    return (
        A
        * np.exp(-((x - x0) ** 2) / 2 / sigma1**2)
        / (np.sqrt(2 * np.pi * sigma1**2))
        * (
            1
            + c4 / 32 * (16 * ((x - x0) / np.sqrt(2) / sigma1) ** 4 - 48 * ((x - x0) / np.sqrt(2) / sigma1) ** 2 + 12)
            + c6
            / 384
            * (
                64 * ((x - x0) / np.sqrt(2) / sigma1) ** 6
                - 480 * ((x - x0) / np.sqrt(2) / sigma1) ** 4
                + 720 * ((x - x0) / np.sqrt(2) / sigma1) ** 2
                - 120
            )
        )
    )


def _gcc4c6_cntr(x, A, sigma1, c4, c6):
    return (
        A
        * np.exp(-(x**2) / 2 / sigma1**2)
        / (np.sqrt(2 * np.pi * sigma1**2))
        * (
            1
            + c4 / 32 * (16 * (x / np.sqrt(2) / sigma1) ** 4 - 48 * (x / np.sqrt(2) / sigma1) ** 2 + 12)
            + c6
            / 384
            * (64 * (x / np.sqrt(2) / sigma1) ** 6 - 480 * (x / np.sqrt(2) / sigma1) ** 4 + 720 * (x / np.sqrt(2) / sigma1) ** 2 - 120)
        )
    )


def _gcc4(x, A, x0, sigma1, c4):
    return (
        A
        * np.exp(-((x - x0) ** 2) / 2 / sigma1**2)
        / (np.sqrt(2 * np.pi * sigma1**2))
        * (1 + c4 / 32 * (16 * ((x - x0) / np.sqrt(2) / sigma1) ** 4 - 48 * ((x - x0) / np.sqrt(2) / sigma1) ** 2 + 12))
    )


def _gcc4_cntr(x, A, sigma1, c4):
    return (
        A
        * np.exp(-(x**2) / 2 / sigma1**2)
        / (np.sqrt(2 * np.pi * sigma1**2))
        * (1 + c4 / 32 * (16 * (x / np.sqrt(2) / sigma1) ** 4 - 48 * (x / np.sqrt(2) / sigma1) ** 2 + 12))
    )


def _gcc6(x, A, x0, sigma1, c6):
    return (
        A
        * np.exp(-((x - x0) ** 2) / 2 / sigma1**2)
        / (np.sqrt(2 * np.pi * sigma1**2))
        * (
            1
            + c6
            / 384
            * (
                64 * ((x - x0) / np.sqrt(2) / sigma1) ** 6
                - 480 * ((x - x0) / np.sqrt(2) / sigma1) ** 4
                + 720 * ((x - x0) / np.sqrt(2) / sigma1) ** 2
                - 120
            )
        )
    )


def _gcc6_cntr(x, A, sigma1, c6):
    return (
        A
        * np.exp(-(x**2) / 2 / sigma1**2)
        / (np.sqrt(2 * np.pi * sigma1**2))
        * (
            1
            + c6
            / 384
            * (64 * (x / np.sqrt(2) / sigma1) ** 6 - 480 * (x / np.sqrt(2) / sigma1) ** 4 + 720 * (x / np.sqrt(2) / sigma1) ** 2 - 120)
        )
    )


def _doublewell(x, A, d, R, sig1, sig2):
    theta = np.linspace(0, np.pi, 300)[:, np.newaxis]
    y = x[np.newaxis, :]
    sigTH = np.sqrt(sig1**2 * np.cos(theta) ** 2 + sig2**2 * np.sin(theta) ** 2)
    alpha = 2 * (d * sig2 * sig1 * np.sin(theta) / sigTH) ** 2
    beta = (2 * sig1**2 * d * np.cos(theta) / sigTH**2) * y
    denom = 2.506628 * sigTH * (1 + R**2 + 2 * R * np.exp(-2 * d**2 * sig1**2))
    jp = np.exp(-(y**2) / (2 * sigTH**2)) * (1 + R**2 + 2 * R * np.exp(-alpha) * np.cos(beta)) / denom
    jp *= np.sin(theta)
    result = np.trapezoid(jp, x=theta, axis=0)
    result /= np.abs(np.trapezoid(result, x=y))
    result *= A
    return result


def _gauss2d(x, A, sig1, sig2):
    theta = np.linspace(0, np.pi, 300)[:, np.newaxis]
    y = x[np.newaxis, :]
    sigTH = np.sqrt(sig1**2 * np.cos(theta) ** 2 + sig2**2 * np.sin(theta) ** 2)
    jp = np.exp(-(y**2) / (2 * sigTH**2)) / (2.506628 * sigTH)
    jp *= np.sin(theta)
    result = np.trapezoid(jp, x=theta, axis=0)
    result /= np.abs(np.trapezoid(result, x=y))
    result *= A
    return result


def _gauss3d(x, A, sig_x, sig_y, sig_z):
    y = x[:, np.newaxis, np.newaxis]
    n_steps = 50
    theta = np.linspace(0, np.pi / 2, n_steps)[np.newaxis, :, np.newaxis]
    phi = np.linspace(0, np.pi / 2, n_steps)[np.newaxis, np.newaxis, :]
    s2_inv = (
        np.sin(theta) ** 2 * np.cos(phi) ** 2 / sig_x**2 + np.sin(theta) ** 2 * np.sin(phi) ** 2 / sig_y**2 + np.cos(theta) ** 2 / sig_z**2
    )
    result = np.sin(theta) / s2_inv * np.exp(-(y**2) / 2 * s2_inv)
    result = np.trapezoid(result, x=phi, axis=2)[:, :, np.newaxis]
    result = np.trapezoid(result, x=theta, axis=1)
    result *= A * 2 / np.pi * 1 / np.sqrt(2 * np.pi) * 1 / (sig_x * sig_y * sig_z)
    return result.squeeze()


_registered = False


def register_fitting_models() -> None:
    """Register built-in and user-added fitting models once per process."""
    global _registered
    if _registered:
        return

    model_manager.subscribe("GAUSS", _gauss, {"A": 1, "x0": 0, "sigma": 5}, ("sigma",), VECTORISED=True)
    model_manager.subscribe("GAUSS_CNTR", _gauss_cntr, {"A": 1, "sigma": 5}, ("sigma",), VECTORISED=True)
    model_manager.subscribe(
        "GCC4C6",
        _gcc4c6,
        {"A": 1, "x0": 0, "sigma1": 6, "c4": 0, "c6": 0},
        ("sigma1", "c4", "c6"),
        VECTORISED=True,
    )
    model_manager.subscribe("GCC4C6_CNTR", _gcc4c6_cntr, {"A": 1, "sigma1": 6, "c4": 0, "c6": 0}, ("sigma1", "c4", "c6"), VECTORISED=True)
    model_manager.subscribe("GCC4", _gcc4, {"A": 1, "x0": 0, "sigma1": 6, "c4": 0}, ("sigma1", "c4"), VECTORISED=True)
    model_manager.subscribe("GCC4_CNTR", _gcc4_cntr, {"A": 1, "sigma1": 6, "c4": 0}, ("sigma1", "c4"), VECTORISED=True)
    model_manager.subscribe("GCC6", _gcc6, {"A": 1, "x0": 0, "sigma1": 6, "c6": 0}, ("sigma1", "c6"), VECTORISED=True)
    model_manager.subscribe("GCC6_CNTR", _gcc6_cntr, {"A": 1, "sigma1": 6, "c6": 0}, ("sigma1", "c6"), VECTORISED=True)
    model_manager.subscribe(
        "DOUBLEWELL",
        _doublewell,
        {"A": 1, "d": 1, "R": 1, "sig1": 3, "sig2": 5},
        ("d", "R", "sig1", "sig2"),
        VECTORISED=False,
    )
    model_manager.subscribe("GAUSS2D", _gauss2d, {"A": 1, "sig1": 3, "sig2": 5}, ("sig1", "sig2"), VECTORISED=False)
    model_manager.subscribe(
        "GAUSS3D", _gauss3d, {"A": 1, "sig_x": 5, "sig_y": 5, "sig_z": 5}, ("sig_x", "sig_y", "sig_z"), VECTORISED=False
    )
    _registered = True
