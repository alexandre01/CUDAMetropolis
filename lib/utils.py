"""Some simple utility functions."""

import torch
import math
import matplotlib.pyplot as plt


def plot_errors(errors):
    if isinstance(errors, torch.Tensor):
        errors = errors.cpu().numpy()

    plt.figure()
    plt.title("Error $e(\widehat{x},X)$ as a function of time")
    plt.plot(errors)
    plt.xlabel("Time t")
    plt.ylabel("Error $e(\widehat{x},X)$")
    plt.show()


def plot_errors_energies(errors, energies):
    if isinstance(energies, torch.Tensor):
        energies = energies.cpu().numpy()

    plt.figure()
    plt.title("Energy $E(\widehat{x},Y)$ as a function of time")
    plt.plot(energies)
    plt.xlabel("Time t")
    plt.ylabel("Energy $E(\widehat{x},Y)$")
    plt.grid()
    plt.show()

    if isinstance(errors, torch.Tensor):
        errors = errors.cpu().numpy()

    plt.figure()
    plt.title("Error $e(\widehat{x},X)$ as a function of time")
    plt.plot(errors)
    plt.xlabel("Time t")
    plt.ylabel("Error $e(\widehat{x},X)$")
    plt.grid()
    plt.show()

    plt.figure()
    plt.title("Error $e(\widehat{x},X)$ as a function of $E(\widehat{x},Y)$")
    plt.plot(energies, errors)
    plt.xlabel("Energy $E(\widehat{x},Y)$")
    plt.ylabel("Error $e(\widehat{x},X)$")
    plt.grid()
    plt.show()

    
def plot_errors_energies_average(errors, energies):
    if isinstance(energies, torch.Tensor):
        energies_avg = energies.mean(1).cpu().numpy()
        energies_std = energies.std(1).cpu().numpy()
    
    plt.figure()
    plt.title("Energy $E(\widehat{x},Y)$ as a function of time")
    plt.plot(energies_avg)
    plt.fill_between(range(len(energies_avg)),
                     energies_avg-energies_std,
                     energies_avg+energies_std,
                     alpha = 0.4)
    plt.xlabel("Time t")
    plt.ylabel("Energy $E(\widehat{x},Y)$")
    plt.grid()
    plt.show()

    if isinstance(errors, torch.Tensor):
        errors_avg = errors.mean(1).cpu().numpy()
        errors_std = errors.std(1).cpu().numpy()
    
    plt.figure()
    plt.title("Error $e(\widehat{x},X)$ as a function of time")
    plt.plot(errors_avg)
    plt.fill_between(range(len(errors_avg)),
                     errors_avg-errors_std,
                     errors_avg+errors_std,
                     alpha = 0.4)
    plt.xlabel("Time t")
    plt.ylabel("Error $e(\widehat{x},X)$")
    plt.grid()
    plt.show()

   
    
"""Standard utility functions."""

def compute_reconstruction_error(x, X):
    """
    Args:
        x: shape (N,)

    Returns:
        The batch-wise reconstruction error
    """
    N = len(X)

    return (x != X).sum().float() / N


def compute_energy(x, W, Y):
    """
    Args:
        x: shape (N,)

    Returns:
        The computed energy H_Y(x) and the result of Wx.
    """
    N = len(x)

    wx = torch.matmul(W, x)
    y = wx.relu() / math.sqrt(N)
    return (Y - y).pow(2).sum(), wx


def compute_energy_diff(wx, x, idx, W, Y):
    """
    Efficiently compute  H_Y(x^(t+1)) using Wx^(t) from the computation of H_Y(x^(t))-

    Args:
        wx: the result of the previously computed Wx^(t), shape (M,)
        x: x^(t), shape (N,)
        idx: the coordinate i of x^(t) which has been flipped

    Returns:
        The computed energy H_Y(x) and the result of Wx^(t+1).
    """
    N = len(x)

    wx_next = wx - 2 * W[:, idx] * x[idx]
    y = wx_next.relu() / math.sqrt(N)
    return (Y - y).pow(2).sum(), wx_next




"""Identical (but batch-wise) utility functions."""

def compute_reconstruction_error_batch(x_batch, X):
    """
    Args:
        x_batch: shape (N, batch_size)

    Returns:
        The batch-wise reconstruction error: shape (batch_size,)
    """
    N = len(X)

    return (x_batch != X.unsqueeze(1)).sum(dim=0).float() / N


def compute_energy_batch(x_batch, W, Y):
    """
    Args:
        x_batch: shape (N, batch_size)
    
    Returns:
        The computed energy H_Y(x) (shape (batch_size,)) and the result of Wx.
    """
    N = len(x_batch)

    wx = torch.matmul(W, x_batch)
    y = wx.relu() / math.sqrt(N)
    return (Y.unsqueeze(1) - y).pow(2).sum(dim=0), wx


def compute_energy_diff_batch(wx, x_batch, idx_batch, W, Y):
    """
    Efficiently compute  H_Y(x^(t+1)) using Wx^(t) from the computation of H_Y(x^(t))-

    Args:
        wx: the result of the previously computed Wx^(t), shape (M, batch_size)
        x: x^(t), shape (N, batch_size)
        idx_batch: the coordinate i (batch-wise) of x^(t) which has been flipped, shape (batch_size)
    
    Returns:
        The computed energy H_Y(x) (shape (batch_size,)) and the result of Wx^(t+1).
    """
    N = len(x_batch)

    wx_next = wx - 2 * W[:, idx_batch] * x_batch.gather(0, idx_batch.unsqueeze(0))
    y = wx_next.relu() / math.sqrt(N)
    return (Y.unsqueeze(1) - y).pow(2).sum(dim=0), wx_next
