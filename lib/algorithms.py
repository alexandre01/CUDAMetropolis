import torch
from . import utils


class Algorithm(object):
    """
    Algorithm base class for the Markov Chain Monte-Carlo method.
    Any algorithm should inherit from this class and implement the step method.
    """
    def step(self, x, W, Y, beta, energy, wx, idx, rand):
        """
        Args:
            x: x^(t), shape (N,)
            beta (float): the inverse temperature
            energy (float): the energy value
            wx: the value of the previously computed Wx^(t)
            idx (float): the index at which we flip x^(t)
            rand (float): the random value that will be compared to the acceptance probability
            
        Returns:
            the updated x, energy and wx
        """
        raise NotImplementedError()

    def step_batch(self, x_batch, W, Y, beta, energy_batch, wx, idx_batch, rand_batch):
        """
        Batch-wise version of step.
        
        Args:
            x_batch: x^(t), shape (N, batch_size)
            beta (float): the inverse temperature
            energy_batch: the energy value (batch-wise), shape (batch_size)
            wx: the value of the previously computed Wx^(t)
            idx_batch: the index (batch-wise) at which we flip x^(t), shape (batch_size)
            rand_batch: the random value (batch-wise) that will be compared to the acceptance probability, shape (batch_size)

        Returns:
            the updated x_batch, energy_batch and wx
        """
        raise NotImplementedError()


class Metropolis(Algorithm):

    def step(self, x, W, Y, beta, energy, wx, idx, rand):
        energy_next, wx_next = utils.compute_energy_diff(wx, x, idx, W, Y)                      # Compute the new energy value if we would update x^(t)

        a = torch.exp(-beta * (energy_next - energy)).clamp(max=1.0)                            # Compute the acceptance probability

        if rand < a:
            x[idx] *= -1
            energy = energy_next
            wx = wx_next

        return x, energy, wx

    def step_batch(self, x_batch, W, Y, beta, energy_batch, wx, idx_batch, rand_batch):
        energy_next, wx_next = utils.compute_energy_diff_batch(wx, x_batch, idx_batch, W, Y)    # Compute the new energy value if we would update x^(t)

        a = torch.exp(-beta * (energy_next - energy_batch)).clamp(max=1.0)                      # Compute the acceptance probability (batch-wise)
        accepted_batches = rand_batch < a                                                       # Which batches pass the acceptance test?

        x_batch[idx_batch[accepted_batches], accepted_batches.nonzero().squeeze(-1)] *= -1      # Update x^(t) for the batches that pass the acceptance test
        energy_batch[accepted_batches] = energy_next[accepted_batches]                          # Update the energy value for the batches that pass the acceptance test
        wx[:, accepted_batches] = wx_next[:, accepted_batches]                                  # Update Wx^(t) value for the batches that pass the acceptance test

        return x_batch, energy_batch, wx


class Glauber(Algorithm):

    def step(self, x, W, Y, beta, energy, wx, idx, rand):
        energy_next, wx_next = utils.compute_energy_diff(wx, x, idx, W, Y)                      # Compute the new energy value if we would update x^(t)

        a = (1 - torch.tanh(beta * (energy_next - energy))) / 2                                  # Compute the acceptance probability

        if rand < a:
            x[idx] *= -1
            energy = energy_next
            wx = wx_next

        return x, energy, wx

    def step_batch(self, x_batch, W, Y, beta, energy_batch, wx, idx_batch, rand_batch):
        energy_next, wx_next = utils.compute_energy_diff_batch(wx, x_batch, idx_batch, W, Y)    # Compute the new energy value if we would update x^(t)

        a = (1 - torch.tanh(beta * (energy_next - energy_batch))) / 2                           # Compute the acceptance probability
        accepted_batches = rand_batch < a                                                       # Which batches pass the acceptance test?

        x_batch[idx_batch[accepted_batches], accepted_batches.nonzero().squeeze(-1)] *= -1      # Update x^(t) for the batches that pass the acceptance test
        energy_batch[accepted_batches] = energy_next[accepted_batches]                          # Update the energy value for the batches that pass the acceptance test
        wx[:, accepted_batches] = wx_next[:, accepted_batches]                                  # Update Wx^(t) value for the batches that pass the acceptance test

        return x_batch, energy_batch, wx
