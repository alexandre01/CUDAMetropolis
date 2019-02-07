"""Simulated annealing beta schedulers"""
import torch
import math


class BetaScheduler(object):
    """
    Scheduler base class for the simulated annealing strategy.
    Any beta cooling strategy should inherit from this class and implement the get_beta method.
    """
    def __init__(self, init_beta):
        self.init_beta = init_beta
        self.beta = init_beta
        self.iteration = 0
        self.batch_size = None

    def step(self, energies):
        self.iteration += 1
        self.beta = self.get_beta(energies)

    def get_beta(self, energies):
        raise NotImplementedError()


class ConstantBetaScheduler(BetaScheduler):
    """
    A trivial cooling strategy where beta is kept constant.
    """
    def __init__(self, init_beta):
        super().__init__(init_beta)

    def get_beta(self, energies):
        return self.init_beta


class StepBetaScheduler(BetaScheduler):
    """
    A simple cooling strategy where beta is increased by a factor gamma every step_size iterations.
    """
    def __init__(self, init_beta, step_size, gamma):
        super().__init__(init_beta)
        self.step_size = step_size
        self.gamma = gamma

    def get_beta(self, energies):
        return self.init_beta * self.gamma ** (self.iteration // self.step_size)

