import numpy as np


def compute_cost_func1(alpha, beta, time, distance):
    return 1 / np.exp(alpha * time**beta)

