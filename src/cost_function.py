

def compute_cost_func1(alpha, beta, time, distance):
    return alpha * time


def compute_cost_func2(alpha, beta, time, distance):
    return alpha * (time**beta)


def compute_cost_func3(alpha, beta, time, distance):
    return alpha * (time**beta) * distance

